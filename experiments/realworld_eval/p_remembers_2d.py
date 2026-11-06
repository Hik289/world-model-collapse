import json, glob, os, sys
sys.path.insert(0, '.')
sys.path.insert(0, 'experiments/trace_evaluation')
import trace_v1_metrics as T
from src.evaluation.runner import world_state_facts
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
KS = [1, 2, 3, 5, 7, 10, 15, 20]
THRESH = 0.3
FIGDIR = 'experiments/realworld_eval/runs/sw_cliff_grid/figures'

def episode_retention_by_k(ep_dir, subgoal_ids):
    ledger = T.ObservedLedger()
    steps = [json.loads(l) for l in open(ep_dir + 'steps.jsonl') if l.strip()]
    per_k = defaultdict(list)
    for rec in steps:
        t = rec['step']
        agent = T.filter_belief_facts(world_state_facts(rec['updater']['canonical_state']), subgoal_ids)
        gold = T.filter_belief_facts(world_state_facts(rec['evaluator_only']['gold_before']), subgoal_ids)
        ledger.update(rec['model_visible']['observation_partial_state'], t, subgoal_ids)
        lt = ledger.facts & gold
        for k in KS:
            elig = {f for f in lt if t - ledger.last_seen.get(f, t) >= k and (not ledger.is_layout(f))}
            if elig:
                per_k[k].append(len(agent & elig) / len(elig))
    return {k: sum(v) / len(v) if v else None for k, v in per_k.items()}

def build_surface(run_dir):
    cell = defaultdict(list)
    n_ep = 0
    for ep_dir in glob.glob(f'{run_dir}/raw/*/'):
        if not os.path.exists(ep_dir + 'manifest.json') or not os.path.exists(ep_dir + 'steps.jsonl'):
            continue
        mani = json.load(open(ep_dir + 'manifest.json'))
        Th = mani['horizon']
        r = episode_retention_by_k(ep_dir, set(mani['public_task_spec']))
        n_ep += 1
        for k in KS:
            if r.get(k) is not None:
                cell[k, Th].append(r[k])
    return (cell, n_ep)

def main():
    models = [('haiku', 'runs/sw_grid_large_haiku'), ('sonnet', 'runs/sw_grid_large_sonnet'), ('opus', 'runs/sw_grid_large_opus')]
    surfaces = {}
    for name, rel in models:
        rd = f'experiments/realworld_eval/{rel}'
        if not os.path.isdir(rd + '/raw'):
            continue
        cell, n_ep = build_surface(rd)
        if n_ep == 0:
            continue
        surfaces[name] = cell
        TS = sorted({t for k, t in cell})
        print(f'\n=== {name}: P(retention@k > {THRESH}) surface (n_ep={n_ep}) ===')
        print('k\\T ' + ' '.join((f'{t:>5}' for t in TS)))
        for k in KS:
            row = []
            for t in TS:
                vs = cell.get((k, t), [])
                row.append(f'{sum((1 for v in vs if v > THRESH)) / len(vs):.2f}' if vs else '  -  ')
            print(f'{k:>3} ' + ' '.join((f'{x:>5}' for x in row)))
    if not surfaces:
        print('no large-grid data yet')
        return
    n = len(surfaces)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for ax, (name, cell) in zip(axes[0], surfaces.items()):
        TS = sorted({t for k, t in cell})
        M = np.full((len(KS), len(TS)), np.nan)
        for i, k in enumerate(KS):
            for j, t in enumerate(TS):
                vs = cell.get((k, t), [])
                if vs:
                    M[i, j] = sum((1 for v in vs if v > THRESH)) / len(vs)
        im = ax.imshow(M, aspect='auto', cmap='viridis', vmin=0, vmax=1, origin='lower')
        ax.set_xticks(range(len(TS)))
        ax.set_xticklabels(TS)
        ax.set_yticks(range(len(KS)))
        ax.set_yticklabels(KS)
        ax.set_xlabel('horizon T')
        ax.set_ylabel('k (fact age)')
        ax.set_title(f'{name}: P(remembers)')
        for i in range(len(KS)):
            for j in range(len(TS)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6, color='white')
        fig.colorbar(im, ax=ax, label='P(ret@k>0.3)')
    plt.tight_layout()
    out = f'{FIGDIR}/p_remembers_2d_crossmodel.png'
    plt.savefig(out, dpi=140)
    print(f'\n[p_remembers_2d] -> {out}')
if __name__ == '__main__':
    main()
