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
run_dir = sys.argv[1]
label = sys.argv[2]
KS = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30]
byT = defaultdict(lambda: defaultdict(list))
for ep_dir in glob.glob(f'{run_dir}/raw/*/'):
    if not os.path.exists(ep_dir + 'steps.jsonl') or not os.path.exists(ep_dir + 'manifest.json'):
        continue
    manifest = json.load(open(ep_dir + 'manifest.json'))
    Th = manifest['horizon']
    subgoal_ids = set(manifest['public_task_spec'])
    ledger = T.ObservedLedger()
    steps = [json.loads(l) for l in open(ep_dir + 'steps.jsonl') if l.strip()]
    per_ep = defaultdict(list)
    for rec in steps:
        t = rec['step']
        agent = T.filter_belief_facts(world_state_facts(rec['updater']['canonical_state']), subgoal_ids)
        gold = T.filter_belief_facts(world_state_facts(rec['evaluator_only']['gold_before']), subgoal_ids)
        ledger.update(rec['model_visible']['observation_partial_state'], t, subgoal_ids)
        ledger_true = ledger.facts & gold
        for k in KS:
            elig = {f for f in ledger_true if t - ledger.last_seen.get(f, t) >= k and (not ledger.is_layout(f))}
            if elig:
                per_ep[k].append(len(agent & elig) / len(elig))
    for k in KS:
        if per_ep[k]:
            byT[Th][k].append(sum(per_ep[k]) / len(per_ep[k]))
TS = sorted(byT.keys())
M = np.full((len(KS), len(TS)), np.nan)
N = np.zeros((len(KS), len(TS)), dtype=int)
for j, Th in enumerate(TS):
    for i, k in enumerate(KS):
        vs = byT[Th][k]
        if vs:
            M[i, j] = sum(vs) / len(vs)
            N[i, j] = len(vs)
print(f'=== {label} retention 2D surface (rows=k, cols=T) ===')
print('k\\T ' + ' '.join((f'{Th:>5}' for Th in TS)))
for i, k in enumerate(KS):
    row = ' '.join((f'{M[i, j]:.2f}' if not np.isnan(M[i, j]) else '  -  ' for j in range(len(TS))))
    print(f'{k:>3} {row}')
plt.figure(figsize=(10, 6))
im = plt.imshow(M, aspect='auto', cmap='RdYlGn', vmin=0, vmax=0.6, origin='lower')
plt.colorbar(im, label='Retention@k')
plt.xticks(range(len(TS)), TS)
plt.yticks(range(len(KS)), KS)
plt.xlabel('horizon T (episode length)', fontsize=12)
plt.ylabel('k  (fact age in steps)', fontsize=12)
plt.title(f'{label}: retention 2D surface (k × T)\na cliff hidden in 1D shows up as a boundary here')
for i in range(len(KS)):
    for j in range(len(TS)):
        if not np.isnan(M[i, j]):
            plt.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6, color='black')
plt.tight_layout()
out = f'experiments/realworld_eval/runs/sw_cliff_grid/figures/retention_2d_{label}.png'
plt.savefig(out, dpi=140)
print(f'[retention_surface_2d] {label} -> {out}')
