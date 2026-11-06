from __future__ import annotations
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.agents.llm_client import ConfiguredLLMClient
from src.runner.pilot_runner import CellSpec, run_pilot_slice
from src.evaluation.jsonl_logger import JSONLWriter
from src.evaluation.runner import EpisodeContext, new_run_id
from src.environments import ENV_REGISTRY
HERE = Path(__file__).resolve().parent
LOG_DIR = ROOT / 'data' / 'raw_logs'
COST_DIR = ROOT / 'data' / 'costs'
GRIDS = {'gpt-5.4-mini': [dict(env='graph_nav', sc_vals=[25, 30, 35], dd_vals=[1, 2, 3, 4]), dict(env='tool_dag', sc_vals=[5, 7, 8, 10], dd_vals=[3, 4, 5, 6])], 'claude-haiku-4-5': [dict(env='graph_nav', sc_vals=[6, 7, 8, 9], dd_vals=[1, 2, 3, 4]), dict(env='tool_dag', sc_vals=[5, 7, 8, 10], dd_vals=[3, 4, 5, 6])]}
N_EP = 15
T_DEF = 40

def cell_key(env, sc, dd, model):
    return f'{env}|sc{sc}|dd{dd}|{model}'

def build_cells(env_name, sc, dd, model, n_ep):
    cells = []
    for ep_idx in range(n_ep):
        seed = hash(f'fine:{env_name}:{sc}:{dd}:{model}:{ep_idx}') & 2147483647
        stress = {'state_card': sc, 'dep_density': dd, 'T': T_DEF, 'branching': 4, 'obs_noise': 'clean', 'mut_rate': 'static'}
        tid = f'fine_{env_name}_sc{sc}_dd{dd}_{model}_ep{ep_idx:03d}'
        cells.append(CellSpec(env_name=env_name, model=model, stress_config=stress, task_config={'task_id': tid, 'stress_config': stress}, task_seed=seed, decoding_seed=seed ^ 51966, world_regime='finegrid', task_id=tid))
    return cells

def load_done(path):
    if path.exists():
        try:
            return set(json.loads(path.read_text()))
        except:
            pass
    return set()

def save_done(path, done):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(done), indent=2))

def print_heatmap(results, env, model):
    rows = {(sc, dd): v for (e, m, sc, dd), v in results.items() if e == env and m == model}
    if not rows:
        return
    sc_vals = sorted(set((k[0] for k in rows)))
    dd_vals = sorted(set((k[1] for k in rows)))
    print(f'\n[{env}] [{model}]  fine-grid')
    print('  sc\\dd  ' + ''.join((f'  dd={d}' for d in dd_vals)))
    for sc in sc_vals:
        row = f'  sc={sc:<4}'
        for dd in dd_vals:
            v = rows.get((sc, dd))
            if v and len(v) == N_EP:
                p = sum(v) / len(v)
                mk = '***' if p < 0.1 else '↓↓' if p < 0.3 else '↓' if p < 0.6 else '  '
                row += f'  {p:.2f}{mk}'
            elif v:
                row += f'  ({sum(v)}/{len(v)})'
            else:
                row += '    — '
        print(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='', help='gpt-5.4-mini or claude-haiku-4-5 (default: both)')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    models = [args.model] if args.model else list(GRIDS)
    client = ConfiguredLLMClient(timeout=120.0)
    results: dict = defaultdict(list)
    for model in models:
        done_path = HERE / f"finegrid_done_{model.replace('-', '_')}.json"
        ep_path = LOG_DIR / f"finegrid_{model.replace('-', '_')}_episode.jsonl"
        step_path = LOG_DIR / f"finegrid_{model.replace('-', '_')}_step.jsonl"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        done = load_done(done_path)
        grids = GRIDS[model]
        total = sum((len(g['sc_vals']) * len(g['dd_vals']) for g in grids))
        skip = sum((1 for g in grids for sc in g['sc_vals'] for dd in g['dd_vals'] if cell_key(g['env'], sc, dd, model) in done))
        print(f'\n=== {model} === ({total} cells, {skip} already done)')
        with JSONLWriter(ep_path) as ep_w, JSONLWriter(step_path) as st_w:
            for grid in grids:
                env_name = grid['env']
                for sc in grid['sc_vals']:
                    for dd in grid['dd_vals']:
                        key = cell_key(env_name, sc, dd, model)
                        if key in done:
                            print(f'  SKIP {key}')
                            continue
                        print(f'  → {key}  (n={N_EP})')
                        cells = build_cells(env_name, sc, dd, model, N_EP)
                        from src.runner.cost_tracker import CostTracker
                        ct = CostTracker(out_path=COST_DIR / 'finegrid_cost.jsonl', phase='finegrid', slice_name=key, emit_every=10)
                        outcomes = run_pilot_slice(cells=cells, client=client, step_jsonl_path=step_path, episode_jsonl_path=ep_path, cost_tracker=ct, n_workers=args.workers)
                        succ = sum((1 for o in outcomes if o.success))
                        p = succ / max(len(outcomes), 1)
                        results[env_name, model, sc, dd] = [o.success for o in outcomes]
                        done.add(key)
                        save_done(done_path, done)
                        mk = '***' if p < 0.1 else '↓↓' if p < 0.3 else '↓' if p < 0.6 else ''
                        print(f'     p̂={p:.3f} ({succ}/{len(outcomes)}) {mk}')
    print('\n' + '=' * 60)
    print('FINE-GRID RESULTS')
    for model in models:
        for env in ['graph_nav', 'tool_dag']:
            print_heatmap(results, env, model)
if __name__ == '__main__':
    main()
