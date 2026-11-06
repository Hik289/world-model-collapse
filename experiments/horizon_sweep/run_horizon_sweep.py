from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.agents.llm_client import ConfiguredLLMClient
from src.runner.pilot_runner import CellSpec, run_pilot_slice
from src.runner.cost_tracker import CostTracker
from src.evaluation.jsonl_logger import JSONLWriter
HERE = Path(__file__).resolve().parent
LOG_DIR = ROOT / 'data' / 'raw_logs'
COST_DIR = ROOT / 'data' / 'costs'
T_VALS = [5, 10, 15, 20, 30, 40, 60, 80, 100]
N_EP = 15
ANCHORS = {'gpt-5.4-mini': [('graph_nav', 20, 1), ('graph_nav', 20, 2), ('tool_dag', 5, 3), ('tool_dag', 5, 4), ('stateful_puzzle', 20, 1)], 'claude-haiku-4-5': [('graph_nav', 10, 1), ('graph_nav', 10, 3), ('tool_dag', 5, 3), ('tool_dag', 5, 4), ('stateful_puzzle', 13, 1)]}

def cell_key(env, sc, dd, model, T):
    return f'horiz|{env}|sc{sc}|dd{dd}|T{T}|{model}'

def build_cells(env, sc, dd, model, T, n_ep):
    cells = []
    for i in range(n_ep):
        seed = hash(f'horiz:{env}:{sc}:{dd}:{model}:{T}:{i}') & 2147483647
        stress = {'state_card': sc, 'dep_density': dd, 'T': T, 'branching': 4, 'obs_noise': 'clean', 'mut_rate': 'static'}
        tid = f'horiz_{env}_sc{sc}_dd{dd}_T{T}_{model}_ep{i:03d}'
        cells.append(CellSpec(env_name=env, model=model, stress_config=stress, task_config={'task_id': tid, 'stress_config': stress}, task_seed=seed, decoding_seed=seed ^ 48879, world_regime='horizon_sweep', task_id=tid))
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

def print_T_curve(results, env, sc, dd, model):
    print(f'\n[{env}] sc={sc} dd={dd} [{model}]  T-sweep')
    print('    T  |  p̂    n')
    for T in T_VALS:
        v = results.get((env, sc, dd, model, T))
        if v:
            p = sum(v) / len(v)
            bar = '█' * int(p * 20) + '░' * int((1 - p) * 20)
            mk = '*** CLIFF' if p < 0.1 else '↓↓' if p < 0.3 else '↓' if p < 0.6 else ''
            print(f'  T={T:>3} | {p:.2f}  {bar}  {mk}')
        else:
            print(f'  T={T:>3} |  —')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    models = [args.model] if args.model else list(ANCHORS)
    client = ConfiguredLLMClient(timeout=180.0)
    results = {}
    for model in models:
        done_path = HERE / f"horiz_done_{model.replace('-', '_')}.json"
        ep_path = LOG_DIR / f"horizon_{model.replace('-', '_')}_episode.jsonl"
        step_path = LOG_DIR / f"horizon_{model.replace('-', '_')}_step.jsonl"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        done = load_done(done_path)
        anchors = ANCHORS[model]
        total = len(anchors) * len(T_VALS)
        skip = sum((1 for env, sc, dd in anchors for T in T_VALS if cell_key(env, sc, dd, model, T) in done))
        print(f'\n=== {model} === {total} cells ({skip} done)')
        with JSONLWriter(ep_path) as ep_w, JSONLWriter(step_path) as st_w:
            for env, sc, dd in anchors:
                for T in T_VALS:
                    key = cell_key(env, sc, dd, model, T)
                    if key in done:
                        print(f'  SKIP {key}')
                        continue
                    print(f'  → {env} sc={sc} dd={dd} T={T:<3}  (n={N_EP})')
                    cells = build_cells(env, sc, dd, model, T, N_EP)
                    ct = CostTracker(out_path=COST_DIR / 'horizon_cost.jsonl', phase='horizon_sweep', slice_name=key, emit_every=10)
                    outcomes = run_pilot_slice(cells=cells, client=client, step_jsonl_path=step_path, episode_jsonl_path=ep_path, cost_tracker=ct, n_workers=args.workers)
                    succ = sum((1 for o in outcomes if o.success))
                    p = succ / max(len(outcomes), 1)
                    results[env, sc, dd, model, T] = [o.success for o in outcomes]
                    done.add(key)
                    save_done(done_path, done)
                    mk = '***CLIFF' if p < 0.1 else '↓↓' if p < 0.3 else '↓' if p < 0.5 else ''
                    print(f'     p̂={p:.3f} ({succ}/{len(outcomes)}) {mk}')
    print('\n' + '=' * 60)
    print('HORIZON SWEEP RESULTS')
    for model in models:
        for env, sc, dd in ANCHORS[model]:
            print_T_curve(results, env, sc, dd, model)
if __name__ == '__main__':
    main()
