from __future__ import annotations
import argparse, hashlib, json, shutil, sys, time
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
LOG_DIR.mkdir(parents=True, exist_ok=True)
COST_DIR.mkdir(parents=True, exist_ok=True)
MIN_FREE_GB = 15.0

def free_gb(path: str='/') -> float:
    return shutil.disk_usage(path).free / 1000000000.0

def mslug(model: str) -> str:
    return model.replace('-', '_').replace('.', '_').replace('/', '_')

def cell_key(env: str, model: str, sc: int, dd: int, T: int) -> str:
    return f'{env}|{model}|sc{sc}|dd{dd}|T{T}'

def load_done(path: Path) -> set:
    if path.exists():
        try:
            return set(json.loads(path.read_text()))
        except Exception:
            return set()
    return set()

def save_done(path: Path, done: set) -> None:
    path.write_text(json.dumps(sorted(done), indent=2))

def build_cells(env: str, model: str, sc: int, dd: int, T: int, n_ep: int) -> list:
    cells = []
    for i in range(n_ep):
        seed = int(hashlib.sha256(f'solgrid2d:{env}:{model}:{sc}:{dd}:{T}:{i}'.encode()).hexdigest()[:8], 16)
        stress = {'state_card': sc, 'dep_density': dd, 'T': T, 'branching': 4, 'obs_noise': 'clean', 'mut_rate': 'static'}
        tid = f'solgrid2d_{env}_sc{sc}_dd{dd}_T{T}_{mslug(model)}_ep{i:03d}'
        cells.append(CellSpec(env_name=env, model=model, stress_config=stress, task_config={'task_id': tid, 'stress_config': stress}, task_seed=seed, decoding_seed=seed ^ 48879, world_regime='sol_grid2d', task_id=tid))
    return cells

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--env', required=True, choices=['stateful_puzzle', 'graph_nav', 'tool_dag'])
    ap.add_argument('--sc', required=True, help='comma list of state_card')
    ap.add_argument('--dd', required=True, help='comma list of dep_density')
    ap.add_argument('--T', default='40')
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--plan-only', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--timeout', type=float, default=1200.0)
    args = ap.parse_args()
    model = args.model
    env = args.env
    sc_vals = [int(x) for x in str(args.sc).split(',') if x.strip()]
    dd_vals = [int(x) for x in str(args.dd).split(',') if x.strip()]
    T = int(str(args.T).split(',')[0])
    tag = f'{mslug(model)}_{env}'
    done_path = HERE / f'grid2d_done_{tag}.json'
    ep_path = LOG_DIR / f'solgrid2d_{tag}_episode.jsonl'
    st_path = LOG_DIR / f'solgrid2d_{tag}_step.jsonl'
    grid_out = HERE / f'grid2d_{tag}_success.json'
    done = load_done(done_path)
    queue = [(sc, dd) for sc in sc_vals for dd in dd_vals]
    print(f'=== sol_grid2d {model} env={env} ===', flush=True)
    print(f'    sc={sc_vals} × dd={dd_vals} T={T} | {len(queue)} cells n={args.n} workers={args.workers} free={free_gb():.0f}GB', flush=True)
    grid = {}
    if grid_out.exists():
        try:
            grid = json.load(open(grid_out))
        except Exception:
            grid = {}
    if args.plan_only:
        print(json.dumps({'model': model, 'environment': env, 'state_size': sc_vals, 'state_dependency': dd_vals, 'horizon': T, 'episodes_per_cell': args.n, 'total_episodes': len(queue) * args.n}, indent=2))
        return 0
    with JSONLWriter(ep_path) as ep_w, JSONLWriter(st_path) as st_w:
        for sc, dd in queue:
            key = cell_key(env, model, sc, dd, T)
            if key in done:
                print(f'  SKIP {key}', flush=True)
                continue
            fg = free_gb()
            if fg < MIN_FREE_GB:
                print(f'  [DISK-GUARD] free {fg:.1f}GB < {MIN_FREE_GB} — STOP before {key}.', flush=True)
                return 2
            print(f'  {env} sc={sc:<5} dd={dd:<2} T={T} (n={args.n}) free={fg:.0f}GB', flush=True)
            t0 = time.time()
            cells = build_cells(env, model, sc, dd, T, args.n)
            ct = CostTracker(out_path=COST_DIR / 'sol_grid2d_cost.jsonl', phase='sol_grid2d', slice_name=key, emit_every=5)
            client = ConfiguredLLMClient(timeout=args.timeout)
            outcomes = run_pilot_slice(cells=cells, client=client, step_jsonl_path=st_path, episode_jsonl_path=ep_path, cost_tracker=ct, n_workers=args.workers)
            if len(outcomes) != args.n or any((o.error for o in outcomes)):
                raise RuntimeError('Cell incomplete; transport failures are excluded from success rates')
            succ = sum((1 for o in outcomes if o.success))
            p = succ / max(len(outcomes), 1)
            grid[f'sc={sc},dd={dd}'] = {'success_rate': p, 'n': len(outcomes), 'succ': succ}
            json.dump(grid, open(grid_out, 'w'), indent=2)
            done.add(key)
            save_done(done_path, done)
            dt = time.time() - t0
            mk = '***CLIFF' if p < 0.1 else '↓↓' if p < 0.3 else '↓' if p < 0.6 else 'flat'
            print(f'    → p̂={p:.2f} ({succ}/{len(outcomes)}) {mk} [{dt / 60:.1f}min]', flush=True)
            if 'haiku' in model:
                time.sleep(5)
    print(f'=== DONE {model} env={env}. grid → {grid_out} ===', flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
