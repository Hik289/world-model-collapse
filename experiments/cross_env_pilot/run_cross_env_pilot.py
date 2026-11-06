from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.agents.llm_client import ConfiguredLLMClient
from src.runner.pilot_runner import CellSpec, run_pilot_slice
from src.runner.cost_tracker import CostTracker
ENVS = ['graph_nav', 'tool_dag']
MODELS = ['claude-haiku-4-5', 'gpt-5.4-mini']
SC_VALS = [5, 10, 20, 40]
DD_VALS = [1, 2, 3, 4]
N_DEFAULT = 15
T_DEFAULT = 40
WORLD_REGIME = 'cross_env_pilot'
MEMORY_MODE = 'C_struct'
TASK_SEED_BASE = 5000000
DECODING_SEED = 42
HERE = Path(__file__).resolve().parent
LOG_DIR = ROOT / 'data' / 'raw_logs'
EP_PATH = LOG_DIR / 'cross_env_pilot_episode.jsonl'
STEP_PATH = LOG_DIR / 'cross_env_pilot_step.jsonl'
RESULTS_PATH = HERE / 'cross_env_pilot_results.json'
COMPLETED_PATH = HERE / 'completed_cells.json'
COST_PATH = HERE / 'logs' / 'cost_tracker.jsonl'
SMOKE_CELLS = [('graph_nav', 5, 1), ('tool_dag', 5, 1)]

def _cell_seed_offset(env_name: str, sc: int, dd: int, model: str) -> int:
    env_idx = ENVS.index(env_name)
    model_idx = MODELS.index(model)
    sc_idx = SC_VALS.index(sc)
    dd_idx = DD_VALS.index(dd)
    return TASK_SEED_BASE + env_idx * 1000000 + model_idx * 100000 + sc_idx * 10000 + dd_idx * 1000

def _cell_key(env_name: str, sc: int, dd: int, model: str) -> str:
    return f'{env_name}|sc{sc}|dd{dd}|{model}'

def build_cells(env_name: str, sc: int, dd: int, model: str, n_ep: int) -> list[CellSpec]:
    base = _cell_seed_offset(env_name, sc, dd, model)
    stress_config = {'state_card': sc, 'dep_density': dd, 'T': T_DEFAULT, 'branching': 4, 'obs_noise': 'clean', 'mut_rate': 'static'}
    cells = []
    for ep_idx in range(n_ep):
        task_id = f'xenv_{env_name}_sc{sc}_dd{dd}_{model}_ep{ep_idx:03d}'
        task_config = {'task_id': task_id, 'stress_config': dict(stress_config)}
        cells.append(CellSpec(env_name=env_name, model=model, stress_config=dict(stress_config), task_config=task_config, task_seed=base + ep_idx, decoding_seed=DECODING_SEED, world_regime=WORLD_REGIME, task_id=task_id, memory_mode=MEMORY_MODE))
    return cells

def load_completed() -> set[str]:
    if COMPLETED_PATH.exists():
        try:
            return set(json.loads(COMPLETED_PATH.read_text()))
        except Exception:
            return set()
    return set()

def mark_completed(key: str, completed: set[str]) -> None:
    completed.add(key)
    COMPLETED_PATH.write_text(json.dumps(sorted(completed), indent=2))

def load_results() -> dict:
    if RESULTS_PATH.exists():
        try:
            return json.loads(RESULTS_PATH.read_text())
        except Exception:
            return {}
    return {}

def save_results(results: dict) -> None:
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

def print_heatmap(results: dict, env_name: str, model: str) -> str:
    lines = []
    lines.append(f'env={env_name} model={model}')
    header = '   sc\\dd |' + ''.join((f'  {d:>4}' for d in DD_VALS))
    lines.append(header)
    for sc in SC_VALS:
        row = f'   {sc:>5}  |'
        for dd in DD_VALS:
            key = _cell_key(env_name, sc, dd, model)
            cell = results.get(key)
            if cell is None:
                row += '    -- '
            else:
                row += f"  {cell['p_hat']:.2f}"
        lines.append(row)
    out = '\n'.join(lines)
    print(out)
    print()
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='8-ep smoke (2/cell × 2 cells × 2 models)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--n', type=int, default=0, help='override episodes/cell')
    args = ap.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    COST_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_ep = 2 if args.smoke else args.n or N_DEFAULT
    if args.smoke:
        cell_groups = [(env, sc, dd, model) for env, sc, dd in SMOKE_CELLS for model in MODELS]
        completed: set[str] = set()
        results: dict = {}
        ep_path = LOG_DIR / 'cross_env_pilot_smoke_episode.jsonl'
        step_path = LOG_DIR / 'cross_env_pilot_smoke_step.jsonl'
    else:
        cell_groups = [(env, sc, dd, model) for env in ENVS for model in MODELS for sc in SC_VALS for dd in DD_VALS]
        completed = load_completed()
        results = load_results()
        ep_path, step_path = (EP_PATH, STEP_PATH)
    client = ConfiguredLLMClient(timeout=120.0)
    total_groups = len(cell_groups)
    print(f"Cross-env {('SMOKE' if args.smoke else 'FULL')} scan")
    print(f'  cell-groups={total_groups} | n={n_ep}/cell | workers={args.workers}')
    print(f'  already completed: {len(completed)} cells')
    print(f'  logs → {ep_path.name}, {step_path.name}\n')
    n_errors = 0
    for gi, (env_name, sc, dd, model) in enumerate(cell_groups, 1):
        key = _cell_key(env_name, sc, dd, model)
        if key in completed:
            print(f'[{gi}/{total_groups}] SKIP (done): {key}')
            continue
        print(f'[{gi}/{total_groups}] → {key}  (n={n_ep})')
        cells = build_cells(env_name, sc, dd, model, n_ep)
        cost_tracker = CostTracker(out_path=COST_PATH, phase='cross_env_pilot', slice_name=key, emit_every=10)
        outcomes = run_pilot_slice(cells=cells, client=client, step_jsonl_path=step_path, episode_jsonl_path=ep_path, cost_tracker=cost_tracker, n_workers=args.workers)
        successes = sum((1 for o in outcomes if o.success))
        errs = [o.error for o in outcomes if o.error]
        n_errors += len(errs)
        p_hat = successes / max(len(outcomes), 1)
        results[key] = {'env': env_name, 'sc': sc, 'dd': dd, 'model': model, 'n': len(outcomes), 'successes': successes, 'p_hat': round(p_hat, 4), 'errors': len(errs)}
        if errs:
            print(f'     ⚠ {len(errs)} errors, sample: {errs[0][:120]}')
        print(f'     p̂={p_hat:.3f} ({successes}/{len(outcomes)})')
        if not args.smoke:
            mark_completed(key, completed)
            save_results(results)
    print('\n' + '=' * 60)
    print('HEATMAPS')
    print('=' * 60 + '\n')
    heatmap_texts = []
    for env_name in ENVS:
        for model in MODELS:
            heatmap_texts.append(print_heatmap(results, env_name, model))
    if not args.smoke:
        save_results(results)
    print(f'\nSmoke errors total: {n_errors}' if args.smoke else f'\nTotal errors: {n_errors}')
    print(f'Results → {RESULTS_PATH}')
    if args.smoke:
        ran = sum((r['n'] for r in results.values()))
        print(f'\n=== SMOKE VERDICT ===')
        print(f'episodes ran: {ran} | errors: {n_errors}')
        if n_errors == 0 and ran == len(cell_groups) * n_ep:
            print('SMOKE PASS ✓ — both proxies respond, ConfiguredLLMClient works with agent')
        else:
            print('SMOKE FAIL ✗ — inspect errors above')
if __name__ == '__main__':
    main()
