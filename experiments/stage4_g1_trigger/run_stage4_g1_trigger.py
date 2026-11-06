from __future__ import annotations
import argparse
import contextlib
import hashlib
import itertools
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.agents.llm_client import LLMClient
from src.runner import CellSpec, CostTracker, EpisodeOutcome, run_pilot_slice
ENV_CONFIG = {'stateful_puzzle': {'slice': 'stage4_g1_trigger_sp_haiku', 'seeds': 'stage4_task_seeds.json', 'results': 'stage4_results.json', 'completed': 'completed_cells.json', 'step_log': 'stage4_step.jsonl', 'episode_log': 'stage4_episode.jsonl'}, 'graph_nav': {'slice': 'stage4_g1_trigger_gn_haiku', 'seeds': 'stage4_task_seeds_graph_nav.json', 'results': 'stage4_graph_nav_results.json', 'completed': 'completed_cells_graph_nav.json', 'step_log': 'stage4_graph_nav_step.jsonl', 'episode_log': 'stage4_graph_nav_episode.jsonl'}, 'tool_dag': {'slice': 'stage4_g1_trigger_td_haiku', 'seeds': 'stage4_task_seeds_tool_dag.json', 'results': 'stage4_tool_dag_results.json', 'completed': 'completed_cells_tool_dag.json', 'step_log': 'stage4_tool_dag_step.jsonl', 'episode_log': 'stage4_tool_dag_episode.jsonl'}, 'tool_dag_fix': {'slice': 'stage4_g1_trigger_td_fix_haiku', 'seeds': 'stage4_task_seeds_tool_dag.json', 'results': 'stage4_tool_dag_fix_results.json', 'completed': 'completed_cells_tool_dag_fix.json', 'step_log': 'stage4_tool_dag_fix_step.jsonl', 'episode_log': 'stage4_tool_dag_fix_episode.jsonl'}, 'stateful_puzzle_gpt55': {'slice': 'stage4_g1_trigger_sp_gpt55', 'seeds': 'stage4_task_seeds_gpt55.json', 'results': 'stage4_stateful_puzzle_gpt55_results.json', 'completed': 'completed_cells_gpt55.json', 'step_log': 'stage4_gpt55_step.jsonl', 'episode_log': 'stage4_gpt55_episode.jsonl'}, 'graph_nav_gpt55': {'slice': 'stage4_g1_trigger_gn_gpt55', 'seeds': 'stage4_task_seeds_graph_nav_gpt55.json', 'results': 'stage4_graph_nav_gpt55_results.json', 'completed': 'completed_cells_graph_nav_gpt55.json', 'step_log': 'stage4_graph_nav_gpt55_step.jsonl', 'episode_log': 'stage4_graph_nav_gpt55_episode.jsonl'}, 'tool_dag_gpt55': {'slice': 'stage4_g1_trigger_td_gpt55', 'seeds': 'stage4_task_seeds_tool_dag_gpt55.json', 'results': 'stage4_tool_dag_gpt55_results.json', 'completed': 'completed_cells_tool_dag_gpt55.json', 'step_log': 'stage4_tool_dag_gpt55_step.jsonl', 'episode_log': 'stage4_tool_dag_gpt55_episode.jsonl'}}
ENV_NAME = 'stateful_puzzle'
SLICE_NAME = ENV_CONFIG[ENV_NAME]['slice']
PREP_DIR = ROOT / 'experiments' / 'stage4_prep'
OUT_DIR = ROOT / 'experiments' / 'stage4_g1_trigger'
LOG_DIR = ROOT / 'data' / 'raw_logs'
SEEDS_PATH = PREP_DIR / ENV_CONFIG[ENV_NAME]['seeds']
COMPLETED_PATH = OUT_DIR / ENV_CONFIG[ENV_NAME]['completed']
RESULTS_PATH = OUT_DIR / ENV_CONFIG[ENV_NAME]['results']
STEP_LOG_PATH = LOG_DIR / ENV_CONFIG[ENV_NAME]['step_log']
EPISODE_LOG_PATH = LOG_DIR / ENV_CONFIG[ENV_NAME]['episode_log']

def _bind_env(env_name: str) -> None:
    global ENV_NAME, SLICE_NAME, SEEDS_PATH, COMPLETED_PATH, RESULTS_PATH
    global STEP_LOG_PATH, EPISODE_LOG_PATH
    cfg = ENV_CONFIG[env_name]
    ENV_NAME = env_name
    SLICE_NAME = cfg['slice']
    SEEDS_PATH = PREP_DIR / cfg['seeds']
    COMPLETED_PATH = OUT_DIR / cfg['completed']
    RESULTS_PATH = OUT_DIR / cfg['results']
    STEP_LOG_PATH = LOG_DIR / cfg['step_log']
    EPISODE_LOG_PATH = LOG_DIR / cfg['episode_log']

def _outcomes_from_episode_log() -> dict[str, EpisodeOutcome]:
    recovered: dict[str, EpisodeOutcome] = {}
    if not EPISODE_LOG_PATH.exists():
        return recovered
    with open(EPISODE_LOG_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = rec.get('task_id')
            sc = rec.get('stress_config') or {}
            if not tid or 'state_card' not in sc:
                continue
            cell = CellSpec(env_name=rec.get('env', ENV_NAME.replace('_gpt55', '')), model=rec.get('model', 'claude-haiku-4-5'), stress_config=sc, task_config={}, task_seed=int(rec.get('task_seed', 0)), decoding_seed=int(rec.get('decoding_seed', 42)), world_regime=rec.get('world_regime', 'G1_trigger_2D'), task_id=tid, memory_mode=rec.get('memory_mode', 'C_struct'))
            recovered[tid] = EpisodeOutcome(cell=cell, success=bool(rec.get('final_success', False)), steps=int(rec.get('steps_taken', 0)), input_tokens=int(rec.get('total_input_tokens', 0)), output_tokens=int(rec.get('total_output_tokens', 0)), cost_usd=float(rec.get('total_cost_usd', 0.0)), json_retries_total=0, json_calls_total=0, error=None)
    return recovered

def jst_now() -> str:
    return datetime.now(tz=timezone(timedelta(hours=9))).isoformat()

def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False))
    os.replace(tmp, path)

def load_completed() -> set[str]:
    if not COMPLETED_PATH.exists():
        return set()
    try:
        data = json.loads(COMPLETED_PATH.read_text())
        return set(data.get('task_ids', []))
    except Exception:
        return set()

def save_completed(task_ids: set[str]) -> None:
    payload = {'schema_version': 'stage4_completed_v1', 'updated_jst': jst_now(), 'n_completed': len(task_ids), 'task_ids': sorted(task_ids)}
    atomic_write_json(COMPLETED_PATH, payload)

def cell_from_dict(d: dict) -> CellSpec:
    return CellSpec(env_name=d['env'], model=d['model'], stress_config=d['stress_config'], task_config=d['task_config'], task_seed=int(d['task_seed']), decoding_seed=int(d['decoding_seed']), world_regime=d['world_regime'], task_id=d['task_id'], memory_mode=d.get('memory_mode', 'C_struct'))

def verify_cost_zero_at_100ep(cost_tracker_path: Path, slice_name: str, tolerance: float=0.001) -> tuple[bool, float, int]:
    total = 0.0
    n = 0
    if not cost_tracker_path.exists():
        return (True, 0.0, 0)
    with open(cost_tracker_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get('slice') != slice_name:
                continue
            ep_records = rec.get('episodes', [])
            if isinstance(ep_records, list):
                for er in ep_records:
                    if isinstance(er, dict):
                        c = float(er.get('cost_usd', 0.0))
                        total += c
                        n += 1
            elif 'total_cost_usd' in rec:
                total = max(total, float(rec.get('total_cost_usd', 0.0)))
                n = max(n, int(rec.get('n_episodes', 0)))
    return (total <= tolerance, total, n)

def analyze_stage4(outcomes: list[EpisodeOutcome]) -> dict:
    from collections import defaultdict
    by_sc_dd = defaultdict(lambda: {'n': 0, 'n_success': 0, 'n_error': 0, 'total_steps': 0, 'total_in': 0, 'total_out': 0, 'total_cost': 0.0, 'task_ids': []})
    for o in outcomes:
        sc = int(o.cell.stress_config['state_card'])
        dd = int(o.cell.stress_config['dep_density'])
        key = f'sc={sc},dd={dd}'
        d = by_sc_dd[key]
        d['n'] += 1
        d['task_ids'].append(o.cell.task_id)
        if o.success:
            d['n_success'] += 1
        if o.error:
            d['n_error'] += 1
        d['total_steps'] += o.steps
        d['total_in'] += o.input_tokens
        d['total_out'] += o.output_tokens
        d['total_cost'] += o.cost_usd
    out = {'per_cell': {}}
    for key, d in by_sc_dd.items():
        n = d['n']
        out['per_cell'][key] = {'n': n, 'n_success': d['n_success'], 'success_rate': d['n_success'] / n if n else 0.0, 'n_error': d['n_error'], 'mean_steps': d['total_steps'] / n if n else 0.0, 'total_input_tokens': d['total_in'], 'total_output_tokens': d['total_out'], 'total_cost_usd': round(d['total_cost'], 6)}
    return out

def legacy_main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='stateful_puzzle', choices=['stateful_puzzle', 'graph_nav', 'tool_dag', 'tool_dag_fix'], help='Which env grid to run (reads/writes env-specific paths).')
    ap.add_argument('--model', default='claude-haiku-4-5', help='Model: claude-haiku-4-5 (default) or gpt-5.5 (cross-model).')
    ap.add_argument('--n-workers', type=int, default=4)
    ap.add_argument('--checkpoint-every', type=int, default=50)
    ap.add_argument('--cost-verify-after', type=int, default=100)
    ap.add_argument('--limit', type=int, default=0, help='If >0, only run the first N remaining cells (smoke test).')
    args = ap.parse_args()
    env_key = args.env + ('_gpt55' if args.model == 'gpt-5.5' else '')
    if env_key not in ENV_CONFIG:
        ap.error('Unsupported environment/model combination')
    _bind_env(env_key)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f'[stage4] === Stage 4 G1-trigger dispatch === env={ENV_NAME} slice={SLICE_NAME}')
    print(f'[stage4] seeds={SEEDS_PATH.name} results={RESULTS_PATH.name}')
    print(f'[stage4] start: {jst_now()}')
    grid = json.loads(SEEDS_PATH.read_text())
    all_cells = [cell_from_dict(c) for c in grid['cells']]
    ALLOWED_MODELS = {'claude-haiku-4-5', 'gpt-5.5'}
    bad = sorted({c.model for c in all_cells if c.model not in ALLOWED_MODELS})
    if bad:
        print(f'[STAGE_4_ABORT_MODEL] unexpected models present: {bad}; aborting')
        return 14
    completed = load_completed()
    if completed:
        cells = [c for c in all_cells if c.task_id not in completed]
        print(f'[stage4] resuming: {len(completed)} done previously, {len(cells)} remaining')
    else:
        cells = all_cells
        print(f'[stage4] fresh start: {len(cells)} cells')
    if args.limit and args.limit > 0:
        cells = cells[:args.limit]
        print(f'[stage4] SMOKE: limiting to first {len(cells)} remaining cells')
    if not cells:
        print('[stage4] all cells already complete; computing final aggregate')
        return _finalize(all_cells, [])
    client = LLMClient()
    ct = CostTracker(out_path=LOG_DIR / 'cost_tracker.jsonl', phase='stage4', slice_name=SLICE_NAME, emit_every=args.checkpoint_every)
    completed_lock = threading.Lock()
    cost_verified = {'done': False, 'abort': False}
    outcomes: list[EpisodeOutcome] = []
    t_start = time.perf_counter()

    def progress(i, n, o: EpisodeOutcome):
        success = '✓' if o.success else '✗'
        tag = 'OK' if o.error is None else f'ERR({o.error[:60]})'
        elapsed = (time.perf_counter() - t_start) / 60.0
        rate = i / max(elapsed, 0.01)
        print(f'[stage4 {i}/{n}] {o.cell.task_id} {success}{tag} steps={o.steps} | elapsed={elapsed:.1f}min rate={rate:.2f}ep/min')
        with completed_lock:
            completed.add(o.cell.task_id)
            outcomes.append(o)
            if i % args.checkpoint_every == 0 or i == n:
                save_completed(completed)
                print(f'[stage4]   checkpoint: {len(completed)}/{len(all_cells)} cells done')
        if args.model == 'gpt-5.5':
            cost_verified['done'] = True
        if not cost_verified['done'] and i >= args.cost_verify_after:
            cost_verified['done'] = True
            ok, total, ncost = verify_cost_zero_at_100ep(LOG_DIR / 'cost_tracker.jsonl', SLICE_NAME, tolerance=0.001)
            print(f'[stage4]   cost-verify @ {i} ep: total=${total:.6f} over {ncost} records, ok={ok}')
            if not ok:
                cost_verified['abort'] = True
                print(f'[STAGE_4_ABORT_COST_NONZERO] total=${total:.4f} > tolerance; aborting')
                save_completed(completed)
                os._exit(13)
    print(f'[stage4] launching run_pilot_slice (n_workers={args.n_workers}) ...')
    outs = run_pilot_slice(cells=cells, client=client, step_jsonl_path=STEP_LOG_PATH, episode_jsonl_path=EPISODE_LOG_PATH, cost_tracker=ct, n_workers=args.n_workers, progress_fn=progress)
    outcomes.extend(outs)
    return _finalize(all_cells, outcomes)

def _finalize(all_cells: list, outcomes: list) -> int:
    seen: dict[str, EpisodeOutcome] = _outcomes_from_episode_log()
    for o in outcomes:
        seen[o.cell.task_id] = o
    uniq = list(seen.values())
    summary = analyze_stage4(uniq)
    models_seen = sorted({o.cell.model for o in uniq}) or ['claude-haiku-4-5']
    real_env = uniq[0].cell.env_name if uniq else ENV_NAME
    summary['meta'] = {'timestamp_jst': jst_now(), 'slice': SLICE_NAME, 'model': models_seen[0] if len(models_seen) == 1 else models_seen, 'env': real_env, 'grid_axes': {'state_cards': [5, 10, 20, 40], 'dep_densities': [1, 2, 4, 6], 'n_task_per_cell': 100, 'total_cells_planned': len(all_cells)}, 'n_completed': len(uniq)}
    atomic_write_json(RESULTS_PATH, summary)
    save_completed({o.cell.task_id for o in uniq})
    print(f'\n[stage4] === DONE === wrote {RESULTS_PATH}')
    print(f'[stage4] completed {len(uniq)}/{len(all_cells)} cells')
    if len(uniq) >= len(all_cells):
        print(f'[stage4] GRID_COMPLETE env={ENV_NAME} n={len(uniq)}')
    print('\n  success_rate grid (rows=state_card, cols=dep_density):')
    print(f"  {'sc\\dd':>6} | {'1':>6} {'2':>6} {'4':>6} {'6':>6}")
    for sc in [5, 10, 20, 40]:
        row = [f'{sc:>6}']
        for dd in [1, 2, 4, 6]:
            key = f'sc={sc},dd={dd}'
            cell = summary['per_cell'].get(key, {})
            sr = cell.get('success_rate')
            row.append(f'{sr:>6.0%}' if sr is not None else '  N/A')
        print('  ' + ' '.join(row).replace(' | ', ' | ', 1) + ' |')
    return 0
PAPER_CONFIGS = {'ablations': {'name': 'paper_ablations', 'seed_namespace': 'world-model-collapse-main42-v1-ablations', 'environments': ['stateful_puzzle'], 'models': ['claude-haiku-4-5'], 'archetypes': 10, 'variants': 10, 'decoding_seed': 42, 'sweeps': [{'state_size': [10], 'state_dependency': [6], 'horizon': [10, 20, 40, 80], 'branching': [4], 'observation': ['clean'], 'mutation': ['static']}, {'state_size': [10], 'state_dependency': [6], 'horizon': [40], 'branching': [2, 4, 8, 16], 'observation': ['clean'], 'mutation': ['static']}, {'state_size': [10], 'state_dependency': [6], 'horizon': [40], 'branching': [4], 'observation': ['clean', 'partial', 'distractor', 'conflict'], 'mutation': ['static']}, {'state_size': [10], 'state_dependency': [6], 'horizon': [40], 'branching': [4], 'observation': ['clean'], 'mutation': ['static', 'low', 'medium', 'high']}]}, 'main_grid': {'name': 'paper_main_grid', 'seed_namespace': 'world-model-collapse-main42-v1-main_grid', 'environments': ['stateful_puzzle', 'graph_nav', 'tool_dag'], 'models': ['claude-haiku-4-5', 'gpt-4o-mini'], 'archetypes': 10, 'variants': 10, 'decoding_seed': 42, 'sweeps': [{'state_size': [5, 10, 20, 40], 'state_dependency': [1, 2, 4, 6], 'horizon': [40], 'branching': [4], 'observation': ['clean'], 'mutation': ['static']}]}, 'pilot_sd': {'name': 'paper_pilot_sd', 'seed_namespace': 'world-model-collapse-main42-v1-pilot_sd', 'environments': ['stateful_puzzle', 'graph_nav', 'tool_dag'], 'models': ['claude-haiku-4-5', 'gpt-4o-mini'], 'archetypes': 10, 'variants': 1, 'decoding_seed': 42, 'sweeps': [{'state_size': [10], 'state_dependency': [1, 2, 4, 6], 'horizon': [40], 'branching': [4], 'observation': ['clean'], 'mutation': ['static']}]}, 'ss_fine': {'name': 'paper_ss_fine', 'seed_namespace': 'world-model-collapse-main42-v1-ss_fine', 'environments': ['stateful_puzzle'], 'models': ['claude-haiku-4-5', 'gpt-4o-mini', 'gpt-4o', 'meta.llama3-70b-instruct-v1:0'], 'archetypes': 10, 'variants': 5, 'decoding_seed': 42, 'sweeps': [{'state_size': [5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 40], 'state_dependency': [1], 'horizon': [40], 'branching': [4], 'observation': ['clean'], 'mutation': ['static']}]}, 't_fine': {'name': 'paper_t_fine', 'seed_namespace': 'world-model-collapse-main42-v1-t_fine', 'environments': ['stateful_puzzle'], 'models': ['claude-haiku-4-5'], 'archetypes': 10, 'variants': 5, 'decoding_seed': 42, 'sweeps': [{'state_size': [10], 'state_dependency': [6], 'horizon': [22, 25, 28, 30, 32, 35, 38, 42, 48, 55, 65], 'branching': [4], 'observation': ['clean'], 'mutation': ['static']}]}}
for _env, _sizes in {'stateful_puzzle': [10, 20, 40, 80, 160], 'graph_nav': [5, 10, 20, 40, 80], 'tool_dag': [40, 80, 160, 320]}.items():
    PAPER_CONFIGS['extended_' + _env] = {'name': 'extended_' + _env, 'seed_namespace': 'world-model-collapse-extended-v1', 'environments': [_env], 'models': ['claude-haiku-4-5', 'gpt-5.6-luna', 'claude-sonnet-4-5', 'gpt-5.6-sol'], 'archetypes': 5, 'variants': 1, 'decoding_seed': 42, 'sweeps': [{'state_size': _sizes, 'state_dependency': [1, 2, 4, 6, 8], 'horizon': [40], 'branching': [4], 'observation': ['clean'], 'mutation': ['static']}]}
PROTOCOL = 'paper-main42-v1'
ENVIRONMENTS = {'stateful_puzzle', 'graph_nav', 'tool_dag'}
AXES = ('state_size', 'state_dependency', 'horizon', 'branching', 'observation', 'mutation')
LEGACY_KEYS = dict(zip(AXES, ('state_card', 'dep_density', 'T', 'branching', 'obs_noise', 'mut_rate')))

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value

def load_config(path):
    config = json.loads(canonical(path)) if isinstance(path, dict) else json.loads(Path(path).read_text())
    required = {'name', 'seed_namespace', 'environments', 'models', 'archetypes', 'variants', 'decoding_seed', 'sweeps'}
    if set(config) != required:
        raise ValueError(f'Config keys must be {sorted(required)}')
    for key in ('name', 'seed_namespace'):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f'{key} must be a nonempty string')
    for key in ('environments', 'models'):
        values = config[key]
        if not isinstance(values, list) or not values or any((not isinstance(v, str) or not v for v in values)):
            raise ValueError(f'Invalid {key}')
        if len(values) != len(set(values)):
            raise ValueError(f'Duplicate {key}')
    if not set(config['environments']) <= ENVIRONMENTS:
        raise ValueError('Unknown environment')
    for model in config['models']:
        if not model.startswith(('gpt-', 'claude-', 'azure:', 'meta.', 'us.meta.')):
            raise ValueError(f'Unsupported provider prefix: {model}')
    positive_int(config['archetypes'], 'archetypes')
    positive_int(config['variants'], 'variants')
    if type(config['decoding_seed']) is not int or config['decoding_seed'] < 0:
        raise ValueError('decoding_seed must be a nonnegative integer')
    if not isinstance(config['sweeps'], list) or not config['sweeps']:
        raise ValueError('sweeps must be a nonempty list')
    for sweep in config['sweeps']:
        if set(sweep) != set(AXES):
            raise ValueError(f'Every sweep must specify {AXES}')
        for key in AXES:
            values = sweep[key]
            if not isinstance(values, list) or not values or any((isinstance(v, (list, dict)) for v in values)):
                raise ValueError(f'Invalid sweep axis {key}')
            if len(values) != len(set(values)):
                raise ValueError(f'Duplicate levels in {key}')
        for key in AXES[:4]:
            for value in sweep[key]:
                positive_int(value, key)
        if not set(sweep['state_dependency']) <= {1, 2, 4, 6, 8}:
            raise ValueError('Supported SD levels: 1, 2, 4, 6, 8')
        if not set(sweep['observation']) <= {'clean', 'partial', 'distractor', 'conflict'}:
            raise ValueError('Unknown observation mode')
        if not set(sweep['mutation']) <= {'static', 'low', 'medium', 'high'}:
            raise ValueError('Unknown mutation mode')
        from src.environments.stateful_puzzle import STATE_CARD_TABLE, GRAPH_NAV_STATE_CARD_TABLE, TOOL_DAG_STATE_CARD_TABLE
        tables = {'stateful_puzzle': STATE_CARD_TABLE, 'graph_nav': GRAPH_NAV_STATE_CARD_TABLE, 'tool_dag': TOOL_DAG_STATE_CARD_TABLE}
        levels = set.intersection(*(set(tables[env]) for env in config['environments']))
        if not set(sweep['state_size']) <= levels:
            raise ValueError(f'State sizes supported by all selected environments: {sorted(levels)}')
    return config

def build_plan(config):
    tasks, seen_seeds = ({}, {})
    for sweep in config['sweeps']:
        for values in itertools.product(*(sweep[key] for key in AXES)):
            stress = dict(zip(AXES, values))
            for env, archetype, variant in itertools.product(config['environments'], range(config['archetypes']), range(config['variants'])):
                identity = {'namespace': config['seed_namespace'], 'env_name': env, 'stress': stress, 'archetype': archetype, 'variant': variant}
                task_id = digest(identity)
                seed = int(task_id[:16], 16)
                if seed in seen_seeds and seen_seeds[seed] != task_id:
                    raise ValueError('SHA-256 seed prefix collision')
                seen_seeds[seed] = task_id
                legacy_stress = {LEGACY_KEYS[k]: v for k, v in stress.items()}
                tasks[task_id] = {'task_id': task_id, 'task_seed': seed, 'env_name': env, 'stress': stress, 'stress_config': legacy_stress, 'task_config': {'archetype': f'archetype_{archetype:02d}', 'variant': variant, 'stress_config': legacy_stress}}
    jobs = []
    for task in tasks.values():
        for model in config['models']:
            job = {**task, 'model': model, 'decoding_seed': config['decoding_seed']}
            job['job_id'] = digest({'task_id': task['task_id'], 'model': model, 'decoding_seed': config['decoding_seed'], 'protocol': PROTOCOL})
            jobs.append(job)
    return {'protocol': PROTOCOL, 'config': config, 'config_hash': digest(config), 'unique_tasks': len(tasks), 'episode_count': len(jobs), 'jobs': jobs}

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)

def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Malformed JSONL at {path}:{lineno}') from exc
    return rows

def source_digest():
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root / 'src').rglob('*.py')) + [Path(__file__).resolve(), root / 'analysis/stage4_g1_acceptance.py']
    return digest({str(p.relative_to(root)): p.read_text() for p in sources})

@contextlib.contextmanager
def output_lock(directory):
    import fcntl
    path = directory / '.run.lock'
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another run is using {directory}') from exc
        yield

class BufferWriter:

    def __init__(self):
        self.rows = []

    def write_record(self, record):
        from dataclasses import asdict
        self.rows.append(asdict(record) if not isinstance(record, dict) else record)

def execute(plan, directory, max_episodes=None):
    from src.agents.llm_client import LLMClient
    from src.agents.llm_agent import build_llm_agent
    from src.environments import ENV_REGISTRY
    from src.evaluation.runner import EpisodeContext, run_episode
    client = LLMClient(fixed_temperature=0.0, strict_api_errors=True)
    completed_dir = directory / 'episodes'
    completed_dir.mkdir(exist_ok=True)
    done = {p.stem for p in completed_dir.glob('*.json')}
    planned_ids = {job['job_id'] for job in plan['jobs']}
    if done - planned_ids:
        raise ValueError('Output directory contains episodes outside this manifest')
    remaining = [job for job in plan['jobs'] if job['job_id'] not in done]
    selected = remaining if max_episodes is None else remaining[:max_episodes]
    for index, job in enumerate(selected, 1):
        try:
            env = ENV_REGISTRY[job['env_name']]()
            env.reset(job['task_config'], job['task_seed'])
            agent = build_llm_agent(client, job['model'], env.get_meta().action_templates)
            ctx = EpisodeContext(run_id=job['job_id'], task_id=job['task_id'], task_seed=job['task_seed'], decoding_seed=job['decoding_seed'], world_regime=plan['config']['name'], stress_config=job['stress_config'])
            steps, episodes = (BufferWriter(), BufferWriter())
            run_episode(env, agent, job['task_config'], ctx, steps, episodes)
            atomic_json(completed_dir / f"{job['job_id']}.json", {'protocol': plan['protocol'], 'job': job, 'episode': episodes.rows[0], 'steps': steps.rows})
        except Exception as exc:
            with (directory / 'errors.jsonl').open('a') as handle:
                handle.write(canonical({'job_id': job['job_id'], 'error_type': type(exc).__name__}) + '\n')
            raise RuntimeError(f"Episode {job['job_id']} incomplete ({type(exc).__name__}); resume after resolving the error") from None
        print(f"Completed {index}/{len(selected)}: {job['env_name']} {job['model']} SS={job['stress']['state_size']} SD={job['stress']['state_dependency']}", flush=True)
    print(f"Stored {len(done) + len(selected)}/{plan['episode_count']} completed episodes")

def paper_main():
    parser = argparse.ArgumentParser(description='Three-environment paper experiment suites')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--suite', choices=sorted(PAPER_CONFIGS), default='main_grid')
    source.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true', help='Write manifest only; no environments or models are run')
    parser.add_argument('--max-episodes', type=int, help='Maximum new episodes in this invocation')
    args = parser.parse_args()
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error('--max-episodes must be positive')
    plan = build_plan(load_config(args.config if args.config else PAPER_CONFIGS[args.suite]))
    plan['source_digest'] = source_digest()
    args.output.mkdir(parents=True, exist_ok=True)
    with output_lock(args.output):
        manifest = args.output / 'manifest.json'
        if manifest.exists():
            existing = json.loads(manifest.read_text())
            if existing != plan:
                raise ValueError('Configuration or implementation changed; use a new output directory')
        else:
            if any((args.output / 'episodes').glob('*.json')):
                raise ValueError('Cannot adopt existing episodes without their original manifest')
            atomic_json(manifest, plan)
        print(f"{plan['episode_count']} episodes, {plan['unique_tasks']} unique tasks; manifest: {manifest}")
        if not args.plan_only:
            execute(plan, args.output, args.max_episodes)

def main():
    if '--legacy' in sys.argv:
        sys.argv.remove('--legacy')
        return legacy_main()
    if any((arg.split('=', 1)[0] in {'--n-workers', '--checkpoint-every', '--cost-verify-after', '--env', '--model', '--limit'} for arg in sys.argv[1:])):
        return legacy_main()
    return paper_main()
if __name__ == '__main__':
    sys.exit(main())
