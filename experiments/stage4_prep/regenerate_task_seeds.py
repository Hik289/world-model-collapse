from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'experiments' / 'stage4_prep'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def jst_now() -> str:
    return datetime.now(tz=timezone(timedelta(hours=9))).isoformat()

def sha256_seed(cell_key: str) -> int:
    h = hashlib.sha256(cell_key.encode('utf-8')).hexdigest()
    return int(h[:8], 16)
OUT_NAME_BY_ENV = {'stateful_puzzle': 'stage4_task_seeds.json', 'graph_nav': 'stage4_task_seeds_graph_nav.json', 'tool_dag': 'stage4_task_seeds_tool_dag.json'}
OUT_NAME_BY_ENV_GPT55 = {'stateful_puzzle': 'stage4_task_seeds_gpt55.json', 'graph_nav': 'stage4_task_seeds_graph_nav_gpt55.json', 'tool_dag': 'stage4_task_seeds_tool_dag_gpt55.json'}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='stateful_puzzle', choices=['stateful_puzzle', 'graph_nav', 'tool_dag'], help='Which env grid to generate (writes to its own seed file).')
    ap.add_argument('--model', default='claude-haiku-4-5', help='Model string written into each cell (default claude-haiku-4-5). Use gpt-5.5 for the codex-proxy cross-model grid.')
    args = ap.parse_args()
    envs = [args.env]
    state_cards = [5, 10, 20, 40]
    dep_densities = [1, 2, 4, 6]
    models = [args.model]
    n_task = 100
    decoding_seed = 42
    cells: list[dict] = []
    seeds_seen: set[int] = set()
    collisions = 0
    for env in envs:
        for sc in state_cards:
            for dd in dep_densities:
                for model in models:
                    for i in range(n_task):
                        cell_key = f'stage4|env={env}|state_card={sc}|dep_density={dd}|model={model}|task_index={i}'
                        seed_key = f'stage4|env={env}|state_card={sc}|dep_density={dd}|model=claude-haiku-4-5|task_index={i}'
                        seed = sha256_seed(seed_key)
                        if seed in seeds_seen:
                            collisions += 1
                        seeds_seen.add(seed)
                        cells.append({'cell_key': cell_key, 'env': env, 'state_card': sc, 'dep_density': dd, 'model': model, 'task_index': i, 'task_seed': seed, 'decoding_seed': decoding_seed, 'world_regime': 'G1_trigger_2D', 'stress_config': {'T': 40, 'state_card': sc, 'branching': 4, 'dep_density': dd, 'obs_noise': 'clean', 'mut_rate': 'static'}, 'task_id': f'stage4_{env}_sc{sc:02d}_dd{dd}_t{i:03d}', 'task_config': {'archetype': 'stage4_g1_trigger', 'stress_config': {'T': 40, 'state_card': sc, 'branching': 4, 'dep_density': dd, 'obs_noise': 'clean', 'mut_rate': 'static'}}, 'memory_mode': 'C_struct'})
    out = {'generated_jst': jst_now(), 'schema_version': 'stage4_g1_trigger_v1', 'grid_axes': {'envs': envs, 'state_cards': state_cards, 'dep_densities': dep_densities, 'models': models, 'n_task_per_cell': n_task}, 'total_cells': len(cells), 'unique_seeds': len(seeds_seen), 'collisions': collisions, 'seed_generator': 'int(sha256(cell_key).hexdigest()[:8], 16)', 'cell_key_template': 'stage4|env={env}|state_card={sc}|dep_density={dd}|model={model}|task_index={i}', 'cells': cells}
    name_map = OUT_NAME_BY_ENV_GPT55 if args.model.startswith('gpt-5') else OUT_NAME_BY_ENV
    out_path = OUT_DIR / name_map[args.env]
    out_path.write_text(json.dumps(out, sort_keys=True, indent=2, ensure_ascii=False))
    print(f'Generated {len(cells)} Stage-4 cell-tasks.')
    print(f'Unique seeds: {len(seeds_seen)} ({collisions} collisions detected)')
    print(f'Output: {out_path}')
    print()
    print('Sample first 3 cells:')
    for c in cells[:3]:
        print(f"  {c['task_id']}: seed={c['task_seed']} (key={c['cell_key']})")
    print()
    print('Sample last 3 cells:')
    for c in cells[-3:]:
        print(f"  {c['task_id']}: seed={c['task_seed']}")
    return 0
if __name__ == '__main__':
    sys.exit(main())
