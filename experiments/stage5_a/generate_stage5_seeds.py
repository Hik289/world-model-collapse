from __future__ import annotations
import hashlib
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'experiments' / 'stage5_a'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def jst_now() -> str:
    return datetime.now(tz=timezone(timedelta(hours=9))).isoformat()

def sha256_seed(cell_key: str) -> int:
    return int(hashlib.sha256(cell_key.encode('utf-8')).hexdigest()[:8], 16)

def main() -> int:
    sc_levels = [5, 10, 20, 40]
    dd_levels = [1, 2, 4, 6]
    n_task = 50
    cells = []
    seeds = set()
    coll = 0
    for sc in sc_levels:
        for dd in dd_levels:
            for i in range(n_task):
                key = f'stage5a|env=stateful_puzzle|state_card={sc}|dep_density={dd}|model=gpt-4o-mini|task_index={i}'
                s = sha256_seed(key)
                if s in seeds:
                    coll += 1
                seeds.add(s)
                cells.append({'cell_key': key, 'env': 'stateful_puzzle', 'state_card': sc, 'dep_density': dd, 'model': 'gpt-4o-mini', 'task_index': i, 'task_seed': s, 'decoding_seed': 42, 'world_regime': 'G1_trigger_2D_mini', 'stress_config': {'T': 40, 'state_card': sc, 'branching': 4, 'dep_density': dd, 'obs_noise': 'clean', 'mut_rate': 'static'}, 'task_id': f'stage5a_sp_sc{sc:02d}_dd{dd}_t{i:03d}', 'task_config': {'archetype': 'stage5_a_g1_mini', 'stress_config': {'T': 40, 'state_card': sc, 'branching': 4, 'dep_density': dd, 'obs_noise': 'clean', 'mut_rate': 'static'}}, 'memory_mode': 'C_struct'})
    out = {'generated_jst': jst_now(), 'schema_version': 'stage5a_g1_mini_v1', 'grid_axes': {'envs': ['stateful_puzzle'], 'state_cards': sc_levels, 'dep_densities': dd_levels, 'models': ['gpt-4o-mini'], 'n_task_per_cell': n_task}, 'total_cells': len(cells), 'unique_seeds': len(seeds), 'collisions': coll, 'seed_generator': 'int(sha256(cell_key).hexdigest()[:8], 16)', 'cells': cells}
    out_path = OUT_DIR / 'stage5_a_task_seeds.json'
    out_path.write_text(json.dumps(out, sort_keys=True, indent=2, ensure_ascii=False))
    print(f'Wrote {out_path}')
    print(f'  {len(cells)} cells, {len(seeds)} unique seeds, {coll} collisions')
    return 0
if __name__ == '__main__':
    sys.exit(main())
