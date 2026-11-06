from __future__ import annotations
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.agents.llm_client import LLMClient
from src.runner import CellSpec, CostTracker, run_pilot_slice
LOG_DIR = ROOT / 'data' / 'raw_logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

def main() -> int:
    cells = []
    for i in range(5):
        stress = {'T': 40, 'branching': 4, 'dep_density': 1, 'mut_rate': 'static', 'obs_noise': 'clean', 'state_card': 11}
        cells.append(CellSpec(env_name='stateful_puzzle', model='claude-haiku-4-5', stress_config=stress, task_config={'archetype': 'critical_scan', 'stress_config': stress}, task_seed=1100000 + i, decoding_seed=42, world_regime='ExpSC_fine_smoke', task_id=f'smoke_sc11_t{i:03d}', memory_mode='C_struct'))
    client = LLMClient()
    ct = CostTracker(out_path=LOG_DIR / 'cost_tracker.jsonl', phase='smoke_sc11', slice_name='smoke_sc11', emit_every=5)
    outcomes = []

    def progress(i, n, o):
        success = '✓' if o.success else '✗'
        err = 'OK' if o.error is None else f'ERR({o.error[:60]})'
        outcomes.append(o)
        print(f'[smoke {i}/{n}] {o.cell.task_id} {success}{err} steps={o.steps} cost=${o.cost_usd:.4f}')
    run_pilot_slice(cells=cells, client=client, step_jsonl_path=LOG_DIR / 'smoke_sc11_step.jsonl', episode_jsonl_path=LOG_DIR / 'smoke_sc11_episode.jsonl', cost_tracker=ct, n_workers=2, progress_fn=progress)
    n = len(outcomes)
    n_success = sum((1 for o in outcomes if o.success))
    n_error = sum((1 for o in outcomes if o.error is not None))
    summary = {'n': n, 'n_success': n_success, 'n_error': n_error, 'success_rate': n_success / n if n else 0.0, 'errors': [o.error[:200] for o in outcomes if o.error]}
    print('\n=== SMOKE SUMMARY ===')
    print(json.dumps(summary, indent=2))
    out_path = Path(__file__).parent / 'smoke_sc11_summary.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print(f'Wrote {out_path}')
    return 0 if n_error == 0 else 1
if __name__ == '__main__':
    sys.exit(main())
