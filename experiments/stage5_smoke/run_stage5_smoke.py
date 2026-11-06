from __future__ import annotations
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.agents.llm_client import LLMClient
from src.runner import CellSpec, CostTracker, EpisodeOutcome, run_pilot_slice

def jst_now() -> str:
    return datetime.now(tz=timezone(timedelta(hours=9))).isoformat()

def main() -> int:
    out_dir = ROOT / 'experiments' / 'stage5_smoke'
    log_dir = ROOT / 'data' / 'raw_logs'
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    backdrop = {'T': 40, 'state_card': 10, 'branching': 4, 'obs_noise': 'clean', 'mut_rate': 'static', 'dep_density': 4}
    cells = []
    for i in range(10):
        task_seed = 700000 + 10 * 10000 + i
        cells.append(CellSpec(env_name='stateful_puzzle', model='gpt-4o-mini', stress_config=backdrop, task_config={'archetype': 'stage5_smoke', 'stress_config': backdrop}, task_seed=task_seed, decoding_seed=42, world_regime='G1_trigger_2D_mini', task_id=f'stage5_smoke_sp_dd4_t{i:02d}', memory_mode='C_struct'))
    client = LLMClient()
    ct = CostTracker(out_path=log_dir / 'cost_tracker.jsonl', phase='stage5_smoke', slice_name='stage5_smoke_sp_dd4_mini', emit_every=5)
    print(f'[smoke] === Stage 5 A pre-dispatch smoke ===')
    t0 = time.perf_counter()

    def progress(i, n, o: EpisodeOutcome):
        s = 'OK' if o.error is None else f'ERR({o.error[:80]})'
        success = '✓' if o.success else '✗'
        print(f'[smoke {i}/{n}] {o.cell.task_id} {success}{s} steps={o.steps} cost=${o.cost_usd:.4f}')
    outcomes = run_pilot_slice(cells=cells, client=client, step_jsonl_path=log_dir / 'stage5_smoke_step.jsonl', episode_jsonl_path=log_dir / 'stage5_smoke_episode.jsonl', cost_tracker=ct, n_workers=2, progress_fn=progress)
    elapsed = time.perf_counter() - t0
    total_cost = sum((o.cost_usd for o in outcomes))
    n_err = sum((1 for o in outcomes if o.error is not None))
    n_succ = sum((1 for o in outcomes if o.success))
    verdict = 'PASS' if n_err == 0 and total_cost > 0.0 and (total_cost < 2.0) else 'FAIL'
    summary = {'timestamp_jst': jst_now(), 'model': 'gpt-4o-mini', 'env': 'stateful_puzzle', 'state_card': 10, 'dep_density': 4, 'n_total': len(outcomes), 'n_success': n_succ, 'n_error_crashes': n_err, 'total_cost_usd': round(total_cost, 6), 'wall_clock_s': round(elapsed, 1), 'verdict': verdict, 'errors': [{'task_id': o.cell.task_id, 'error': o.error} for o in outcomes if o.error]}
    (out_dir / 'stage5_smoke_results.json').write_text(json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False))
    print(f'\n[smoke] === RESULT ===')
    print(f'  n_total={len(outcomes)} n_success={n_succ} n_crashes={n_err}')
    print(f'  total_cost=${total_cost:.4f}')
    print(f'  wall_clock={elapsed / 60:.1f}min')
    print(f'  verdict: {verdict}')
    return 0 if verdict == 'PASS' else 1
if __name__ == '__main__':
    sys.exit(main())
