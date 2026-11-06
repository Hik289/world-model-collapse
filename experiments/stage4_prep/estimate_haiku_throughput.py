from __future__ import annotations
import json
import statistics
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'experiments' / 'stage4_prep'

def jst_now() -> str:
    return datetime.now(tz=timezone(timedelta(hours=9))).isoformat()

def load_jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def main() -> int:
    smoke_steps = load_jsonl(ROOT / 'data/raw_logs/err_fix_smoke_step.jsonl')
    p4_steps = load_jsonl(ROOT / 'data/raw_logs/pilot_p4_step.jsonl')
    p4_eps = load_jsonl(ROOT / 'data/raw_logs/pilot_p4_episode.jsonl')
    grid = json.loads((OUT_DIR / 'stage4_task_seeds.json').read_text())
    smoke_ms = [s['wallclock_ms'] for s in smoke_steps]
    p4_ms = [s['wallclock_ms'] for s in p4_steps]

    def pct(xs, p):
        return statistics.quantiles(xs, n=100)[p - 1] if len(xs) >= 100 else max(xs)
    step_latency = {'smoke_n_steps': len(smoke_ms), 'smoke_median_ms_per_step_3calls': int(statistics.median(smoke_ms)), 'smoke_mean_ms_per_step_3calls': int(statistics.mean(smoke_ms)), 'smoke_p95_ms_per_step_3calls': int(pct(smoke_ms, 95)) if smoke_ms else 0, 'p4_n_steps': len(p4_ms), 'p4_median_ms_per_step_3calls': int(statistics.median(p4_ms)), 'p4_mean_ms_per_step_3calls': int(statistics.mean(p4_ms)), 'p4_p95_ms_per_step_3calls': int(pct(p4_ms, 95))}
    median_ms_per_step = step_latency['smoke_median_ms_per_step_3calls']
    sp_step_means_by_dd: dict[int, float] = {}
    for ep in p4_eps:
        if 'stateful_puzzle' not in ep['task_id']:
            continue
        dd = ep['stress_config']['dep_density']
        sp_step_means_by_dd.setdefault(dd, []).append(ep['steps_taken'])
    sp_step_means_by_dd = {dd: statistics.mean(v) for dd, v in sp_step_means_by_dd.items()}
    sc_scaling = {5: 0.7, 10: 1.0, 20: 1.2, 40: 1.4}
    T_max = 40
    cell_step_estimates: dict[str, float] = {}
    for c in grid['cells']:
        dd = c['dep_density']
        sc = c['state_card']
        anchor = sp_step_means_by_dd.get(dd, 20.0)
        est = min(T_max, anchor * sc_scaling.get(sc, 1.0))
        cell_step_estimates[c['task_id']] = est
    total_steps_grid = sum(cell_step_estimates.values())
    total_steps_grid_p95 = total_steps_grid * 1.2
    n_workers = 4
    ms_per_step = median_ms_per_step
    total_wallclock_min_median = total_steps_grid * ms_per_step / (1000 * 60 * n_workers)
    total_wallclock_min_p95 = total_steps_grid_p95 * step_latency['smoke_p95_ms_per_step_3calls'] / (1000 * 60 * n_workers)
    eta_per_dd: dict[int, float] = {}
    for dd in [1, 2, 4, 6]:
        cells_in_dd = [c for c in grid['cells'] if c['dep_density'] == dd]
        cell_steps = sum((cell_step_estimates[c['task_id']] for c in cells_in_dd))
        eta_per_dd[dd] = round(cell_steps * ms_per_step / (1000 * 60 * n_workers), 1)
    out = {'timestamp_jst': jst_now(), 'method_summary': 'Step-latency from err_fix_smoke (post-fix, 5 ep, tool_dag dep=4); mean-step-count anchor from P4 sp eps per dd; sc projected with sc_scaling = {5:0.7,10:1.0,20:1.2,40:1.4} capped at T_max=40.', 'step_latency_haiku_3call': step_latency, 'sp_step_means_by_dd_from_p4': sp_step_means_by_dd, 'sc_scaling': sc_scaling, 'grid_summary': {'total_cells': grid['total_cells'], 'total_steps_estimated_median': int(total_steps_grid), 'total_steps_estimated_p95': int(total_steps_grid_p95)}, 'wall_clock_estimate': {'n_workers': n_workers, 'median_min': round(total_wallclock_min_median, 1), 'median_hours': round(total_wallclock_min_median / 60, 2), 'p95_min': round(total_wallclock_min_p95, 1), 'p95_hours': round(total_wallclock_min_p95 / 60, 2), 'eta_per_dd_bucket_min': eta_per_dd}, 'cost_estimate_usd': {'haiku_lab_proxy_rate_per_m_in': 0.0, 'haiku_lab_proxy_rate_per_m_out': 0.0, 'expected_actual_billed': 0.0, 'note': 'Lab Anthropic proxy uses local OAuth quota. cost_tracker.py updated 2026-06-01 (haiku rates → 0 from list-price 0.8/4.0). Past P4 $50.51 + smoke $2.40 = phantom estimates only.'}, 'recommendation_for_director': {'dispatch_concurrency': n_workers, 'expected_wall_clock_hours_range': [round(total_wallclock_min_median / 60, 1), round(total_wallclock_min_p95 / 60, 1)], 'staging_suggestion': 'Optional: dispatch dd=1,2 (low-dep buckets, fast) first as a warm-up; if no anomalies, fan out dd=4,6. Total grid is 1600 cells; full-concurrency dispatch is acceptable since cost=$0.'}}
    out_path = OUT_DIR / 'haiku_throughput_estimate.json'
    out_path.write_text(json.dumps(out, sort_keys=True, indent=2, ensure_ascii=False))
    print(f'Wrote {out_path}')
    print()
    print('=== Summary ===')
    print(f"  Step latency (haiku 3-call, post-fix): median {ms_per_step}ms, p95 {step_latency['smoke_p95_ms_per_step_3calls']}ms")
    print(f"  Stage 4 grid: {grid['total_cells']} cells, est {int(total_steps_grid)} steps median")
    print(f'  ETA at 4 workers: median {round(total_wallclock_min_median / 60, 1)}h, p95 {round(total_wallclock_min_p95 / 60, 1)}h')
    print(f'  Cost: $0 (haiku via lab proxy)')
    print(f'  Per-dd ETA min: {eta_per_dd}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
