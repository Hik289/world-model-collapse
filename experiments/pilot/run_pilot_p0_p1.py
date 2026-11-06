from __future__ import annotations
import argparse
import hashlib
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
REGIME_I_STRESS = {'T': 40, 'state_card': 5, 'dep_density': 1, 'branching': 2, 'obs_noise': 'clean', 'mut_rate': 'static'}
P1_BACKDROP = {'T': 40, 'state_card': 10, 'branching': 4, 'obs_noise': 'clean', 'mut_rate': 'static'}
P1_DEP_LEVELS = [1, 2, 4, 6]
ENVS_P0 = ['graph_nav', 'tool_dag', 'stateful_puzzle']
MODEL = 'gpt-4o-mini'
MEMORY_MODE = 'C_struct'

def _env_hash(env_name: str) -> int:
    return int.from_bytes(hashlib.sha256(env_name.encode('utf-8')).digest()[:4], 'big') & 65535

def build_p0_cells(n_task_per_env: int=30, decoding_seed: int=42) -> list[CellSpec]:
    cells: list[CellSpec] = []
    for env in ENVS_P0:
        for i in range(n_task_per_env):
            task_seed = 200000 + _env_hash(env) * 1000 + i
            cells.append(CellSpec(env_name=env, model=MODEL, stress_config=REGIME_I_STRESS, task_config={'archetype': 'pilot_p0', 'stress_config': REGIME_I_STRESS}, task_seed=task_seed, decoding_seed=decoding_seed, world_regime='I_stable', task_id=f'p0_{env}_t{i:03d}', memory_mode=MEMORY_MODE))
    return cells

def build_p1_cells(n_task_per_level: int=10, decoding_seed: int=42) -> list[CellSpec]:
    cells: list[CellSpec] = []
    env = 'graph_nav'
    for level in P1_DEP_LEVELS:
        stress = dict(P1_BACKDROP, dep_density=level)
        for i in range(n_task_per_level):
            task_seed = 300000 + level * 10000 + i
            cells.append(CellSpec(env_name=env, model=MODEL, stress_config=stress, task_config={'archetype': 'pilot_p1_anchor5', 'stress_config': stress}, task_seed=task_seed, decoding_seed=decoding_seed, world_regime='III_coupled_backdrop', task_id=f'p1_dep{level}_t{i:02d}', memory_mode=MEMORY_MODE))
    return cells

def analyze_p0(outcomes: list[EpisodeOutcome]) -> dict:
    by_env: dict[str, dict] = {}
    for o in outcomes:
        env = o.cell.env_name
        d = by_env.setdefault(env, {'n': 0, 'n_success': 0, 'n_error': 0, 'total_steps': 0, 'total_in_tok': 0, 'total_out_tok': 0, 'total_cost_usd': 0.0})
        d['n'] += 1
        if o.success:
            d['n_success'] += 1
        if o.error:
            d['n_error'] += 1
        d['total_steps'] += o.steps
        d['total_in_tok'] += o.input_tokens
        d['total_out_tok'] += o.output_tokens
        d['total_cost_usd'] += o.cost_usd
    summary: dict = {'per_env': {}, 'anchor_4_threshold': 0.8}
    all_pass = True
    for env, d in by_env.items():
        rate = d['n_success'] / d['n'] if d['n'] else 0.0
        passed = rate >= 0.8
        all_pass = all_pass and passed
        summary['per_env'][env] = {'n': d['n'], 'n_success': d['n_success'], 'success_rate': rate, 'n_error': d['n_error'], 'mean_steps': d['total_steps'] / d['n'] if d['n'] else 0.0, 'total_input_tokens': d['total_in_tok'], 'total_output_tokens': d['total_out_tok'], 'total_cost_usd': round(d['total_cost_usd'], 6), 'anchor_4_passed': passed}
    summary['anchor_4_overall_passed'] = all_pass
    return summary

def analyze_p1(outcomes: list[EpisodeOutcome], baseline_rate: float | None=None) -> dict:
    by_level: dict[int, dict] = {}
    for o in outcomes:
        level = int(o.cell.stress_config['dep_density'])
        d = by_level.setdefault(level, {'n': 0, 'n_success': 0, 'n_error': 0, 'total_steps': 0, 'total_in_tok': 0, 'total_out_tok': 0, 'total_cost_usd': 0.0, 'task_seeds': []})
        d['n'] += 1
        if o.success:
            d['n_success'] += 1
        if o.error:
            d['n_error'] += 1
        d['total_steps'] += o.steps
        d['total_in_tok'] += o.input_tokens
        d['total_out_tok'] += o.output_tokens
        d['total_cost_usd'] += o.cost_usd
        d['task_seeds'].append(o.cell.task_seed)
    per_level = {}
    for level in sorted(by_level):
        d = by_level[level]
        rate = d['n_success'] / d['n'] if d['n'] else 0.0
        per_level[level] = {'n': d['n'], 'n_success': d['n_success'], 'success_rate': rate, 'n_error': d['n_error'], 'mean_steps': d['total_steps'] / d['n'] if d['n'] else 0.0, 'total_cost_usd': round(d['total_cost_usd'], 6)}
    sorted_levels = sorted(per_level.keys())
    adjacent_drops = []
    auxiliary_p0_to_l1 = None
    if baseline_rate is not None and sorted_levels:
        l1 = sorted_levels[0]
        r_l1 = per_level[l1]['success_rate']
        auxiliary_p0_to_l1 = {'p0_baseline_rate': baseline_rate, 'p1_l1_rate': r_l1, 'delta_pp': round((baseline_rate - r_l1) * 100.0, 2), 'note': 'Auxiliary: P0 (Regime I dep=1) vs P1 L1 (Regime III backdrop dep=1). NOT counted in anchor_5 decision.'}
    for i in range(len(sorted_levels) - 1):
        a, b = (sorted_levels[i], sorted_levels[i + 1])
        ra, rb = (per_level[a]['success_rate'], per_level[b]['success_rate'])
        delta_pp = (ra - rb) * 100.0
        adjacent_drops.append({'pair_lower_level': a, 'pair_upper_level': b, 'success_rate_lower': ra, 'success_rate_upper': rb, 'delta_pp': round(delta_pp, 2)})
    max_drop = max((d['delta_pp'] for d in adjacent_drops), default=0.0)
    if max_drop >= 20.0:
        decision = 'PASS'
        rationale = f'max adjacent Δp̂ = {max_drop:.1f}pp ≥ 20pp'
    elif max_drop >= 10.0:
        decision = 'BORDERLINE_AWAIT_DIRECTOR'
        rationale = f'max adjacent Δp̂ = {max_drop:.1f}pp in [10pp, 20pp); Step 2 supplementary AWAITING DIRECTOR GO/NO-GO'
    else:
        decision = 'NO_GO'
        rationale = f'max adjacent Δp̂ = {max_drop:.1f}pp < 10pp on all pairs'
    candidate_step2_pair = None
    if decision == 'BORDERLINE_AWAIT_DIRECTOR':
        bp = max(adjacent_drops, key=lambda d: d['delta_pp'])
        candidate_step2_pair = {'lower': bp['pair_lower_level'], 'upper': bp['pair_upper_level']}
    return {'per_level': per_level, 'adjacent_drops': adjacent_drops, 'auxiliary_p0_to_l1': auxiliary_p0_to_l1, 'max_adjacent_delta_pp': round(max_drop, 2), 'anchor_5_decision': decision, 'anchor_5_rationale': rationale, 'candidate_step2_pair': candidate_step2_pair, 'step2_auto_launched': False, 'director_gate_required': decision == 'BORDERLINE_AWAIT_DIRECTOR'}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-workers', type=int, default=4, help='Concurrent episode workers (default 4).')
    ap.add_argument('--skip-p0', action='store_true', help='Skip P0 (use existing p0_results.json).')
    ap.add_argument('--skip-p1', action='store_true', help='Skip P1.')
    ap.add_argument('--p0-n-task', type=int, default=30, help='N task per env in P0 (default 30 per Director).')
    ap.add_argument('--p1-n-task', type=int, default=10, help='N task per dep_density level in P1 (default 10).')
    args = ap.parse_args()
    out_dir = ROOT / 'experiments' / 'pilot'
    log_dir = ROOT / 'data' / 'raw_logs'
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    cost_tracker_path = log_dir / 'cost_tracker.jsonl'
    client = LLMClient()
    if not args.skip_p0:
        print(f'[pilot] === P0 START (3 env × mini × Regime I × {args.p0_n_task} task = {3 * args.p0_n_task} ep) ===')
        ct = CostTracker(out_path=cost_tracker_path, phase='pilot', slice_name='pilot_p0_regime_I', emit_every=10)
        cells = build_p0_cells(n_task_per_env=args.p0_n_task)

        def progress(i, n, o: EpisodeOutcome):
            tag = 'OK' if o.error is None else 'ERR'
            success = '✓' if o.success else '✗'
            print(f'[P0 {i}/{n}] {o.cell.task_id} {success}{tag} steps={o.steps} in={o.input_tokens} out={o.output_tokens} cost=${o.cost_usd:.4f}' + (f' err={o.error}' if o.error else ''))
        outcomes_p0 = run_pilot_slice(cells=cells, client=client, step_jsonl_path=log_dir / 'pilot_p0_step.jsonl', episode_jsonl_path=log_dir / 'pilot_p0_episode.jsonl', cost_tracker=ct, n_workers=args.n_workers, progress_fn=progress)
        p0_summary = analyze_p0(outcomes_p0)
        p0_summary['meta'] = {'timestamp_jst': jst_now(), 'model': MODEL, 'memory_mode': MEMORY_MODE, 'regime': 'I_stable', 'stress_config': REGIME_I_STRESS, 'n_workers': args.n_workers, 'cost_tracker_triggered': ct.is_stopped(), 'cost_tracker_stop_reason': ct.stop_reason()}
        with (out_dir / 'p0_results.json').open('w') as f:
            json.dump(p0_summary, f, sort_keys=True, indent=2, ensure_ascii=False)
        print('\n[pilot] === P0 SUMMARY ===')
        for env, info in p0_summary['per_env'].items():
            print(f"  {env}: {info['n_success']}/{info['n']} ({info['success_rate']:.0%}) steps_mean={info['mean_steps']:.1f} cost=${info['total_cost_usd']:.3f} anchor_4={('PASS' if info['anchor_4_passed'] else 'FAIL')}")
        print(f"  overall anchor_4 (≥80% per env): {('PASS' if p0_summary['anchor_4_overall_passed'] else 'FAIL')}")
        print(f"  cost_tracker triggered? {p0_summary['meta']['cost_tracker_triggered']} reason={p0_summary['meta']['cost_tracker_stop_reason']}")
        if not p0_summary['anchor_4_overall_passed']:
            print('\n[pilot] P0 FAILED anchor_4 — env may be too hard. STOPPING. Reporting to Director.')
            return 2
        if ct.is_stopped():
            print(f'\n[pilot] cost_tracker triggered in P0. STOPPING. Reason: {ct.stop_reason()}')
            return 3
    if not args.skip_p1:
        print(f'\n[pilot] === P1 START (anchor_5: graph_nav × mini × dep_density × {args.p1_n_task} task = {len(P1_DEP_LEVELS) * args.p1_n_task} ep) ===')
        ct = CostTracker(out_path=cost_tracker_path, phase='pilot', slice_name='pilot_p1_anchor5_dep_density', emit_every=10)
        cells = build_p1_cells(n_task_per_level=args.p1_n_task)

        def progress(i, n, o: EpisodeOutcome):
            tag = 'OK' if o.error is None else 'ERR'
            success = '✓' if o.success else '✗'
            print(f'[P1 {i}/{n}] {o.cell.task_id} {success}{tag} steps={o.steps} in={o.input_tokens} out={o.output_tokens} cost=${o.cost_usd:.4f}' + (f' err={o.error}' if o.error else ''))
        outcomes_p1 = run_pilot_slice(cells=cells, client=client, step_jsonl_path=log_dir / 'pilot_p1_step.jsonl', episode_jsonl_path=log_dir / 'pilot_p1_episode.jsonl', cost_tracker=ct, n_workers=args.n_workers, progress_fn=progress)
        p0_baseline_rate = None
        p0_path = out_dir / 'p0_results.json'
        if p0_path.exists():
            try:
                p0_doc = json.loads(p0_path.read_text())
                p0_baseline_rate = float(p0_doc.get('per_env', {}).get('graph_nav', {}).get('success_rate', None))
                print(f'[pilot] using P0 graph_nav baseline = {p0_baseline_rate:.3f} as L1 starting reference')
            except Exception as e:
                print(f'[pilot] WARNING could not load P0 baseline ({e}); proceeding without')
        p1_summary = analyze_p1(outcomes_p1, baseline_rate=p0_baseline_rate)
        p1_summary['meta'] = {'timestamp_jst': jst_now(), 'model': MODEL, 'memory_mode': MEMORY_MODE, 'env': 'graph_nav', 'regime_backdrop': 'III_coupled', 'stress_backdrop': P1_BACKDROP, 'dep_density_levels': P1_DEP_LEVELS, 'n_workers': args.n_workers, 'cost_tracker_triggered': ct.is_stopped(), 'cost_tracker_stop_reason': ct.stop_reason()}
        with (out_dir / 'p1_results.json').open('w') as f:
            json.dump(p1_summary, f, sort_keys=True, indent=2, ensure_ascii=False)
        print('\n[pilot] === P1 SUMMARY ===')
        if p1_summary.get('auxiliary_p0_to_l1'):
            a = p1_summary['auxiliary_p0_to_l1']
            print(f"  [aux] P0 baseline (Regime I) → P1 L1 (Regime III backdrop dep=1): {a['p0_baseline_rate']:.1%} → {a['p1_l1_rate']:.1%}, Δp̂={a['delta_pp']:+.1f}pp (NOT in anchor_5 decision)")
        for lev, info in p1_summary['per_level'].items():
            print(f"  dep_density={lev}: {info['n_success']}/{info['n']} ({info['success_rate']:.0%}) steps_mean={info['mean_steps']:.1f} cost=${info['total_cost_usd']:.3f}")
        for d in p1_summary['adjacent_drops']:
            print(f"  Δp̂(L{d['pair_lower_level']}→L{d['pair_upper_level']}) = {d['delta_pp']:+.1f}pp ({d['success_rate_lower']:.0%} → {d['success_rate_upper']:.0%})")
        print(f"  max adjacent Δp̂ = {p1_summary['max_adjacent_delta_pp']:.1f}pp")
        print(f"  anchor_5 decision = {p1_summary['anchor_5_decision']}: {p1_summary['anchor_5_rationale']}")
        if p1_summary['director_gate_required']:
            print(f"  ⚠️  Step 2 NOT auto-launched (Director condition #3 — manual gate). Candidate pair: {p1_summary['candidate_step2_pair']}")
    return 0
if __name__ == '__main__':
    sys.exit(main())
