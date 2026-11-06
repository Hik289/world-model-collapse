from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WSA_DROP_THRESH = 0.5
WSA_DROP_DELTA = 0.15
T10_MAX = 10

def _bad(step: dict) -> bool:
    return not step.get('action_valid', True) or not step.get('world_consistent', True) or (not step.get('dependency_correct', True))

def _tau_o_v1(bads: list[bool], T: int) -> int | None:
    n = len(bads)
    for i in range(0, max(0, n - 4)):
        if sum(bads[i:i + 5]) >= 3:
            return i
    return None

def _tau_o_v2(bads: list[bool], T: int) -> int | None:
    n = len(bads)
    for i in range(0, max(0, n - 5)):
        if sum(bads[i:i + 6]) >= 4:
            return i
    return None

def _tau_o_v3(wsa: list[float], T: int) -> int | None:
    n = len(wsa)
    run = 0
    for i in range(n):
        if wsa[i] < WSA_DROP_THRESH:
            run += 1
            if run >= 5:
                return i - 4
        else:
            run = 0
    return None

def _tau_w(wsa: list[float]) -> int | None:
    for t in range(1, len(wsa)):
        if wsa[t] < WSA_DROP_THRESH and wsa[t] - wsa[t - 1] <= -WSA_DROP_DELTA:
            return t
    return None

def _tau_a(action_valid: list[bool]) -> int | None:
    for t in range(1, len(action_valid)):
        if not action_valid[t]:
            return t
    return None

def analyze_episode(steps: list[dict]) -> dict:
    steps = sorted(steps, key=lambda r: r.get('step', 0))
    if not steps:
        return {}
    T = int(steps[0].get('stress_config', {}).get('T', len(steps)))
    T_max_effective = max(T, len(steps))
    bads = [_bad(s) for s in steps]
    wsa = [float(s.get('world_state_accuracy', 1.0)) for s in steps]
    av = [bool(s.get('action_valid', True)) for s in steps]
    degenerate = T_max_effective <= T10_MAX
    if degenerate:
        return {'T_max': T, 'n_steps': len(steps), 'degenerate_T10': True, 'mean_action_valid': sum(av) / len(av) if av else None, 'tau_o_primary': None, 'tau_o_v2_strict': None, 'tau_o_v3_continuous': None, 'tau_w': None, 'tau_a': None, 'collapse_detected': None, 'cliff_type': 'degenerate_T10'}
    tau_o1 = _tau_o_v1(bads, T)
    tau_o2 = _tau_o_v2(bads, T)
    tau_o3 = _tau_o_v3(wsa, T)
    tw = _tau_w(wsa)
    ta = _tau_a(av)
    collapse_detected = tau_o1 is not None
    return {'T_max': T, 'n_steps': len(steps), 'degenerate_T10': False, 'mean_action_valid': sum(av) / len(av) if av else None, 'min_wsa': min(wsa) if wsa else None, 'final_wsa': wsa[-1] if wsa else None, 'tau_o_primary': tau_o1, 'tau_o_v2_strict': tau_o2, 'tau_o_v3_continuous': tau_o3, 'tau_w': tw, 'tau_a': ta, 'collapse_detected': collapse_detected, '_tw_present': tw is not None, '_ta_present': ta is not None}

def classify_cliff(tw, ta, success: bool, tau_o=None) -> str:
    if tw is not None:
        return 'forgetting'
    if not success and ta is not None:
        return 'capability_invalid'
    if not success and ta is None and (tau_o is not None):
        return 'capability_dependency'
    if not success:
        return 'fail_other'
    return 'none'

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--step-log', required=True)
    ap.add_argument('--episode-log', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    from src.evaluation.jsonl_logger import load_completed_traces
    completed, excluded = load_completed_traces(args.step_log, args.episode_log)
    steps_by_ep = {str(i): steps for i, (_, steps) in enumerate(completed)}
    ep_meta = {str(i): meta for i, (meta, _) in enumerate(completed)}
    cell_eps: dict[tuple, list[dict]] = defaultdict(list)
    per_ep_rows = []
    for tid, steps in steps_by_ep.items():
        meta = ep_meta.get(tid, {})
        sc = meta.get('stress_config', {}).get('state_card')
        dd = meta.get('stress_config', {}).get('dep_density')
        T = meta.get('stress_config', {}).get('T')
        model = meta.get('model', '?')
        success = bool(meta.get('final_success', False))
        res = analyze_episode(steps)
        res['cliff_type'] = res.get('cliff_type') or classify_cliff(res.get('tau_w'), res.get('tau_a'), success, res.get('tau_o_primary'))
        res['task_id'] = meta['task_id']
        res['success'] = success
        res['model'] = model
        res['state_card'] = sc
        res['dep_density'] = dd
        per_ep_rows.append(res)
        cell_eps[model, sc, dd, T].append(res)
    cells = []
    for (model, sc, dd, T), eps in sorted(cell_eps.items(), key=lambda x: (str(x[0][0]), x[0][1] or 0, x[0][3] or 0)):
        n = len(eps)
        succ = sum((1 for e in eps if e['success']))
        tw_vals = [e['tau_w'] for e in eps if e.get('tau_w') is not None]
        ta_vals = [e['tau_a'] for e in eps if e.get('tau_a') is not None]
        to_vals = [e['tau_o_primary'] for e in eps if e.get('tau_o_primary') is not None]
        dtau = [e['tau_a'] - e['tau_w'] for e in eps if e.get('tau_w') is not None and e.get('tau_a') is not None]
        types = defaultdict(int)
        for e in eps:
            types[e['cliff_type']] += 1
        dom = max(types.items(), key=lambda kv: kv[1])[0] if types else 'none'

        def med(v):
            if not v:
                return None
            s = sorted(v)
            m = len(s) // 2
            return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2
        cells.append({'model': model, 'state_card': sc, 'dep_density': dd, 'T': T, 'n_ep': n, 'n_success': succ, 'success_rate': succ / n if n else None, 'n_collapse_detected': sum((1 for e in eps if e.get('collapse_detected'))), 'n_tau_o': len(to_vals), 'median_tau_o': med(to_vals), 'n_tau_w': len(tw_vals), 'median_tau_w': med(tw_vals), 'n_tau_a': len(ta_vals), 'median_tau_a': med(ta_vals), 'n_paired_dtau': len(dtau), 'median_dtau': med(dtau), 'cliff_type_counts': dict(types), 'dominant_cliff_type': dom, 'min_wsa_mean': sum((e['min_wsa'] for e in eps if e.get('min_wsa') is not None)) / n if n else None})
    out = {'cells': cells, 'per_episode': per_ep_rows, 'defs': {'wsa_drop_thresh': WSA_DROP_THRESH, 'wsa_drop_delta': WSA_DROP_DELTA}}
    out['excluded'] = excluded
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"{'model':16} {'sc':>5} {'dd':>3} {'T':>4} {'n':>3} {'succ':>5} {'type':>12} {'τw(n)':>8} {'τa(n)':>8} {'Δτ':>6} {'minWSA':>7}")
    for c in cells:
        tw = f"{c['median_tau_w']}({c['n_tau_w']})" if c['median_tau_w'] is not None else f"-({c['n_tau_w']})"
        ta = f"{c['median_tau_a']}({c['n_tau_a']})" if c['median_tau_a'] is not None else f"-({c['n_tau_a']})"
        dt = f"{c['median_dtau']}" if c['median_dtau'] is not None else '-'
        sr = f"{c['success_rate']:.2f}" if c['success_rate'] is not None else '-'
        mw = f"{c['min_wsa_mean']:.3f}" if c['min_wsa_mean'] is not None else '-'
        print(f"{str(c['model'])[:16]:16} {str(c['state_card']):>5} {str(c['dep_density']):>3} {str(c['T']):>4} {c['n_ep']:>3} {sr:>5} {c['dominant_cliff_type']:>12} {tw:>8} {ta:>8} {dt:>6} {mw:>7}")
    print(f'\nWrote {args.out}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
