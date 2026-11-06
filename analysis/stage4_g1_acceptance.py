from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
import random
import re
import statistics
import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.stage4_g1_trigger.run_stage4_g1_trigger import AXES, PAPER_CONFIGS, atomic_json, build_plan, digest, load_config
RESULTS_PATH = ROOT / 'experiments' / 'stage4_g1_trigger' / 'stage4_results.json'
OUT_JSON = ROOT / 'analysis' / 'stage4_g1_acceptance.json'
OUT_MD = ROOT / 'analysis' / 'stage4_g1_acceptance_table.md'
SC_LEVELS = [5, 10, 20, 40]
DD_LEVELS = [1, 2, 4, 6]

def jst_now() -> str:
    return datetime.now(tz=timezone(timedelta(hours=9))).isoformat()

def wilson_ci(k: int, n: int, alpha: float=0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def newcombe_diff_ci(k1: int, n1: int, k2: int, n2: int, alpha: float=0.05) -> tuple[float, float]:
    l1, u1 = wilson_ci(k1, n1, alpha)
    l2, u2 = wilson_ci(k2, n2, alpha)
    p1, p2 = (k1 / n1, k2 / n2)
    diff = p1 - p2
    lo = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (lo, hi)

def barnard_p(k1: int, n1: int, k2: int, n2: int) -> float:
    from scipy.stats import barnard_exact
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    res = barnard_exact(table, alternative='two-sided')
    return float(res.pvalue)

def legacy_main() -> int:
    data = json.loads(RESULTS_PATH.read_text())
    per_cell = data['per_cell']
    grid_n = {}
    grid_k = {}
    for sc in SC_LEVELS:
        for dd in DD_LEVELS:
            key = f'sc={sc},dd={dd}'
            c = per_cell[key]
            grid_n[sc, dd] = c['n']
            grid_k[sc, dd] = c['n_success']
    wilson95 = {}
    wilson99 = {}
    for sc, dd in grid_n:
        n, k = (grid_n[sc, dd], grid_k[sc, dd])
        wilson95[sc, dd] = wilson_ci(k, n, alpha=0.05)
        wilson99[sc, dd] = wilson_ci(k, n, alpha=0.01)
    sc_adj = []
    for dd in DD_LEVELS:
        for i in range(len(SC_LEVELS) - 1):
            a, b = (SC_LEVELS[i], SC_LEVELS[i + 1])
            ka, na = (grid_k[a, dd], grid_n[a, dd])
            kb, nb = (grid_k[b, dd], grid_n[b, dd])
            sc_adj.append({'axis': 'sc', 'dd': dd, 'lower': a, 'upper': b, 'p_lower': ka / na, 'p_upper': kb / nb, 'delta_pp': round((ka / na - kb / nb) * 100, 2), 'k_lower': ka, 'n_lower': na, 'k_upper': kb, 'n_upper': nb})
    dd_adj = []
    for sc in SC_LEVELS:
        for i in range(len(DD_LEVELS) - 1):
            a, b = (DD_LEVELS[i], DD_LEVELS[i + 1])
            ka, na = (grid_k[sc, a], grid_n[sc, a])
            kb, nb = (grid_k[sc, b], grid_n[sc, b])
            dd_adj.append({'axis': 'dd', 'sc': sc, 'lower': a, 'upper': b, 'p_lower': ka / na, 'p_upper': kb / nb, 'delta_pp': round((ka / na - kb / nb) * 100, 2), 'k_lower': ka, 'n_lower': na, 'k_upper': kb, 'n_upper': nb})
    all_adj = sc_adj + dd_adj
    max_pair = max(all_adj, key=lambda x: x['delta_pp'])
    barn_p = barnard_p(max_pair['k_lower'], max_pair['n_lower'], max_pair['k_upper'], max_pair['n_upper'])
    new99 = newcombe_diff_ci(max_pair['k_lower'], max_pair['n_lower'], max_pair['k_upper'], max_pair['n_upper'], alpha=0.01)
    new95 = newcombe_diff_ci(max_pair['k_lower'], max_pair['n_lower'], max_pair['k_upper'], max_pair['n_upper'], alpha=0.05)
    all_cells = [(sc, dd) for sc in SC_LEVELS for dd in DD_LEVELS]
    extreme = None
    for c1 in all_cells:
        for c2 in all_cells:
            if c1 == c2:
                continue
            ka, na = (grid_k[c1], grid_n[c1])
            kb, nb = (grid_k[c2], grid_n[c2])
            delta = ka / na - kb / nb
            if extreme is None or delta > extreme['delta']:
                extreme = {'delta': delta, 'p_lower': ka / na, 'p_upper': kb / nb, 'c_lower': {'sc': c1[0], 'dd': c1[1]}, 'c_upper': {'sc': c2[0], 'dd': c2[1]}, 'k_lower': ka, 'n_lower': na, 'k_upper': kb, 'n_upper': nb}
    ext_barn = barnard_p(extreme['k_lower'], extreme['n_lower'], extreme['k_upper'], extreme['n_upper'])
    ext_new99 = newcombe_diff_ci(extreme['k_lower'], extreme['n_lower'], extreme['k_upper'], extreme['n_upper'], alpha=0.01)
    corner_low = (5, 1)
    corner_high = (40, 6)
    wlow_low, wlow_hi = wilson95[corner_low]
    whigh_low, whigh_hi = wilson95[corner_high]
    corner_ci_separated = wlow_low > whigh_hi
    monotone_sc_within_dd1 = all((per_cell[f'sc={SC_LEVELS[i]},dd=1']['success_rate'] >= per_cell[f'sc={SC_LEVELS[i + 1]},dd=1']['success_rate'] for i in range(len(SC_LEVELS) - 1)))
    monotone_dd_within_sc10 = all((per_cell[f'sc=10,dd={DD_LEVELS[i]}']['success_rate'] >= per_cell[f'sc=10,dd={DD_LEVELS[i + 1]}']['success_rate'] for i in range(len(DD_LEVELS) - 1)))
    g1_pass = (monotone_sc_within_dd1 or monotone_dd_within_sc10) and max_pair['delta_pp'] >= 20.0 and corner_ci_separated and (barn_p < 0.001)
    out = {'generated_jst': jst_now(), 'source': str(RESULTS_PATH.relative_to(ROOT)), 'grid_axes': {'state_cards': SC_LEVELS, 'dep_densities': DD_LEVELS}, 'grid_4x4_success_rate': {f'sc={sc},dd={dd}': grid_k[sc, dd] / grid_n[sc, dd] for sc in SC_LEVELS for dd in DD_LEVELS}, 'wilson_95_per_cell': {f'sc={sc},dd={dd}': {'lo': round(wilson95[sc, dd][0], 4), 'hi': round(wilson95[sc, dd][1], 4)} for sc in SC_LEVELS for dd in DD_LEVELS}, 'wilson_99_per_cell': {f'sc={sc},dd={dd}': {'lo': round(wilson99[sc, dd][0], 4), 'hi': round(wilson99[sc, dd][1], 4)} for sc in SC_LEVELS for dd in DD_LEVELS}, 'adjacent_sc_axis': sc_adj, 'adjacent_dd_axis': dd_adj, 'max_drop_adjacent_pair': {**max_pair, 'barnard_exact_p_two_sided': barn_p, 'newcombe_99_ci_pp': [round(new99[0] * 100, 2), round(new99[1] * 100, 2)], 'newcombe_95_ci_pp': [round(new95[0] * 100, 2), round(new95[1] * 100, 2)]}, 'extreme_corner_pair_any': {**{k: v for k, v in extreme.items() if k != 'delta'}, 'delta_pp': round(extreme['delta'] * 100, 2), 'barnard_exact_p_two_sided': ext_barn, 'newcombe_99_ci_pp': [round(ext_new99[0] * 100, 2), round(ext_new99[1] * 100, 2)]}, 'g1_acceptance': {'monotone_sc_within_dd1': monotone_sc_within_dd1, 'monotone_dd_within_sc10': monotone_dd_within_sc10, 'max_adjacent_delta_pp': max_pair['delta_pp'], 'corner_pair': {'low_stress': {'sc': 5, 'dd': 1, 'wilson95': [wlow_low, wlow_hi]}, 'high_stress': {'sc': 40, 'dd': 6, 'wilson95': [whigh_low, whigh_hi]}, 'ci_separated': corner_ci_separated}, 'barnard_p_max_pair': barn_p, 'verdict_g1_pass': g1_pass}}

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(x) for x in o]
        if isinstance(o, tuple):
            return [_clean(x) for x in o]
        if hasattr(o, 'item'):
            return o.item()
        return o
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(_clean(out), sort_keys=True, indent=2, ensure_ascii=False))
    md = []
    md.append(f'# Stage 4 G1-Trigger Acceptance Analysis\n')
    md.append(f'Generated: {jst_now()}\n')
    md.append(f'Source: `{RESULTS_PATH.relative_to(ROOT)}`\n\n')
    md.append('## 4×4 success_rate grid (rows=state_card, cols=dep_density)\n\n')
    md.append('| sc\\dd | 1 | 2 | 4 | 6 |\n|------:|---:|---:|---:|---:|')
    for sc in SC_LEVELS:
        row = [f'| **{sc}** ']
        for dd in DD_LEVELS:
            n, k = (grid_n[sc, dd], grid_k[sc, dd])
            row.append(f' | {k}/{n} ({100 * k / n:.0f}%) ')
        md.append(''.join(row) + ' |')
    md.append('\n')
    md.append('## Wilson 95% CI per cell (lo, hi)\n\n')
    md.append('| sc\\dd | 1 | 2 | 4 | 6 |\n|------:|:---:|:---:|:---:|:---:|')
    for sc in SC_LEVELS:
        row = [f'| **{sc}** ']
        for dd in DD_LEVELS:
            lo, hi = wilson95[sc, dd]
            row.append(f' | [{lo:.3f}, {hi:.3f}] ')
        md.append(''.join(row) + ' |')
    md.append('\n')
    md.append('## Max-drop adjacent pair\n\n')
    mp = max_pair
    md.append(f"- Axis: **{mp['axis']}**, {('dd=' + str(mp['dd']) if mp['axis'] == 'sc' else 'sc=' + str(mp['sc']))}, {mp['lower']} → {mp['upper']}\n")
    md.append(f"- success: {mp['p_lower']:.0%} → {mp['p_upper']:.0%}, Δp̂ = **{mp['delta_pp']:+.1f}pp**\n")
    md.append(f'- Barnard exact (two-sided) **p = {barn_p:.3e}**\n')
    md.append(f'- Newcombe 99% CI: [{round(new99[0] * 100, 2)}pp, {round(new99[1] * 100, 2)}pp]\n')
    md.append(f'- Newcombe 95% CI: [{round(new95[0] * 100, 2)}pp, {round(new95[1] * 100, 2)}pp]\n\n')
    md.append('## Extreme corner pair (any two cells)\n\n')
    md.append(f"- {extreme['c_lower']} → {extreme['c_upper']}\n")
    md.append(f"- success: {extreme['p_lower']:.0%} → {extreme['p_upper']:.0%}, Δ = **{round(extreme['delta'] * 100, 2):+.1f}pp**\n")
    md.append(f'- Barnard exact (two-sided) **p = {ext_barn:.3e}**\n')
    md.append(f'- Newcombe 99% CI: [{round(ext_new99[0] * 100, 2)}pp, {round(ext_new99[1] * 100, 2)}pp]\n\n')
    md.append('## G1 Acceptance Verdict\n\n')
    md.append(f'- monotone(sc | dd=1): **{monotone_sc_within_dd1}**\n')
    md.append(f'- monotone(dd | sc=10): **{monotone_dd_within_sc10}**\n')
    md.append(f"- max adjacent Δp̂: **{mp['delta_pp']:.1f}pp** (≥20pp threshold = {mp['delta_pp'] >= 20})\n")
    md.append(f'- Corner pair (sc=5,dd=1) Wilson95 vs (sc=40,dd=6): [{wlow_low:.3f},{wlow_hi:.3f}] vs [{whigh_low:.3f},{whigh_hi:.3f}] → separated = **{corner_ci_separated}**\n')
    md.append(f'- Barnard p on max pair: **{barn_p:.3e}** (<0.001 = {barn_p < 0.001})\n')
    md.append(f"\n### **G1 VERDICT: {('PASS ✅' if g1_pass else 'FAIL ❌')}**\n")
    OUT_MD.write_text('\n'.join(md))
    print(f'Wrote {OUT_JSON}')
    print(f'Wrote {OUT_MD}')
    print()
    print(f"G1 verdict: {('PASS ✅' if g1_pass else 'FAIL ❌')}")
    print(f"Max-drop adjacent pair: {mp['axis']} dd={mp.get('dd', '-')}sc={mp.get('sc', '-')} {mp['lower']}→{mp['upper']}: {mp['p_lower']:.0%}→{mp['p_upper']:.0%} ({mp['delta_pp']:+.1f}pp)")
    print(f'  Barnard p = {barn_p:.3e}')
    print(f'  Newcombe 99% CI: [{round(new99[0] * 100, 2)}, {round(new99[1] * 100, 2)}]pp')
    print(f"Extreme corner: {extreme['c_lower']}→{extreme['c_upper']}: {round(extreme['delta'] * 100, 2):+.1f}pp")
    print(f'  Barnard p = {ext_barn:.3e}')
    print(f'  Newcombe 99% CI: [{round(ext_new99[0] * 100, 2)}, {round(ext_new99[1] * 100, 2)}]pp')
    return 0

def wilson(k, n):
    if n <= 0 or not 0 <= k <= n:
        raise ValueError('Invalid binomial counts')
    z = 1.959963984540054
    p, den = (k / n, 1 + z * z / n)
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))

def mn_score(k1, n1, k2, n2, margin=0.3):
    if not (n1 > 0 and n2 > 0 and (0 <= k1 <= n1) and (0 <= k2 <= n2) and (0 <= margin < 1)):
        raise ValueError('Invalid counts or null margin')

    def binomial_loglik(k, n, p):
        if p <= 0:
            return 0.0 if k == 0 else -math.inf
        if p >= 1:
            return 0.0 if k == n else -math.inf
        return k * math.log(p) + (n - k) * math.log1p(-p)

    def objective(q):
        return binomial_loglik(k1, n1, q + margin) + binomial_loglik(k2, n2, q)
    lo, hi = (0.0, 1.0 - margin)
    for _ in range(100):
        a, b = (lo + (hi - lo) / 3, hi - (hi - lo) / 3)
        if objective(a) < objective(b):
            lo = a
        else:
            hi = b
    q = max((0.0, 1.0 - margin, (lo + hi) / 2), key=objective)
    p = q + margin
    variance = p * (1 - p) / n1 + q * (1 - q) / n2
    variance *= (n1 + n2) / (n1 + n2 - 1)
    delta = k1 / n1 - k2 / n2
    z = (delta - margin) / math.sqrt(max(variance, 1e-300))
    return {'drop': delta, 'margin': margin, 'z': z, 'p_value': math.erfc(z / math.sqrt(2)) / 2, 'p1_constrained': p, 'p2_constrained': q}

def adjust_pvalues(rows, alpha=0.01, q=0.05):
    m = len(rows)
    ordered = sorted(range(m), key=lambda i: rows[i]['p_value'])
    previous = 1.0
    for rank0 in range(m - 1, -1, -1):
        row = rows[ordered[rank0]]
        previous = min(previous, row['p_value'] * m / (rank0 + 1))
        row['p_bh'] = previous
        row['p_bonferroni'] = min(1.0, row['p_value'] * m)
        row['significant_bh'] = previous < q
        row['significant_bonferroni'] = row['p_bonferroni'] < alpha

def crossing(points, threshold=0.5):
    if len(points) < 2:
        return None
    points = sorted(points)
    for (x1, p1), (x2, p2) in zip(points, points[1:]):
        if p1 >= threshold and p2 <= threshold and (p1 > p2):
            return x1 + (threshold - p1) * (x2 - x1) / (p2 - p1)
    return None

def bootstrap_crossing(cells, repetitions=1000, seed=42, threshold=0.5):
    if repetitions <= 0:
        raise ValueError('Bootstrap repetitions must be positive')
    estimate = crossing([(c['state_size'], c['success_rate']) for c in cells], threshold)
    rng = random.Random(seed)
    draws = []
    for _ in range(repetitions):
        points = [(c['state_size'], sum((rng.random() < c['success_rate'] for _ in range(c['n']))) / c['n']) for c in cells]
        value = crossing(points, threshold)
        if value is not None:
            draws.append(value)
    draws.sort()
    enough = estimate is not None and len(draws) >= 20
    return {'state_size_star': estimate, 'threshold': threshold, 'ci_low': draws[int(0.025 * (len(draws) - 1))] if enough else None, 'ci_high': draws[int(0.975 * (len(draws) - 1))] if enough else None, 'bracketed_fraction': len(draws) / repetitions, 'bootstrap_repetitions': repetitions, 'method': 'first downward linear crossing; pointwise within-cell bootstrap; CI conditional on a bracket'}
GROUP = ('env_name', 'model', *AXES)

def group_record(job):
    return {'env_name': job['env_name'], 'model': job['model'], **job['stress']}

def key(record, fields=GROUP):
    return tuple((record[k] for k in fields))

def failure_times(steps, window, threshold=0.5):
    if type(window) is not int or window <= 0 or (not 0 < threshold < 1):
        raise ValueError('Invalid onset window or threshold')
    accuracy = [row['world_state_accuracy'] for row in steps]
    tau_w = next((i + 1 for i in range(window - 1, len(steps)) if sum(accuracy[i - window + 1:i + 1]) / window < threshold), None)
    tau_a = next((i + 1 for i, row in enumerate(steps) if not row['action_valid']), None)
    return (tau_w, tau_a)

def load_completed(directory, manifest):
    from src.evaluation.runner import jaccard, world_state_facts
    expected = {job['job_id']: job for job in manifest['jobs']}
    if len(expected) != len(manifest['jobs']):
        raise ValueError('Duplicate job IDs in manifest')
    completed = []
    for path in sorted((directory / 'episodes').glob('*.json')):
        record = json.loads(path.read_text())
        job = expected.get(path.stem)
        if job is None or record['job'] != job or record['protocol'] != manifest['protocol']:
            raise ValueError(f'Manifest mismatch in {path.name}')
        episode, steps = (record['episode'], record['steps'])
        if not steps or len(steps) != episode['steps_taken'] or len(steps) > job['stress']['horizon']:
            raise ValueError(f'Incomplete episode {path.name}')
        if type(episode['final_success']) is not bool:
            raise ValueError(f'Non-boolean success in {path.name}')
        for row in [episode, *steps]:
            for field in ('task_id', 'task_seed', 'decoding_seed', 'model', 'env_name', 'stress_config'):
                if row[field] != job[field]:
                    raise ValueError(f'Episode identity mismatch: {field} in {path.name}')
            if row['run_id'] != job['job_id']:
                raise ValueError(f'Run identity mismatch in {path.name}')
        for i, row in enumerate(steps):
            if row['step'] != i or type(row['action_valid']) is not bool:
                raise ValueError(f'Invalid or discontinuous steps in {path.name}')
            value = row['world_state_accuracy']
            if type(value) not in (int, float) or not math.isfinite(value) or (not 0 <= value <= 1):
                raise ValueError(f'Invalid fidelity in {path.name}')
            gold = row.get('gold_world_state_before')
            if not isinstance(gold, dict) or not gold:
                raise ValueError(f'Missing pre-action gold state in {path.name}')
            recomputed = jaccard(world_state_facts(row['agent_world_state']), world_state_facts(gold))
            if not math.isclose(value, recomputed, abs_tol=1e-10):
                raise ValueError(f'Fidelity does not match the pre-action state in {path.name}')
        completed.append(record)
    return completed

def load_boundary(path, manifest):
    if path is None:
        return None
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('Boundary selection must be a nonempty JSON list')
    fields = ('env_name', 'model', 'state_size', 'state_dependency')
    allowed = {key(group_record(job), fields) for job in manifest['jobs']}
    selected = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(fields):
            raise ValueError(f'Boundary cells must have exactly {fields}')
        entry = key(row, fields)
        if entry not in allowed or entry in selected:
            raise ValueError(f'Unknown or duplicated boundary cell: {entry}')
        selected.add(entry)
    return selected

def summarize(manifest, completed, window, boundary=None, repetitions=1000):
    planned, observed = (defaultdict(int), defaultdict(list))
    for job in manifest['jobs']:
        planned[key(group_record(job))] += 1
    for record in completed:
        observed[key(group_record(record['job']))].append(record)
    cells = []
    for cell_key, planned_n in sorted(planned.items()):
        records = observed[cell_key]
        n = len(records)
        successes = sum((r['episode']['final_success'] for r in records))
        ci = wilson(successes, n) if n else (None, None)
        cells.append({**dict(zip(GROUP, cell_key)), 'planned_n': planned_n, 'n': n, 'successes': successes, 'success_rate': successes / n if n else None, 'ci_low': ci[0], 'ci_high': ci[1], 'complete': n == planned_n})
    tests, incomplete_pairs = ([], 0)
    for axis in ('state_size', 'state_dependency'):
        group_fields = tuple((f for f in GROUP if f != axis))
        lines = defaultdict(list)
        for cell in cells:
            lines[key(cell, group_fields)].append(cell)
        for line in lines.values():
            line.sort(key=lambda c: c[axis])
            for left, right in zip(line, line[1:]):
                if not left['complete'] or not right['complete']:
                    incomplete_pairs += 1
                    continue
                tests.append({**{f: left[f] for f in group_fields}, 'axis': axis, 'lower': left[axis], 'upper': right[axis], **mn_score(left['successes'], left['n'], right['successes'], right['n'])})
    if incomplete_pairs == 0:
        adjust_pvalues(tests)
    onsets, groups = ([], defaultdict(list))
    boundary_fields = ('env_name', 'model', 'state_size', 'state_dependency')
    summary_fields = ('env_name', 'model', 'horizon', 'branching', 'observation', 'mutation')
    for record in completed:
        job, episode, steps = (record['job'], record['episode'], record['steps'])
        info = group_record(job)
        tau_w, tau_a = failure_times(steps, window)
        selected = boundary is None or key(info, boundary_fields) in boundary
        collapsed = not episode['final_success']
        paired = collapsed and tau_w is not None and (tau_a is not None)
        onset = {**info, 'job_id': job['job_id'], 'tau_w': tau_w, 'tau_a': tau_a, 'collapsed': collapsed, 'selected': selected, 'paired': paired, 'lead': tau_a - tau_w if paired else None}
        onsets.append(onset)
        if selected:
            groups[key(info, summary_fields)].append((onset, steps))
    failure_order, fidelity = ([], [])
    for group_key, records in sorted(groups.items()):
        info = dict(zip(summary_fields, group_key))
        paired = [(o, steps) for o, steps in records if o['paired']]
        n = len(paired)
        leads = [o['lead'] for o, _ in paired]
        failure_order.append({**info, 'completed_n': len(records), 'collapsed_n': sum((o['collapsed'] for o, _ in records)), 'paired_n': n, 'missing_world_onset_n': sum((o['collapsed'] and o['tau_w'] is None for o, _ in records)), 'missing_action_onset_n': sum((o['collapsed'] and o['tau_a'] is None for o, _ in records)), 'world_first_pct': 100 * sum((v > 0 for v in leads)) / n if n else None, 'same_step_pct': 100 * sum((v == 0 for v in leads)) / n if n else None, 'action_first_pct': 100 * sum((v < 0 for v in leads)) / n if n else None, 'median_lead': statistics.median(leads) if n else None})
        action_observed = [(o, steps) for o, steps in records if o['collapsed'] and o['tau_a'] is not None]
        for offset in (-3, -2, -1, 0):
            values = [steps[o['tau_a'] - 1 + offset]['world_state_accuracy'] for o, steps in action_observed if o['tau_a'] - 1 + offset >= 0]
            fidelity.append({**info, 'relative_step': offset, 'n': len(values), 'mean_fidelity': statistics.mean(values) if values else None})
    critical = []
    lines = defaultdict(list)
    fields = tuple((f for f in GROUP if f != 'state_size'))
    for cell in cells:
        if cell['state_dependency'] == 1:
            lines[key(cell, fields)].append(cell)
    for line_key, line in sorted(lines.items()):
        if all((c['complete'] for c in line)) and len(line) >= 2:
            critical.append({**dict(zip(fields, line_key)), **bootstrap_crossing(line, repetitions=repetitions)})
    pilot_lines = defaultdict(list)
    pilot_fields = tuple((f for f in GROUP if f not in ('state_dependency', 'model')))
    if manifest['config']['name'] == 'paper_pilot_sd':
        for cell in cells:
            pilot_lines[key(cell, pilot_fields)].append(cell)
    pilots = []
    for line_key, line in sorted(pilot_lines.items()):
        by_model = defaultdict(list)
        for cell in line:
            by_model[cell['model']].append(cell)
        metrics = []
        for model, series in sorted(by_model.items()):
            series.sort(key=lambda c: c['state_dependency'])
            complete = all((c['complete'] for c in series)) and len(series) >= 2
            drops = [a['success_rate'] - b['success_rate'] for a, b in zip(series, series[1:])] if complete else []
            metrics.append({'model': model, 'complete': complete, 'nonincreasing': all((d >= -1e-12 for d in drops)) if complete else None, 'max_drop': max(drops) if complete else None})
        passed = len(metrics) >= 2 and all((m['complete'] and m['nonincreasing'] and (m['max_drop'] >= 0.2) for m in metrics))
        pilots.append({**dict(zip(pilot_fields, line_key)), 'models': metrics, 'passes_pilot_criteria': passed})
    return {'protocol': manifest['protocol'], 'config_hash': manifest['config_hash'], 'source_digest': manifest['source_digest'], 'window': window, 'fidelity_threshold': 0.5, 'time_index': 'one-based, pre-action state compared with validity of the same action', 'collapse_definition': 'final_success is false', 'selection': 'all tested cells' if boundary is None else sorted(boundary), 'completed_episodes': len(completed), 'planned_episodes': manifest['episode_count'], 'incomplete_adjacent_pairs': incomplete_pairs, 'multiple_testing_family': 'all planned adjacent SS and SD pairs; corrections withheld until complete', 'cells': cells, 'cliff_tests': tests, 'onsets': onsets, 'failure_order': failure_order, 'fidelity_before_action': fidelity, 'critical_points': critical, 'pilot_checks': pilots}

def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        if rows:
            fields = list(dict.fromkeys((k for row in rows for k in row)))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

def analyze_run(args):
    if args.window <= 0 or args.bootstrap_repetitions <= 0:
        raise ValueError('Window and bootstrap repetitions must be positive')
    manifest = json.loads((args.run / 'manifest.json').read_text())
    if digest(manifest['config']) != manifest['config_hash']:
        raise ValueError('Manifest config hash mismatch')
    completed = load_completed(args.run, manifest)
    if not completed:
        raise ValueError('No completed episodes; a plan is not experimental data')
    boundary = load_boundary(args.boundary_cells, manifest)
    report = summarize(manifest, completed, args.window, boundary, args.bootstrap_repetitions)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / 'report.json', report)
    for name in ('cells', 'cliff_tests', 'onsets', 'failure_order', 'fidelity_before_action', 'critical_points'):
        write_csv(args.output / f'{name}.csv', report[name])
    print(f"Analyzed {len(completed)}/{manifest['episode_count']} completed episodes; {args.output / 'report.json'}")
DISPLAY = {'stateful_puzzle': 'StatefulPuzzle', 'graph_nav': 'GraphNav', 'tool_dag': 'ToolDAG'}
BACKDROP = ('horizon', 'branching', 'observation', 'mutation')

def plot_report(report_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    report = json.loads(report_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 400})

    def save(fig, stem):
        fig.savefig(output / f'{stem}.pdf', bbox_inches='tight')
        fig.savefig(output / f'{stem}.png', bbox_inches='tight', dpi=400)
        plt.close(fig)

    def slug(values):
        return re.sub('[^A-Za-z0-9_-]+', '_', '_'.join(map(str, values)))
    facets = defaultdict(list)
    for cell in report['cells']:
        facets[cell['env_name'], *(cell[k] for k in BACKDROP)].append(cell)
    for facet, cells in sorted(facets.items()):
        models = sorted({c['model'] for c in cells})
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models) + 0.3, 4.0), squeeze=False, layout='constrained')
        for ax, model in zip(axes[0], models):
            rows = [c for c in cells if c['model'] == model]
            sizes = sorted({c['state_size'] for c in rows})
            deps = sorted({c['state_dependency'] for c in rows})
            values = np.full((len(sizes), len(deps)), np.nan)
            for row in rows:
                i, j = (sizes.index(row['state_size']), deps.index(row['state_dependency']))
                if row['success_rate'] is not None:
                    values[i, j] = row['success_rate']
                    mark = '' if row['complete'] else '*'
                    ax.text(j, i, f"{row['success_rate']:.2f}{mark}", ha='center', va='center', fontsize=10)
            cmap = plt.get_cmap('RdYlGn').copy()
            cmap.set_bad('#E6E6E6')
            im = ax.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect='auto')
            for test in report['cliff_tests']:
                if not test.get('significant_bonferroni') or test['env_name'] != facet[0] or test['model'] != model:
                    continue
                if tuple((test[k] for k in BACKDROP)) != facet[1:]:
                    continue
                if test['axis'] == 'state_size':
                    y = sizes.index(test['lower']) + 0.5
                    x = deps.index(test['state_dependency'])
                    ax.plot([x - 0.47, x + 0.47], [y, y], color='#202020', lw=2.4)
                else:
                    x = deps.index(test['lower']) + 0.5
                    y = sizes.index(test['state_size'])
                    ax.plot([x, x], [y - 0.47, y + 0.47], color='#202020', lw=2.4)
            finite = [r for r in rows if r['complete'] and r['success_rate'] is not None]
            if finite and min((r['success_rate'] for r in finite)) <= 0.5 <= max((r['success_rate'] for r in finite)):
                c = min(finite, key=lambda r: (abs(r['success_rate'] - 0.5), r['state_size'], r['state_dependency']))
                ax.add_patch(Rectangle((deps.index(c['state_dependency']) - 0.45, sizes.index(c['state_size']) - 0.45), 0.9, 0.9, fill=False, edgecolor='#7B3294', linewidth=2, linestyle='--'))
            ax.set(xticks=range(len(deps)), xticklabels=deps, yticks=range(len(sizes)), yticklabels=sizes, xlabel='State dependency (SD)', ylabel='State size (SS)', title=model)
        fig.colorbar(im, ax=list(axes[0]), label='Success rate', shrink=0.85)
        backdrop = f'T={facet[1]}, branching={facet[2]}, {facet[3]}, {facet[4]}'
        fig.suptitle(f'{DISPLAY.get(facet[0], facet[0])} | {backdrop}')
        fig.supxlabel('Black: corrected cliff; purple: cell nearest 0.5; *: incomplete cell', fontsize=8)
        save(fig, 'phase_' + slug(facet))
    temporal = defaultdict(list)
    for row in report['failure_order']:
        temporal[tuple((row[k] for k in BACKDROP))].append(row)
    for backdrop, rows in sorted(temporal.items()):
        rows = [row for row in rows if row['paired_n']]
        if not rows:
            continue
        labels = [f"{DISPLAY.get(r['env_name'], r['env_name'])}\n{r['model']} (n={r['paired_n']})" for r in rows]
        fig, ax = plt.subplots(figsize=(8.5, max(3, 0.65 * len(rows))), layout='constrained')
        left = np.zeros(len(rows))
        for name, title, color in (('world_first_pct', 'World first', '#377EB8'), ('same_step_pct', 'Same step', '#BDBDBD'), ('action_first_pct', 'Action first', '#E66101')):
            values = np.array([r[name] for r in rows])
            ax.barh(range(len(rows)), values, left=left, label=title, color=color, height=0.7)
            for i, (v, start) in enumerate(zip(values, left)):
                if v >= 6:
                    ax.text(start + v / 2, i, f'{v:.1f}%', ha='center', va='center', fontsize=9)
            left += values
        ax.set(yticks=range(len(rows)), yticklabels=labels, xlim=(0, 100), xlabel='Paired collapsed rollouts (%)')
        ax.invert_yaxis()
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
        save(fig, 'failure_order_' + slug(backdrop))
    trajectories = defaultdict(list)
    for row in report['fidelity_before_action']:
        trajectories[tuple((row[k] for k in BACKDROP))].append(row)
    for backdrop, rows in sorted(trajectories.items()):
        environments = sorted({r['env_name'] for r in rows})
        fig, axes = plt.subplots(1, len(environments), figsize=(4 * len(environments), 3.4), squeeze=False, layout='constrained')
        for ax, env in zip(axes[0], environments):
            for model in sorted({r['model'] for r in rows if r['env_name'] == env}):
                line = sorted([r for r in rows if r['env_name'] == env and r['model'] == model], key=lambda r: r['relative_step'])
                ax.plot([r['relative_step'] for r in line], [r['mean_fidelity'] if r['mean_fidelity'] is not None else np.nan for r in line], marker='o', linewidth=2, label=model)
            ax.axvline(0, color='#777777', linestyle='--', linewidth=1)
            ax.set(title=DISPLAY.get(env, env), ylim=(0, 1.02), xticks=[-3, -2, -1, 0], xlabel='Steps relative to first invalid action', ylabel='World-state fidelity')
            ax.legend(frameon=False, fontsize=8)
            ax.grid(axis='y', alpha=0.2)
        save(fig, 'fidelity_before_action_' + slug(backdrop))
    for estimate in report['critical_points']:
        rows = sorted([c for c in report['cells'] if all((c[k] == estimate[k] for k in ('env_name', 'model', 'state_dependency', *BACKDROP)))], key=lambda c: c['state_size'])
        x, y = ([r['state_size'] for r in rows], [r['success_rate'] for r in rows])
        fig, ax = plt.subplots(figsize=(5.2, 3.6), layout='constrained')
        ax.plot(x, y, color='#BD3037', marker='o', linewidth=2)
        ax.fill_between(x, [r['ci_low'] for r in rows], [r['ci_high'] for r in rows], color='#BD3037', alpha=0.16, label='95% Wilson interval')
        ax.axhline(0.5, color='#777777', linestyle='--', linewidth=1)
        if estimate['state_size_star'] is not None:
            ax.axvline(estimate['state_size_star'], color='#BD3037', linestyle='--', label=f"SS*={estimate['state_size_star']:.2f}")
        ax.set(xlabel='State size (SS)', ylabel='Success rate', ylim=(-0.02, 1.02), title=f"{DISPLAY[estimate['env_name']]} | {estimate['model']} | SD=1")
        ax.legend(frameon=False, fontsize=8)
        save(fig, 'critical_' + slug([estimate[k] for k in ('env_name', 'model', *BACKDROP)]))

def run_self_tests():
    import copy
    import json
    from pathlib import Path
    import tempfile
    import unittest
    from unittest.mock import patch
    from src.agents.llm_client import LLMClient, RawCallResult
    from src.agents.base import BaseAgent, CallOutcome
    from src.agents.prompts import planner_user
    from src.environments import ENV_REGISTRY
    from src.evaluation.runner import EpisodeContext, run_episode
    from experiments.stage4_g1_trigger.run_stage4_g1_trigger import BufferWriter, execute

    class ProtocolTests(unittest.TestCase):

        def setUp(self):
            self.config = load_config(PAPER_CONFIGS['main_grid'])

        def test_main_grid_count_and_paired_seeds(self):
            plan = build_plan(self.config)
            self.assertEqual(plan['episode_count'], 9600)
            self.assertEqual(plan['unique_tasks'], 4800)
            tasks = {}
            for job in plan['jobs']:
                tasks.setdefault(job['task_id'], []).append(job)
            self.assertEqual(len({jobs[0]['task_seed'] for jobs in tasks.values()}), 4800)
            for jobs in tasks.values():
                self.assertEqual(len(jobs), 2)
                self.assertEqual(jobs[0]['task_seed'], jobs[1]['task_seed'])
                self.assertNotEqual(jobs[0]['job_id'], jobs[1]['job_id'])
            self.assertEqual(plan, build_plan(self.config))

        def test_configs_supported_and_ablations_deduplicated(self):
            for source in PAPER_CONFIGS.values():
                config = load_config(source)
                plan = build_plan(config)
                self.assertEqual(len({j['job_id'] for j in plan['jobs']}), plan['episode_count'])
            config = load_config(PAPER_CONFIGS['ablations'])
            self.assertEqual(build_plan(config)['episode_count'], 1300)

        def test_invalid_config_rejected(self):
            for field, value in (('archetypes', 0), ('variants', True), ('environments', ['unknown']), ('models', ['gpt-x', 'gpt-x'])):
                config = copy.deepcopy(self.config)
                config[field] = value
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / 'config.json'
                    atomic_json(path, config)
                    with self.assertRaises(ValueError):
                        load_config(path)

        def test_restored_environment_seed_determinism(self):
            task = {'stress_config': {'state_card': 5, 'dep_density': 2, 'T': 2, 'branching': 4, 'obs_noise': 'clean', 'mut_rate': 'static'}}
            for name in ('graph_nav', 'tool_dag'):
                a, b = (ENV_REGISTRY[name](), ENV_REGISTRY[name]())
                self.assertEqual(a.reset(task, 12).to_dict(), b.reset(task, 12).to_dict())
                self.assertEqual(a.get_gold_state(), b.get_gold_state())

        def test_all_action_templates_reach_planner(self):
            actions = [f'inspect(v{i})' for i in range(100)] + ['move(goal)']
            self.assertIn('move(goal)', planner_user('observation', {}, actions))

    class StatisticsTests(unittest.TestCase):

        def test_window_and_action_use_same_clock(self):
            rows = [{'world_state_accuracy': v, 'action_valid': i != 4} for i, v in enumerate([1, 0.2, 0.2, 0.2, 0.2])]
            self.assertEqual(failure_times(rows, 2), (3, 5))
            rows[1]['world_state_accuracy'] = 0.8
            self.assertEqual(failure_times(rows, 2), (4, 5))

        def test_strict_threshold_and_censoring(self):
            rows = [{'world_state_accuracy': 0.5, 'action_valid': True}] * 3
            self.assertEqual(failure_times(rows, 2), (None, None))
            rows[0] = {'world_state_accuracy': 0.1, 'action_valid': False}
            self.assertEqual(failure_times(rows, 4), (None, 1))

        def test_mn_against_independent_implementation(self):
            import math
            from scipy.optimize import brentq
            for counts in ((99, 100, 1, 100), (78, 100, 45, 100), (45, 51, 3, 31)):
                ours = mn_score(*counts)
                k1, n1, k2, n2 = counts
                score = lambda q: k1 / (q + 0.3) - (n1 - k1) / (0.7 - q) + k2 / q - (n2 - k2) / (1 - q)
                q = brentq(score, 1e-12, 0.7 - 1e-12)
                variance = ((q + 0.3) * (0.7 - q) / n1 + q * (1 - q) / n2) * (n1 + n2) / (n1 + n2 - 1)
                expected = 0.5 * math.erfc((k1 / n1 - k2 / n2 - 0.3) / math.sqrt(2 * variance))
                self.assertAlmostEqual(ours['p2_constrained'], q, places=6)
                self.assertAlmostEqual(ours['p_value'], expected, places=6)

        def test_mn_boundary_likelihoods(self):
            result = mn_score(100, 100, 0, 100)
            self.assertAlmostEqual(result['p1_constrained'], 0.65, places=6)
            self.assertAlmostEqual(result['p2_constrained'], 0.35, places=6)
            self.assertLess(result['p_value'], 0.01)
            zero = mn_score(0, 100, 0, 100)
            self.assertEqual(zero['p1_constrained'], 0.3)
            self.assertEqual(zero['p2_constrained'], 0)
            one = mn_score(100, 100, 100, 100)
            self.assertEqual(one['p1_constrained'], 1)
            self.assertAlmostEqual(one['p2_constrained'], 0.7)
            self.assertGreater(zero['p_value'], 0.99)
            self.assertGreater(one['p_value'], 0.99)

        def test_wilson_and_multiple_testing(self):
            lo, hi = wilson(0, 100)
            self.assertAlmostEqual(lo, 0)
            self.assertGreater(hi, 0)
            rows = [{'p_value': p} for p in [0.001, 0.02, 0.9]]
            adjust_pvalues(rows)
            self.assertTrue(rows[0]['significant_bonferroni'])
            self.assertFalse(rows[1]['significant_bonferroni'])
            self.assertAlmostEqual(rows[1]['p_bh'], 0.03)

        def test_localization_does_not_extrapolate_or_use_upward_crossings(self):
            self.assertEqual(crossing([(10, 0.8), (20, 0.2)]), 15)
            self.assertIsNone(crossing([(10, 0.2), (20, 0.8)]))
            self.assertIsNone(crossing([(10, 0.8), (20, 0.6)]))
            self.assertIsNone(crossing([(10, 0.5)]))
            cells = [{'state_size': x, 'n': 10, 'success_rate': p} for x, p in [(10, 1), (20, 0)]]
            result = bootstrap_crossing(cells, repetitions=20)
            self.assertEqual(result['state_size_star'], 15)
            self.assertEqual(result['bracketed_fraction'], 1)

        def test_analysis_denominators_and_missing_offsets(self):
            config = load_config(PAPER_CONFIGS['main_grid'])
            config.update(environments=['stateful_puzzle'], models=['gpt-4o-mini'], archetypes=1, variants=4)
            config['sweeps'][0].update(state_size=[5], state_dependency=[1])
            plan = build_plan(config)
            plan['source_digest'] = 'fixture'
            traces = [([0.1, 0.1, 0.1], [True, True, False], False), ([1, 1, 1], [True, True, True], False), ([0.1, 0.1, 0.1], [False, True, True], True), ([0.8, 0.7, 0.6], [False, True, True], False)]
            completed = []
            for job, (values, valid, success) in zip(plan['jobs'], traces):
                completed.append({'job': job, 'episode': {'final_success': success}, 'steps': [{'world_state_accuracy': v, 'action_valid': a} for v, a in zip(values, valid)]})
            report = summarize(plan, completed, 2, repetitions=20)
            row = report['failure_order'][0]
            self.assertEqual(row['collapsed_n'], 3)
            self.assertEqual(row['paired_n'], 1)
            self.assertEqual(row['world_first_pct'], 100)
            self.assertEqual(row['median_lead'], 1)
            fidelity = {r['relative_step']: r for r in report['fidelity_before_action']}
            self.assertIsNone(fidelity[-3]['mean_fidelity'])
            self.assertEqual(fidelity[-1]['n'], 1)
            self.assertEqual(fidelity[0]['n'], 2)
            self.assertAlmostEqual(fidelity[0]['mean_fidelity'], 0.45)

        def test_incomplete_grid_does_not_bridge_gaps(self):
            config = load_config(PAPER_CONFIGS['main_grid'])
            config.update(environments=['stateful_puzzle'], models=['gpt-4o-mini'], archetypes=1, variants=1)
            config['sweeps'][0].update(state_size=[5, 10, 20], state_dependency=[1])
            plan = build_plan(config)
            plan['source_digest'] = 'fixture'
            completed = [{'job': j, 'episode': {'final_success': True}, 'steps': [{'world_state_accuracy': 1, 'action_valid': True}]} for j in plan['jobs'] if j['stress']['state_size'] != 10]
            report = summarize(plan, completed, 2, repetitions=20)
            self.assertEqual(report['incomplete_adjacent_pairs'], 2)
            self.assertEqual(report['cliff_tests'], [])
            self.assertEqual(report['critical_points'], [])

    class RunnerTests(unittest.TestCase):

        def test_api_error_never_becomes_a_behavioral_failure(self):
            with patch.dict('os.environ', {}, clear=True):
                client = LLMClient(fixed_temperature=0.0, strict_api_errors=True)
            with patch.object(client, 'call_raw', return_value=RawCallResult(text='', api_error='unavailable')) as call:
                with self.assertRaises(RuntimeError):
                    client.call_typed('gpt-4o-mini', 'planner', 'system', 'user', 42)
                self.assertEqual(call.call_count, 4)
                self.assertTrue(all((c.kwargs['temperature'] == 0 for c in call.call_args_list)))

        def test_same_time_logging_and_nonblocking_self_diag(self):
            env = ENV_REGISTRY['stateful_puzzle']()
            task = {'stress_config': {'state_card': 5, 'dep_density': 1, 'T': 1, 'branching': 4, 'obs_noise': 'clean', 'mut_rate': 'static'}}
            agent = BaseAgent('fixture', 'fixture', 'C_struct', planner=lambda **kw: CallOutcome({'next_action': 'noop'}), updater=lambda **kw: CallOutcome({'full_world_state': kw['observation_partial_state']}), self_diag=lambda **kw: CallOutcome({'self_check_valid': False, 'should_replan': True}))
            ctx = EpisodeContext('r', 't', 42, 42, 'fixture', task['stress_config'])
            sw, ew = (BufferWriter(), BufferWriter())
            run_episode(env, agent, task, ctx, sw, ew)
            self.assertEqual(len(sw.rows), 1)
            row = sw.rows[0]
            self.assertEqual(row['env_name'], 'stateful_puzzle')
            self.assertTrue(row['gold_world_state_before'])
            self.assertEqual(row['world_state_accuracy'], 1)
            self.assertTrue(row['action_valid'])
            self.assertFalse(row['self_check_valid'])

        def test_atomic_execution_resume_and_log_validation(self):
            config = load_config(PAPER_CONFIGS['main_grid'])
            config.update(environments=['stateful_puzzle'], models=['gpt-4o-mini'], archetypes=1, variants=1)
            config['sweeps'][0].update(state_size=[5], state_dependency=[1], horizon=[1])
            plan = build_plan(config)
            agent = BaseAgent('fixture', 'gpt-4o-mini', 'C_struct', planner=lambda **kw: CallOutcome({'next_action': 'noop'}), updater=lambda **kw: CallOutcome({'full_world_state': kw['observation_partial_state']}), self_diag=lambda **kw: CallOutcome({'self_check_valid': True}))
            with tempfile.TemporaryDirectory() as tmp, patch('src.agents.llm_client.LLMClient'), patch('src.agents.llm_agent.build_llm_agent', return_value=agent) as builder:
                execute(plan, Path(tmp))
                execute(plan, Path(tmp))
                self.assertEqual(builder.call_count, 1)
                self.assertEqual(len(load_completed(Path(tmp), plan)), 1)
                path = next((Path(tmp) / 'episodes').glob('*.json'))
                record = json.loads(path.read_text())
                record['steps'][0]['world_state_accuracy'] = 0.123
                atomic_json(path, record)
                with self.assertRaises(ValueError):
                    load_completed(Path(tmp), plan)

        def test_incomplete_episode_is_not_saved(self):
            config = load_config(PAPER_CONFIGS['main_grid'])
            config.update(environments=['stateful_puzzle'], models=['gpt-4o-mini'], archetypes=1, variants=1)
            config['sweeps'][0].update(state_size=[5], state_dependency=[1], horizon=[1])
            plan = build_plan(config)
            with tempfile.TemporaryDirectory() as tmp, patch('src.agents.llm_client.LLMClient'), patch('src.agents.llm_agent.build_llm_agent', side_effect=RuntimeError('fixture')):
                with self.assertRaises(RuntimeError):
                    execute(plan, Path(tmp))
                self.assertEqual(list((Path(tmp) / 'episodes').glob('*.json')), [])
                self.assertEqual(len((Path(tmp) / 'errors.jsonl').read_text().splitlines()), 1)
    import os
    class ExtendedMetricsTests(unittest.TestCase):
        def test_latest_completed_run_is_not_concatenated(self):
            import gzip
            from src.evaluation.jsonl_logger import load_completed_traces
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                base = {'task_id': 'sample', 'model': 'fixture', 'decoding_seed': 1, 'steps_taken': 2, 'stress_config': {'T': 40, 'state_card': 10, 'dep_density': 1}, 'final_success': False}
                metadata = [{**base, 'run_id': 'old'}, {**base, 'run_id': 'new'}]
                (root / 'episode.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in metadata))
                rows = [{**meta, 'step': i, 'world_state_accuracy': (.2 if meta['run_id'] == 'old' else 1.), 'action_valid': True, 'output_tokens_this_step': 10} for meta in metadata for i in range(2)]
                with gzip.open(root / 'step.jsonl.gz', 'wt') as handle:
                    handle.write(''.join(json.dumps(x) + '\n' for x in rows))
                completed, excluded = load_completed_traces(root / 'step.jsonl', root / 'episode.jsonl')
                self.assertEqual(len(completed), 1)
                self.assertEqual(excluded, [])
                self.assertEqual([x['world_state_accuracy'] for x in completed[0][1]], [1., 1.])
                self.assertEqual([x['step'] for x in completed[0][1]], [0, 1])

        def test_empty_transport_is_not_a_failed_rollout(self):
            from src.evaluation.jsonl_logger import load_completed_traces
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                meta = {'task_id': 'sample', 'run_id': 'run', 'model': 'fixture', 'steps_taken': 1, 'stress_config': {'T': 40, 'state_card': 10, 'dep_density': 1}, 'final_success': False}
                (root / 'episode.jsonl').write_text(json.dumps(meta) + '\n')
                (root / 'step.jsonl').write_text(json.dumps({**meta, 'step': 0, 'output_tokens_this_step': 0}) + '\n')
                completed, excluded = load_completed_traces(root / 'step.jsonl', root / 'episode.jsonl')
                self.assertFalse(completed)
                self.assertEqual(excluded[0]['reason'], 'empty_model_response')

        def test_censored_onsets_keep_their_own_denominators(self):
            from experiments.sol_cliff_probe.aligned_cell_metrics import summarize_cells
            meta = {'stress_config': {'state_card': 10, 'dep_density': 1}, 'final_success': False}
            no_event = [{'world_state_accuracy': 1., 'action_valid': True} for _ in range(3)]
            event = [{'world_state_accuracy': v, 'action_valid': i != 2} for i, v in enumerate([1., .3, .2])]
            cell = summarize_cells([(meta, no_event), (meta, event)])['sc=10,dd=1']
            self.assertEqual(cell['n'], 2)
            self.assertEqual(cell['tau_w_n'], 1)
            self.assertEqual(cell['tau_a_n'], 1)
            self.assertEqual(cell['paired_n'], 1)
            self.assertEqual(cell['lead_values'], [1])
            self.assertIsNone(cell['retention_at_5'])

        def test_all_excluded_cell_is_missing_not_zero_success(self):
            from experiments.sol_cliff_probe.aligned_cell_metrics import summarize_cells
            cell = summarize_cells([], [{'stress_config': {'state_card': 320, 'dep_density': 4}}])['sc=320,dd=4']
            self.assertEqual(cell['n'], 0)
            self.assertIsNone(cell['success_rate'])
            self.assertIsNone(cell['tau_w_median'])

        def test_empty_raw_response_aborts_strict_client(self):
            from src.agents.llm_client import ConfiguredLLMClient, RawCallResult
            with patch.dict(os.environ, {}, clear=True):
                client = ConfiguredLLMClient()
            with patch.object(client, 'call_raw', return_value=RawCallResult(text='')):
                with self.assertRaises(RuntimeError):
                    client.call_typed(model='gpt-5.6-sol', call_type='planner', system_prompt='', user_prompt='', seed=1, max_retries=0)

        def test_sonnet_actual_model_route_is_explicit(self):
            from src.agents.llm_client import ConfiguredLLMClient, RawCallResult
            model_id = 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'
            with patch.dict(os.environ, {'MODEL_ID_MAP': json.dumps({'claude-sonnet-4-5': 'bedrock:' + model_id})}, clear=True):
                client = ConfiguredLLMClient()
            with patch.object(client, '_bedrock_anthropic_raw', return_value=RawCallResult(text='ok', raw_meta={'model': model_id})) as backend:
                result = client.call_raw('claude-sonnet-4-5', '', '', None, 0., 100)
            self.assertEqual(backend.call_args.args[0], model_id)
            self.assertEqual(result.raw_meta['model'], model_id)

        def test_extended_grid_plans_use_supported_levels(self):
            expected = {'stateful_puzzle': 500, 'graph_nav': 500, 'tool_dag': 400}
            for env, count in expected.items():
                config = load_config(PAPER_CONFIGS['extended_' + env])
                plan = build_plan(config)
                self.assertEqual(plan['episode_count'], count)
                self.assertEqual(config['sweeps'][0]['state_dependency'], [1, 2, 4, 6, 8])
                instance = ENV_REGISTRY[env]()
                instance.reset({'stress_config': {'state_card': max(config['sweeps'][0]['state_size']), 'dep_density': 8, 'T': 40}}, 42)
                self.assertTrue(instance.get_gold_state())

    suite = unittest.TestSuite((unittest.defaultTestLoader.loadTestsFromTestCase(case) for case in (ProtocolTests, StatisticsTests, RunnerTests, ExtendedMetricsTests)))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1

def plot_updated_critical(data, output_dir):
    from pathlib import Path
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from statistics import NormalDist
    fine = data['legacy']['critical_scan_sc_fine_episode.jsonl']
    coarse = [c for c in data['legacy']['stage4_episode.jsonl'] if c['ss'] in (10, 20) and c['sd'] == 1]
    points = sorted(fine + coarse, key=lambda c: c['ss'])
    x = np.array([c['ss'] for c in points], dtype=float)
    n = np.array([c['n'] for c in points], dtype=float)
    p = np.array([c['successes'] / c['n'] for c in points], dtype=float)
    z = NormalDist().inv_cdf(.995)
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)) / denominator
    def curve(x, x0, k, low, high):
        return low + (high-low)/(1+np.exp(k*(x-x0)))
    fit, _ = curve_fit(curve, x, p, p0=[13.5,1.5,0,1], bounds=([10,.1,-.01,.9],[16,10,.01,1.01]), maxfev=5000)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':9,'axes.labelsize':10,'axes.labelweight':'bold','pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(4.6,2.7))
    xx=np.linspace(9.5,20.5,400)
    ax.plot(xx,curve(xx,*fit),color='#355f8d',lw=1.8)
    ax.errorbar(x,p,yerr=[p-(center-half),center+half-p],fmt='o',mfc='#355f8d',mec='#17232e',mew=.7,ms=4.5,ecolor='#444444',elinewidth=.8,capsize=3,zorder=3)
    ax.axhline(.5,color='#777777',ls=':',lw=.9)
    ax.axvline(fit[0],color='#ff2929',ls='--',lw=1.4)
    ax.text(fit[0]+.15,.89,r'$ss^\star$',color='#ff2929',fontsize=11)
    ax.set(xlim=(9.5,20.5),ylim=(-.04,1.06),xticks=range(10,21),yticks=[0,.25,.5,.75,1])
    ax.set_xlabel(r'State size $ss$');ax.set_ylabel(r'Success rate $\hat p$')
    ax.set_title('Critical Point Localisation',fontweight='bold',pad=9)
    ax.grid(axis='y',ls=':',lw=.5,alpha=.4)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    fig.tight_layout()
    for ext in ['pdf','png']:
        fig.savefig(Path(output_dir)/('fig4_critical_points.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)

def plot_exported_grids(data_path, output_dir):
    import json
    from pathlib import Path
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    data = json.loads(Path(data_path).read_text())
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 9, 'axes.labelsize': 10, 'axes.labelweight': 'bold', 'axes.titlesize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    models = [('claude_haiku_4_5','claude-haiku-4-5'), ('gpt_5_6_luna','gpt-5.6-luna'), ('claude_sonnet_4_5','claude-sonnet-4-5'), ('gpt_5_6_sol','gpt-5.6-sol')]

    def draw(ax, matrix, ss, sd, name, legacy=False):
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('#eeeeee')
        im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=1, origin='lower', aspect='auto', interpolation='nearest')
        for row in range(len(ss)):
            for col in range(len(sd)):
                value = matrix[row, col]
                ax.text(col, row, '--' if np.isnan(value) else f'{value:.0%}', ha='center', va='center', color='white' if np.isfinite(value) and value < .55 else 'black', fontweight='bold')
        ax.set_xticks(range(len(sd)), sd)
        ax.set_yticks(range(len(ss)), ss)
        ax.set_xlabel('State Dependency (sd)')
        ax.set_ylabel('State Size (ss)')
        ax.set_title(name, fontweight='bold', pad=8)
        if legacy:
            ax.axhline(1.5, color='#ff7f0e', ls='--', lw=1.8)
            ax.add_patch(Rectangle((2.5,.5),1,1,fill=False,edgecolor='#e7298a',lw=2))
        else:
            if np.isfinite(matrix).any() and np.nanmin(matrix) < .5 < np.nanmax(matrix):
                ax.contour(np.ma.masked_invalid(matrix), levels=[.5], colors=['#ff7f0e'], linewidths=1.8, linestyles='--')
                masked=np.where(np.isfinite(matrix), np.abs(matrix-.5), np.inf)
                ri,ci=np.unravel_index(np.argmin(masked), matrix.shape)
                ax.add_patch(Rectangle((ci-.5,ri-.5),1,1,fill=False,edgecolor='#e7298a',lw=2))
        return im

    def save(fig, name):
        for extension in ('pdf','png'):
            fig.savefig(out / (name+'.'+extension), dpi=300, bbox_inches='tight', metadata={'Creator':'Matplotlib'} if extension=='pdf' else None)
        plt.close(fig)

    if 'legacy' in data:
        ss,sd=[5,10,20,40],[1,2,4,6]
        mats=[]
        for filename in ['stage4_episode.jsonl','stage5_b_episode.jsonl']:
            rows=[x for x in data['legacy'][filename] if x['environment']=='stateful_puzzle']
            lookup={(x['ss'],x['sd']):x['successes']/x['n'] for x in rows}
            mats.append(np.array([[lookup[(x,y)] for y in sd] for x in ss]))
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.3))
        fig.subplots_adjust(left=.09,right=.84,bottom=.23,top=.77,wspace=.28)
        for ax,matrix,name in zip(axes,mats,['claude-haiku-4-5','gpt-4o-mini']):im=draw(ax,matrix,ss,sd,name,True)
        axes[1].set_ylabel('')
        cb=fig.colorbar(im,cax=fig.add_axes([.87,.23,.025,.54]));cb.ax.yaxis.set_major_formatter(PercentFormatter(1));cb.set_ticks([0,.25,.5,.75,1]);cb.set_label('Success Rate',fontweight='bold')
        fig.suptitle('StatefulPuzzle: Cross-Model Capability Cliffs',fontsize=12,fontweight='bold',y=.96)
        fig.legend(handles=[Line2D([0],[0],color='#ff7f0e',ls='--',lw=1.8,label='Capability cliff (ss=10\u219220)'),Rectangle((0,0),1,1,fill=False,edgecolor='#e7298a',lw=2,label='Critical cell (ss=10, sd=6)')],loc='lower center',bbox_to_anchor=(.5,.005),ncol=2,fontsize=8,frameon=True)
        save(fig,'fig5_cross_model_comparison')
        plot_updated_critical(data, out)
    for env,title in [('stateful_puzzle','StatefulPuzzle'),('graph_nav','GraphNav'),('tool_dag','ToolDAG')]:
        fig,axes=plt.subplots(2,2,figsize=(7.2,6.0))
        fig.subplots_adjust(left=.09,right=.84,bottom=.14,top=.88,hspace=.55,wspace=.28)
        for ax,(mid,name) in zip(axes.flat,models):
            cells=data['grids'][mid+'_'+env]['cells']
            ss=sorted({int(k.split(',')[0][3:]) for k in cells});sd=[1,2,4,6,8]
            matrix=np.array([[np.nan if cells[f'sc={x},dd={y}']['success_rate'] is None else cells[f'sc={x},dd={y}']['success_rate'] for y in sd] for x in ss])
            im=draw(ax,matrix,ss,sd,name)
        cb=fig.colorbar(im,cax=fig.add_axes([.88,.23,.025,.55]));cb.ax.yaxis.set_major_formatter(PercentFormatter(1));cb.set_ticks([0,.25,.5,.75,1]);cb.set_label('Success Rate',fontweight='bold')
        fig.suptitle(title+': Cross-Model Capability Cliffs',fontsize=12,fontweight='bold',y=.97)
        fig.legend(handles=[Line2D([0],[0],color='#ff7f0e',ls='--',lw=1.8,label='Capability cliff (50% crossing)'),Rectangle((0,0),1,1,fill=False,edgecolor='#e7298a',lw=2,label='Critical cell (closest to 50%)')],loc='lower center',bbox_to_anchor=(.48,.005),ncol=2,fontsize=8,frameon=True)
        save(fig,'fig_extended_'+env)
    return 0

def main():
    parser = argparse.ArgumentParser(description='Acceptance analysis, failure timing, critical-point localization, and figures')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', type=Path)
    mode.add_argument('--report', type=Path)
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--exported-grids', type=Path)
    mode.add_argument('--legacy', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--window', type=int)
    parser.add_argument('--boundary-cells', type=Path)
    parser.add_argument('--bootstrap-repetitions', type=int, default=1000)
    args = parser.parse_args()
    if args.exported_grids:
        if args.output is None:
            parser.error('--output is required with --exported-grids')
        return plot_exported_grids(args.exported_grids, args.output)
    if args.self_test:
        return run_self_tests()
    if (args.run or args.report) and args.output is None:
        parser.error('--output is required with --run or --report')
    if args.run:
        if args.window is None:
            parser.error('--window is required with --run')
        return analyze_run(args)
    if args.report:
        return plot_report(args.report, args.output)
    return legacy_main()
if __name__ == '__main__':
    sys.exit(main())
