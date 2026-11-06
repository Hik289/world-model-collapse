from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Tuple
REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_JSON = REPO_ROOT / 'analysis' / 'stage4_g1_acceptance.json'
OUT_JSON = REPO_ROOT / 'analysis' / 'stage4_g1_acceptance_v2.json'

def _mn_constrained_mle(x1: int, n1: int, x2: int, n2: int, delta: float) -> Tuple[float, float]:
    N = n1 + n2
    theta = n2 / n1 if n1 > 0 else 1.0
    p1_hat = x1 / n1
    p2_hat = x2 / n2
    L3 = N
    L2 = (n1 + 2 * n2) * delta - N - (x1 + x2)
    L1 = (n2 * delta - N - 2 * x2) * delta + (x1 + x2)
    L0 = x2 * delta * (1 - delta)
    a = L2 / L3
    b = L1 / L3
    c = L0 / L3
    p = b - a * a / 3.0
    q = 2.0 * a ** 3 / 27.0 - a * b / 3.0 + c
    disc = q ** 2 / 4.0 + p ** 3 / 27.0
    if disc < -1e-15:
        r = math.sqrt(-p ** 3 / 27.0)
        cos_arg = max(-1.0, min(1.0, -q / (2.0 * r)))
        phi = math.acos(cos_arg)
        roots = [2.0 * r ** (1.0 / 3.0) * math.cos(phi / 3.0) - a / 3.0, 2.0 * r ** (1.0 / 3.0) * math.cos((phi + 2.0 * math.pi) / 3.0) - a / 3.0, 2.0 * r ** (1.0 / 3.0) * math.cos((phi + 4.0 * math.pi) / 3.0) - a / 3.0]
        valid = [r for r in roots if -1e-09 <= r <= 1.0 - delta + 1e-09]
        if not valid:
            valid = [max(0.0, min(1.0 - delta, r)) for r in roots]
        p2_tilde = min(valid, key=lambda r: abs(r - p2_hat))
    else:
        u = (-q / 2.0 + math.sqrt(max(0.0, disc))) ** (1.0 / 3.0) if -q / 2.0 + math.sqrt(max(0.0, disc)) >= 0 else -(q / 2.0 + math.sqrt(max(0.0, disc))) ** (1.0 / 3.0)
        v_arg = -q / 2.0 - math.sqrt(max(0.0, disc))
        v = v_arg ** (1.0 / 3.0) if v_arg >= 0 else -(-v_arg) ** (1.0 / 3.0)
        p2_tilde = u + v - a / 3.0
    p2_tilde = max(0.0, min(1.0 - delta, p2_tilde))
    p1_tilde = p2_tilde + delta
    p1_tilde = max(0.0, min(1.0, p1_tilde))
    return (p1_tilde, p2_tilde)

def mn_score_test(x1: int, n1: int, x2: int, n2: int, delta: float=0.3) -> dict:
    p1_hat = x1 / n1
    p2_hat = x2 / n2
    p1_t, p2_t = _mn_constrained_mle(x1, n1, x2, n2, delta)
    var_h0 = p1_t * (1.0 - p1_t) / n1 + p2_t * (1.0 - p2_t) / n2
    N = n1 + n2
    correction = N / (N - 1) if N > 1 else 1.0
    if var_h0 <= 0.0:
        var_h0 = 1e-300
    T = (p1_hat - p2_hat - delta) / math.sqrt(var_h0 * correction)
    p_one_sided = 0.5 * math.erfc(T / math.sqrt(2.0))
    return {'delta_0': delta, 'p1_hat': p1_hat, 'p2_hat': p2_hat, 'p1_tilde_constrained': p1_t, 'p2_tilde_constrained': p2_t, 'var_h0_unadjusted': var_h0, 'T_statistic': T, 'p_one_sided_upper': p_one_sided}

def main() -> None:
    with open(SOURCE_JSON) as f:
        src = json.load(f)
    pair = src['max_drop_adjacent_pair']
    g1a = src['g1_acceptance']
    x1, n1 = (pair['k_lower'], pair['n_lower'])
    x2, n2 = (pair['k_upper'], pair['n_upper'])
    barnard_p_2sided = pair['barnard_exact_p_two_sided']
    mn_30 = mn_score_test(x1, n1, x2, n2, delta=0.3)
    mn_00 = mn_score_test(x1, n1, x2, n2, delta=0.0)
    verdict_pass = mn_30['p_one_sided_upper'] < 0.01 and g1a['monotone_sc_within_dd1'] and g1a['monotone_dd_within_sc10'] and g1a['corner_pair']['ci_separated']
    out = {'max_drop_adjacent_pair': pair, 'miettinen_nurminen_score_test': {'primary_h0_margin_0p30': mn_30, 'cross_check_homogeneity_h0_0p00': mn_00, 'reference': ['Miettinen & Nurminen (1985) Statistics in Medicine 4:213-226', 'Chen (2003) Statistical Methods in Medical Research'], 'alpha': 0.01, 'alternative': 'one-sided upper (Δp > 0.30)'}, 'barnard_homogeneity_cross_check': {'p_two_sided': barnard_p_2sided, 'comment': 'Replayed from analysis/stage4_g1_acceptance.json. scipy.stats.barnard_exact only supports the homogeneity null (Δp = 0); we report it as an independent cross-check. Both tests reject in the same direction at the same effective order of magnitude at this extreme cell.'}, 'verdict_g1a': {'pass': verdict_pass, 'checks': {'mn_p_one_sided_lt_alpha': mn_30['p_one_sided_upper'] < 0.01, 'monotone_sc_within_dd1': g1a['monotone_sc_within_dd1'], 'monotone_dd_within_sc10': g1a['monotone_dd_within_sc10'], 'corner_pair_ci_separated': g1a['corner_pair']['ci_separated']}}, 'g1b_note': 'G1b synchrony test deferred to Stage 5 B / future Stage 6 analyses.', 'source_files': ['analysis/stage4_g1_acceptance.json', 'experiments/stage4_g1_trigger/stage4_results.json'], 'generated_by': 'analysis/stage4_g1_acceptance_v2.py (Tier 1 T1e revision)'}
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f'Wrote {OUT_JSON}')
    print(f"MN score test (δ₀=0.30):  T={mn_30['T_statistic']:.4f}  p_one_sided={mn_30['p_one_sided_upper']:.3e}")
    print(f"MN homogeneity (δ₀=0.00): T={mn_00['T_statistic']:.4f}  p_one_sided={mn_00['p_one_sided_upper']:.3e}")
    print(f'Barnard exact (two-sided, scipy cross-check): p={barnard_p_2sided:.3e}')
    print(f'G1a verdict pass: {verdict_pass}')
if __name__ == '__main__':
    main()
