#!/usr/bin/env python3
"""Stage 4 G1-trigger acceptance analysis.

Reads experiments/stage4_g1_trigger/stage4_results.json and computes:
  - 4x4 success_rate grid (sc x dd)
  - Wilson 95% / 99% CI per cell
  - Adjacent-cell Δp̂ along sc axis (within each dd) and dd axis (within each sc)
  - Max-drop adjacent pair (cell-pair with largest signed |Δp̂|)
  - Barnard exact p-value for the max-drop pair (unconditional, scipy)
  - Newcombe (Method 10 / score) 99% CI for the max-drop Δp̂
  - G1 acceptance verdict per EXP_PLAN §6.1:
      * monotone decrease along at least one axis with max |Δp̂| >= 20pp
      * Wilson lower-CI(low-stress) > Wilson upper-CI(high-stress) of corner pair
      * (additional) Barnard p < 0.001 on the max-drop pair

Outputs:
  - analysis/stage4_g1_acceptance.json (full numerical results)
  - analysis/stage4_g1_acceptance_table.md (human-readable summary)
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "experiments" / "stage4_g1_trigger" / "stage4_results.json"
OUT_JSON = ROOT / "analysis" / "stage4_g1_acceptance.json"
OUT_MD = ROOT / "analysis" / "stage4_g1_acceptance_table.md"

SC_LEVELS = [5, 10, 20, 40]
DD_LEVELS = [1, 2, 4, 6]


def jst_now() -> str:
    return datetime.now(tz=timezone(timedelta(hours=9))).isoformat()


# ---- Wilson CI ---------------------------------------------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


# ---- Newcombe Method 10 (score CI for diff of independent proportions) -------

def newcombe_diff_ci(k1: int, n1: int, k2: int, n2: int,
                     alpha: float = 0.05) -> tuple[float, float]:
    """Newcombe Hybrid Score (Method 10) CI for p1 - p2."""
    l1, u1 = wilson_ci(k1, n1, alpha)
    l2, u2 = wilson_ci(k2, n2, alpha)
    p1, p2 = k1 / n1, k2 / n2
    diff = p1 - p2
    lo = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return lo, hi


# ---- Barnard exact (unconditional) -------------------------------------------

def barnard_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Barnard exact unconditional two-sided p-value via scipy.

    Returns p-value for H0: p1 == p2 against H1: p1 != p2.
    """
    from scipy.stats import barnard_exact
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    res = barnard_exact(table, alternative="two-sided")
    return float(res.pvalue)


# ---- Main analysis -----------------------------------------------------------

def main() -> int:
    data = json.loads(RESULTS_PATH.read_text())
    per_cell = data["per_cell"]

    # Build grid
    grid_n = {}
    grid_k = {}
    for sc in SC_LEVELS:
        for dd in DD_LEVELS:
            key = f"sc={sc},dd={dd}"
            c = per_cell[key]
            grid_n[(sc, dd)] = c["n"]
            grid_k[(sc, dd)] = c["n_success"]

    # Wilson 95/99 per cell
    wilson95 = {}
    wilson99 = {}
    for sc, dd in grid_n:
        n, k = grid_n[(sc, dd)], grid_k[(sc, dd)]
        wilson95[(sc, dd)] = wilson_ci(k, n, alpha=0.05)
        wilson99[(sc, dd)] = wilson_ci(k, n, alpha=0.01)

    # Adjacent Δp̂ along sc axis (within each dd)
    sc_adj = []
    for dd in DD_LEVELS:
        for i in range(len(SC_LEVELS) - 1):
            a, b = SC_LEVELS[i], SC_LEVELS[i + 1]
            ka, na = grid_k[(a, dd)], grid_n[(a, dd)]
            kb, nb = grid_k[(b, dd)], grid_n[(b, dd)]
            sc_adj.append({
                "axis": "sc",
                "dd": dd,
                "lower": a, "upper": b,
                "p_lower": ka / na, "p_upper": kb / nb,
                "delta_pp": round((ka / na - kb / nb) * 100, 2),
                "k_lower": ka, "n_lower": na,
                "k_upper": kb, "n_upper": nb,
            })

    # Adjacent Δp̂ along dd axis (within each sc)
    dd_adj = []
    for sc in SC_LEVELS:
        for i in range(len(DD_LEVELS) - 1):
            a, b = DD_LEVELS[i], DD_LEVELS[i + 1]
            ka, na = grid_k[(sc, a)], grid_n[(sc, a)]
            kb, nb = grid_k[(sc, b)], grid_n[(sc, b)]
            dd_adj.append({
                "axis": "dd",
                "sc": sc,
                "lower": a, "upper": b,
                "p_lower": ka / na, "p_upper": kb / nb,
                "delta_pp": round((ka / na - kb / nb) * 100, 2),
                "k_lower": ka, "n_lower": na,
                "k_upper": kb, "n_upper": nb,
            })

    all_adj = sc_adj + dd_adj
    # Max-drop adjacent pair (signed, positive = success FALLS)
    max_pair = max(all_adj, key=lambda x: x["delta_pp"])
    # Barnard + Newcombe for max-drop pair
    barn_p = barnard_p(
        max_pair["k_lower"], max_pair["n_lower"],
        max_pair["k_upper"], max_pair["n_upper"],
    )
    new99 = newcombe_diff_ci(
        max_pair["k_lower"], max_pair["n_lower"],
        max_pair["k_upper"], max_pair["n_upper"],
        alpha=0.01,
    )
    new95 = newcombe_diff_ci(
        max_pair["k_lower"], max_pair["n_lower"],
        max_pair["k_upper"], max_pair["n_upper"],
        alpha=0.05,
    )

    # Also: corner-to-corner max drop (e.g. sc=5,dd=1 vs sc=40,dd=6 — extreme)
    # Already captured by adjacent pairs? No — corner pair is non-adjacent.
    # Define "extreme corner pair" = max delta across ANY two cells in the grid.
    all_cells = [(sc, dd) for sc in SC_LEVELS for dd in DD_LEVELS]
    extreme = None
    for c1 in all_cells:
        for c2 in all_cells:
            if c1 == c2: continue
            ka, na = grid_k[c1], grid_n[c1]
            kb, nb = grid_k[c2], grid_n[c2]
            delta = ka / na - kb / nb
            if extreme is None or delta > extreme["delta"]:
                extreme = {
                    "delta": delta, "p_lower": ka / na, "p_upper": kb / nb,
                    "c_lower": {"sc": c1[0], "dd": c1[1]},
                    "c_upper": {"sc": c2[0], "dd": c2[1]},
                    "k_lower": ka, "n_lower": na,
                    "k_upper": kb, "n_upper": nb,
                }
    ext_barn = barnard_p(extreme["k_lower"], extreme["n_lower"],
                         extreme["k_upper"], extreme["n_upper"])
    ext_new99 = newcombe_diff_ci(extreme["k_lower"], extreme["n_lower"],
                                 extreme["k_upper"], extreme["n_upper"], alpha=0.01)

    # G1 acceptance verdict (EXP_PLAN §6.1):
    # (a) monotone decrease along at least one axis with max |Δp̂| >= 20pp
    # (b) Wilson lower-CI(low-stress corner pair) > Wilson upper-CI(high-stress)
    # We test corner pair (sc=5,dd=1) high vs (sc=40,dd=6) low.
    corner_low = (5, 1)
    corner_high = (40, 6)
    wlow_low, wlow_hi = wilson95[corner_low]
    whigh_low, whigh_hi = wilson95[corner_high]
    corner_ci_separated = wlow_low > whigh_hi
    monotone_sc_within_dd1 = all(
        per_cell[f"sc={SC_LEVELS[i]},dd=1"]["success_rate"]
        >= per_cell[f"sc={SC_LEVELS[i+1]},dd=1"]["success_rate"]
        for i in range(len(SC_LEVELS) - 1)
    )
    monotone_dd_within_sc10 = all(
        per_cell[f"sc=10,dd={DD_LEVELS[i]}"]["success_rate"]
        >= per_cell[f"sc=10,dd={DD_LEVELS[i+1]}"]["success_rate"]
        for i in range(len(DD_LEVELS) - 1)
    )
    g1_pass = (
        (monotone_sc_within_dd1 or monotone_dd_within_sc10)
        and max_pair["delta_pp"] >= 20.0
        and corner_ci_separated
        and barn_p < 0.001
    )

    out = {
        "generated_jst": jst_now(),
        "source": str(RESULTS_PATH.relative_to(ROOT)),
        "grid_axes": {"state_cards": SC_LEVELS, "dep_densities": DD_LEVELS},
        "grid_4x4_success_rate": {
            f"sc={sc},dd={dd}": grid_k[(sc, dd)] / grid_n[(sc, dd)]
            for sc in SC_LEVELS for dd in DD_LEVELS
        },
        "wilson_95_per_cell": {
            f"sc={sc},dd={dd}": {"lo": round(wilson95[(sc, dd)][0], 4),
                                  "hi": round(wilson95[(sc, dd)][1], 4)}
            for sc in SC_LEVELS for dd in DD_LEVELS
        },
        "wilson_99_per_cell": {
            f"sc={sc},dd={dd}": {"lo": round(wilson99[(sc, dd)][0], 4),
                                  "hi": round(wilson99[(sc, dd)][1], 4)}
            for sc in SC_LEVELS for dd in DD_LEVELS
        },
        "adjacent_sc_axis": sc_adj,
        "adjacent_dd_axis": dd_adj,
        "max_drop_adjacent_pair": {
            **max_pair,
            "barnard_exact_p_two_sided": barn_p,
            "newcombe_99_ci_pp": [round(new99[0] * 100, 2),
                                  round(new99[1] * 100, 2)],
            "newcombe_95_ci_pp": [round(new95[0] * 100, 2),
                                  round(new95[1] * 100, 2)],
        },
        "extreme_corner_pair_any": {
            **{k: v for k, v in extreme.items() if k != "delta"},
            "delta_pp": round(extreme["delta"] * 100, 2),
            "barnard_exact_p_two_sided": ext_barn,
            "newcombe_99_ci_pp": [round(ext_new99[0] * 100, 2),
                                  round(ext_new99[1] * 100, 2)],
        },
        "g1_acceptance": {
            "monotone_sc_within_dd1": monotone_sc_within_dd1,
            "monotone_dd_within_sc10": monotone_dd_within_sc10,
            "max_adjacent_delta_pp": max_pair["delta_pp"],
            "corner_pair": {
                "low_stress": {"sc": 5, "dd": 1, "wilson95": [wlow_low, wlow_hi]},
                "high_stress": {"sc": 40, "dd": 6, "wilson95": [whigh_low, whigh_hi]},
                "ci_separated": corner_ci_separated,
            },
            "barnard_p_max_pair": barn_p,
            "verdict_g1_pass": g1_pass,
        },
    }
    # Coerce numpy types → Python natives for JSON
    def _clean(o):
        if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list): return [_clean(x) for x in o]
        if isinstance(o, tuple): return [_clean(x) for x in o]
        if hasattr(o, "item"): return o.item()  # numpy scalar
        return o
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(_clean(out), sort_keys=True, indent=2, ensure_ascii=False))

    # Markdown summary
    md = []
    md.append(f"# Stage 4 G1-Trigger Acceptance Analysis\n")
    md.append(f"Generated: {jst_now()}\n")
    md.append(f"Source: `{RESULTS_PATH.relative_to(ROOT)}`\n\n")
    md.append("## 4×4 success_rate grid (rows=state_card, cols=dep_density)\n\n")
    md.append("| sc\\dd | 1 | 2 | 4 | 6 |\n|------:|---:|---:|---:|---:|")
    for sc in SC_LEVELS:
        row = [f"| **{sc}** "]
        for dd in DD_LEVELS:
            n, k = grid_n[(sc, dd)], grid_k[(sc, dd)]
            row.append(f" | {k}/{n} ({100*k/n:.0f}%) ")
        md.append("".join(row) + " |")
    md.append("\n")
    md.append("## Wilson 95% CI per cell (lo, hi)\n\n")
    md.append("| sc\\dd | 1 | 2 | 4 | 6 |\n|------:|:---:|:---:|:---:|:---:|")
    for sc in SC_LEVELS:
        row = [f"| **{sc}** "]
        for dd in DD_LEVELS:
            lo, hi = wilson95[(sc, dd)]
            row.append(f" | [{lo:.3f}, {hi:.3f}] ")
        md.append("".join(row) + " |")
    md.append("\n")
    md.append("## Max-drop adjacent pair\n\n")
    mp = max_pair
    md.append(f"- Axis: **{mp['axis']}**, {'dd='+str(mp['dd']) if mp['axis']=='sc' else 'sc='+str(mp['sc'])}, "
              f"{mp['lower']} → {mp['upper']}\n")
    md.append(f"- success: {mp['p_lower']:.0%} → {mp['p_upper']:.0%}, Δp̂ = **{mp['delta_pp']:+.1f}pp**\n")
    md.append(f"- Barnard exact (two-sided) **p = {barn_p:.3e}**\n")
    md.append(f"- Newcombe 99% CI: [{round(new99[0]*100,2)}pp, {round(new99[1]*100,2)}pp]\n")
    md.append(f"- Newcombe 95% CI: [{round(new95[0]*100,2)}pp, {round(new95[1]*100,2)}pp]\n\n")
    md.append("## Extreme corner pair (any two cells)\n\n")
    md.append(f"- {extreme['c_lower']} → {extreme['c_upper']}\n")
    md.append(f"- success: {extreme['p_lower']:.0%} → {extreme['p_upper']:.0%}, Δ = **{round(extreme['delta']*100,2):+.1f}pp**\n")
    md.append(f"- Barnard exact (two-sided) **p = {ext_barn:.3e}**\n")
    md.append(f"- Newcombe 99% CI: [{round(ext_new99[0]*100,2)}pp, {round(ext_new99[1]*100,2)}pp]\n\n")
    md.append("## G1 Acceptance Verdict\n\n")
    md.append(f"- monotone(sc | dd=1): **{monotone_sc_within_dd1}**\n")
    md.append(f"- monotone(dd | sc=10): **{monotone_dd_within_sc10}**\n")
    md.append(f"- max adjacent Δp̂: **{mp['delta_pp']:.1f}pp** (≥20pp threshold = {mp['delta_pp']>=20})\n")
    md.append(f"- Corner pair (sc=5,dd=1) Wilson95 vs (sc=40,dd=6): "
              f"[{wlow_low:.3f},{wlow_hi:.3f}] vs [{whigh_low:.3f},{whigh_hi:.3f}] → separated = **{corner_ci_separated}**\n")
    md.append(f"- Barnard p on max pair: **{barn_p:.3e}** (<0.001 = {barn_p<0.001})\n")
    md.append(f"\n### **G1 VERDICT: {'PASS ✅' if g1_pass else 'FAIL ❌'}**\n")
    OUT_MD.write_text("\n".join(md))

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print()
    print(f"G1 verdict: {'PASS ✅' if g1_pass else 'FAIL ❌'}")
    print(f"Max-drop adjacent pair: {mp['axis']} dd={mp.get('dd','-')}sc={mp.get('sc','-')} "
          f"{mp['lower']}→{mp['upper']}: {mp['p_lower']:.0%}→{mp['p_upper']:.0%} ({mp['delta_pp']:+.1f}pp)")
    print(f"  Barnard p = {barn_p:.3e}")
    print(f"  Newcombe 99% CI: [{round(new99[0]*100,2)}, {round(new99[1]*100,2)}]pp")
    print(f"Extreme corner: {extreme['c_lower']}→{extreme['c_upper']}: {round(extreme['delta']*100,2):+.1f}pp")
    print(f"  Barnard p = {ext_barn:.3e}")
    print(f"  Newcombe 99% CI: [{round(ext_new99[0]*100,2)}, {round(ext_new99[1]*100,2)}]pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
