#!/usr/bin/env python3
"""T2d — Multiple Testing Correction (Bonferroni + BH FDR).

Five primary p-values:
  P1  G1a (Stage 4 trigger)        — Miettinen–Nurminen score test, H0: Δp ≤ 0.30
                                     value pulled from analysis/stage4_g1_acceptance_v2.json
  P2  G2 ablation T                — omnibus Fisher-Freeman-Halton (2×K success table)
  P3  G2 ablation branching        — same
  P4  G2 ablation obs_noise        — same
  P5  G2 ablation mut_rate         — same

We apply both Bonferroni (family α=0.01) and Benjamini–Hochberg FDR (q=0.05) and
report which claims survive each correction.

The four ablation tests are *omnibus* "any difference among the four levels"
homogeneity tests on a 2×4 (success, fail) table per axis; the directional
shape of each axis (monotone / non-monotone / null) is a *descriptive*
secondary claim, not a Goal-trigger p-value, so we use a single omnibus per
axis to avoid arbitrary "primary direction" choices.

Method: Fisher–Freeman–Halton exact homogeneity test via Monte-Carlo
permutation (B = 50,000) on the 2×K table; reference test stat = G²
(likelihood-ratio). This is robust to small expected counts in the way the
asymptotic χ² is not.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
from collections import defaultdict
from _common import ROOT, ANALYSIS, DATA, jst_now, write_json


RNG = np.random.default_rng(20260607)
N_PERM = 50000
FAMILY_ALPHA = 0.01
FDR_Q = 0.05


def collect_ablation_table(axis):
    """Return (level_names, succ, fail) — counts per level."""
    cells = defaultdict(lambda: [0, 0])
    with open(DATA / f"stage5b_ablation_{axis}_episode.jsonl") as f:
        for line in f:
            d = json.loads(line)
            sc = d["stress_config"]
            lvl = sc[axis] if axis != "branching" else sc["branching"]
            if d["final_success"]:
                cells[lvl][0] += 1
            else:
                cells[lvl][1] += 1
    # sort by level — numeric for T/branching, alphabetic for the categorical axes
    if axis in {"T", "branching"}:
        levels = sorted(cells.keys(), key=lambda x: int(x))
    else:
        levels = sorted(cells.keys())
    succ = np.array([cells[l][0] for l in levels])
    fail = np.array([cells[l][1] for l in levels])
    return [str(l) for l in levels], succ, fail


def g2_stat(succ, fail):
    """Likelihood-ratio G² for 2×K homogeneity (null: equal success rate)."""
    n_per_col = succ + fail
    n_total = succ.sum() + fail.sum()
    if n_total == 0:
        return 0.0
    p_pool = succ.sum() / n_total
    expect_succ = n_per_col * p_pool
    expect_fail = n_per_col * (1 - p_pool)
    g2 = 0.0
    for o, e in zip(succ, expect_succ):
        if o > 0 and e > 0:
            g2 += 2 * o * np.log(o / e)
    for o, e in zip(fail, expect_fail):
        if o > 0 and e > 0:
            g2 += 2 * o * np.log(o / e)
    return g2


def fisher_freeman_halton_p(succ, fail, n_perm=N_PERM, rng=RNG):
    """Monte-Carlo exact p-value for 2×K homogeneity, G² reference stat.

    Algorithm: under H0 of homogeneity, conditional on row + column totals,
    the cell counts follow a generalized hypergeometric. We sample by
    permuting outcome labels (success/fail) across the pooled episodes; each
    column keeps its size. G² is monotone in deviation from homogeneity, so
    the MC p = fraction of permutations with G² >= G²_obs.
    """
    obs_g2 = g2_stat(succ, fail)
    n_per_col = succ + fail
    n_total = n_per_col.sum()
    n_succ = int(succ.sum())
    if n_total == 0 or n_succ == 0 or n_succ == n_total:
        return 1.0, obs_g2  # nothing to test
    cum = np.concatenate([[0], np.cumsum(n_per_col)])
    n_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(n_total)
        # First n_succ permuted positions get a "success" label
        idx_succ = set(perm[:n_succ].tolist())
        ps = np.zeros_like(succ)
        pf = np.zeros_like(fail)
        for k in range(len(n_per_col)):
            col_idx = set(range(int(cum[k]), int(cum[k + 1])))
            ps[k] = len(col_idx & idx_succ)
            pf[k] = n_per_col[k] - ps[k]
        if g2_stat(ps, pf) >= obs_g2 - 1e-12:
            n_ge += 1
    p = (n_ge + 1) / (n_perm + 1)  # add-one rule (Agresti & Coull)
    return float(p), float(obs_g2)


def bonferroni(ps, alpha):
    m = len(ps)
    return [p * m for p in ps], [p * m <= alpha for p in ps]


def bh_fdr(ps, q):
    """Benjamini–Hochberg adjusted p-values + decision."""
    m = len(ps)
    idx = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    running_min = 1.0
    for rank, i in enumerate(reversed(idx)):
        bh = ps[i] * m / (m - rank)
        running_min = min(running_min, bh)
        adj[i] = min(1.0, running_min)
    decisions = [adj[i] <= q for i in range(m)]
    return adj, decisions


def main():
    # --- 1. Pull G1a primary p-value
    g1a = json.loads((ANALYSIS / "stage4_g1_acceptance_v2.json").read_text())
    mn = g1a["miettinen_nurminen_score_test"]["primary_h0_margin_0p30"]
    p_g1a = float(mn["p_one_sided_upper"])
    g1a_test_meta = {
        "test": "Miettinen–Nurminen score test, H0: Δp ≤ 0.30, one-sided",
        "T_stat": float(mn["T_statistic"]),
        "delta_0": float(mn["delta_0"]),
        "source": "analysis/stage4_g1_acceptance_v2.json",
    }

    # --- 2. Compute ablation omnibus p-values
    ablations = {}
    p_abl = {}
    for axis in ["T", "branching", "obs_noise", "mut_rate"]:
        levels, succ, fail = collect_ablation_table(axis)
        p, g = fisher_freeman_halton_p(succ, fail)
        ablations[axis] = {
            "levels": levels,
            "n_success_per_level": [int(s) for s in succ],
            "n_fail_per_level":    [int(s) for s in fail],
            "n_per_level":         [int(s + t) for s, t in zip(succ, fail)],
            "success_rate":        [float(s / (s + t)) for s, t in zip(succ, fail)],
            "G2_statistic": g,
            "p_value_monte_carlo": p,
            "n_permutations": N_PERM,
            "test": "Fisher–Freeman–Halton exact homogeneity (G² reference), MC permutation",
        }
        p_abl[axis] = p

    # --- 3. Family of 5 p-values
    names = ["G1a", "G2_T", "G2_branching", "G2_obs_noise", "G2_mut_rate"]
    ps = [p_g1a, p_abl["T"], p_abl["branching"], p_abl["obs_noise"], p_abl["mut_rate"]]

    # Bonferroni
    bonf_adj, bonf_pass = bonferroni(ps, FAMILY_ALPHA)
    # BH FDR
    bh_adj, bh_pass = bh_fdr(ps, FDR_Q)

    table = []
    for nm, p_raw, p_b, p_bh, b_ok, bh_ok in zip(names, ps, bonf_adj, bh_adj, bonf_pass, bh_pass):
        table.append({
            "claim": nm,
            "p_raw": p_raw,
            "p_bonferroni_adj": min(1.0, p_b),
            "bonferroni_survives_alpha_0p01": bool(b_ok),
            "p_bh_adj": p_bh,
            "bh_survives_q_0p05": bool(bh_ok),
        })

    out = {
        "generated_jst": jst_now(),
        "analysis": "T2d_multiple_testing_correction",
        "family_size": 5,
        "family_alpha_bonferroni": FAMILY_ALPHA,
        "fdr_q": FDR_Q,
        "g1a_meta": g1a_test_meta,
        "ablation_tests": ablations,
        "raw_p_values": dict(zip(names, ps)),
        "summary": table,
        "claims_surviving_bonferroni_at_alpha_0p01": [
            t["claim"] for t in table if t["bonferroni_survives_alpha_0p01"]
        ],
        "claims_surviving_bh_at_q_0p05": [
            t["claim"] for t in table if t["bh_survives_q_0p05"]
        ],
    }
    write_json(ANALYSIS / "multiple_testing_correction.json", out)

    md = []
    md.append("# T2d — Multiple Testing Correction (Bonferroni + BH FDR)\n")
    md.append(f"Generated: {jst_now()}\n\n")
    md.append("## Family of 5 primary p-values\n\n")
    md.append("| Claim | Test | Raw p | Bonferroni-adj | Survives α=0.01? | BH-adj | Survives q=0.05? |\n"
              "|---|---|---:|---:|:---:|---:|:---:|")
    test_desc = {
        "G1a": "MN score, Δp≤0.30 (one-sided)",
        "G2_T": "Fisher-Freeman-Halton",
        "G2_branching": "Fisher-Freeman-Halton",
        "G2_obs_noise": "Fisher-Freeman-Halton",
        "G2_mut_rate": "Fisher-Freeman-Halton",
    }
    for row in table:
        md.append(
            f"| {row['claim']} | {test_desc[row['claim']]} | "
            f"{row['p_raw']:.3e} | "
            f"{row['p_bonferroni_adj']:.3e} | "
            f"{'✅' if row['bonferroni_survives_alpha_0p01'] else '❌'} | "
            f"{row['p_bh_adj']:.3e} | "
            f"{'✅' if row['bh_survives_q_0p05'] else '❌'} |"
        )
    md.append("\n## Ablation contingency tables (2×K success / fail)\n")
    for axis, info in ablations.items():
        md.append(f"\n### {axis}\n")
        md.append("| level | n | success | rate |\n|---|---:|---:|---:|")
        for lvl, n, s, r in zip(info["levels"], info["n_per_level"],
                                 info["n_success_per_level"], info["success_rate"]):
            md.append(f"| {lvl} | {n} | {s} | {r:.3f} |")
        md.append(f"\nG² = {info['G2_statistic']:.3f}, MC p ({info['n_permutations']:,} perms) = {info['p_value_monte_carlo']:.3e}\n")

    md.append("\n## Verdict\n")
    md.append(f"- Surviving Bonferroni (α=0.01): {out['claims_surviving_bonferroni_at_alpha_0p01']}\n")
    md.append(f"- Surviving BH FDR (q=0.05):      {out['claims_surviving_bh_at_q_0p05']}\n\n")
    md.append("## Notes\n")
    md.append("- G1a's headline raw p (~1e-52) is dominated by the comparison; multiple-testing "
              "correction does not threaten it.\n")
    md.append("- Ablation p-values are *omnibus* (any difference among the four levels). "
              "Directional shape claims (T monotone-increasing, partial-only cliff, mut_rate "
              "hump) are secondary descriptive findings.\n")
    md.append("- Branching ablation is the pre-registered *null* (no detectable effect); a "
              "non-significant p is the expected outcome, consistent with the paper §4D claim.\n")
    (ANALYSIS / "multiple_testing_correction.md").write_text("\n".join(md))

    print(f"Wrote multiple_testing_correction.json + .md")
    for row in table:
        print(f"  {row['claim']:18s} p_raw={row['p_raw']:.3e} bonf={row['p_bonferroni_adj']:.3e}  "
              f"bonf{'✅' if row['bonferroni_survives_alpha_0p01'] else '❌'}  "
              f"bh={row['p_bh_adj']:.3e} bh{'✅' if row['bh_survives_q_0p05'] else '❌'}")


if __name__ == "__main__":
    main()
