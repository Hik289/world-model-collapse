#!/usr/bin/env python3
"""T2e — Cluster-Robust Bootstrap CI.

Stage 4 trigger grid uses 100 episodes per cell allocated as 10 archetype × 10
instance (per EXP_PLAN §6.5). task_id encodes the (archetype, instance) pair:
the trailing `t000..t099` index decomposes as
  archetype = task_index // 10
  instance  = task_index %  10
The same convention is applied to Stage 5b ablations (n=25/level allocated as
5 archetype × 5 instance: archetype = idx // 5, instance = idx % 5).

For each cell we report:
  - Wilson 95% CI on success rate (i.i.d. assumption, as in the paper).
  - Cluster-bootstrap 95% percentile CI resampling *archetypes* (with all
    instances of the resampled archetype kept intact), B = 10,000.
  - Deflation factor = cluster CI width / Wilson CI width.

A deflation factor > 1 indicates the i.i.d. Wilson CI under-states uncertainty
(positive intra-archetype correlation); < 1 indicates the Wilson CI over-states
(negative correlation, rare in practice).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
from collections import defaultdict
from _common import (
    ROOT, ANALYSIS, DATA, jst_now, write_json, wilson_ci, cluster_bootstrap_ci,
    parse_stage4_task_id, cell_key,
)

B = 10000
RNG = np.random.default_rng(20260607)


def cluster_ci_on_rate(succ_arr, cluster_arr, B=B, rng_seed=20260607):
    """Cluster-percentile bootstrap CI on mean(succ_arr)."""
    lo, hi, _ = cluster_bootstrap_ci(cluster_arr, succ_arr,
                                     stat_fn=lambda a: float(a.mean()),
                                     B=B, ci_pct=0.95, rng_seed=rng_seed)
    return lo, hi


def analyse_stage4():
    """Return per-cell stats."""
    by_cell = defaultdict(list)  # ck -> list of (t_idx, success)
    with open(DATA / "stage4_episode.jsonl") as f:
        for line in f:
            d = json.loads(line)
            sc, dd, t = parse_stage4_task_id(d["task_id"])
            by_cell[cell_key(sc, dd)].append((t, 1 if d["final_success"] else 0))

    per_cell = {}
    for ck, eps in sorted(by_cell.items()):
        idx = np.array([e[0] for e in eps])
        succ = np.array([e[1] for e in eps], dtype=float)
        n = len(succ)
        k = int(succ.sum())
        # Wilson 95%
        w_lo, w_hi = wilson_ci(k, n, ci_pct=0.95)
        wilson_width = w_hi - w_lo
        # Cluster bootstrap (archetype = idx // 10)
        clusters = idx // 10
        n_clusters = len(np.unique(clusters))
        c_lo, c_hi = cluster_ci_on_rate(succ, clusters)
        cluster_width = c_hi - c_lo
        defl = (cluster_width / wilson_width) if wilson_width > 0 else float("nan")
        per_cell[ck] = {
            "n": n,
            "n_success": k,
            "n_clusters": n_clusters,
            "p_hat": float(k / n) if n > 0 else None,
            "wilson_95_ci": [w_lo, w_hi],
            "wilson_95_width": wilson_width,
            "cluster_bootstrap_95_ci": [c_lo, c_hi],
            "cluster_bootstrap_95_width": cluster_width,
            "deflation_factor": defl,
        }
    return per_cell


def analyse_ablation(axis, archetype_size=5):
    """Stage 5b ablation: parse `s5b_abl_{axis}_{level}_t{idx}`."""
    by_lvl = defaultdict(list)
    with open(DATA / f"stage5b_ablation_{axis}_episode.jsonl") as f:
        for line in f:
            d = json.loads(line)
            tid = d["task_id"]
            # parse trailing t000..t024
            t = int(tid.split("_t")[-1])
            lvl = (d["stress_config"][axis]
                   if axis != "branching" else d["stress_config"]["branching"])
            by_lvl[lvl].append((t, 1 if d["final_success"] else 0))

    per_level = {}
    for lvl in sorted(by_lvl.keys(), key=lambda x: (isinstance(x, str), x)):
        eps = by_lvl[lvl]
        idx = np.array([e[0] for e in eps])
        succ = np.array([e[1] for e in eps], dtype=float)
        n = len(succ); k = int(succ.sum())
        w_lo, w_hi = wilson_ci(k, n, ci_pct=0.95)
        wilson_width = w_hi - w_lo
        clusters = idx // archetype_size  # 5 instances per archetype
        n_clusters = len(np.unique(clusters))
        c_lo, c_hi = cluster_ci_on_rate(succ, clusters)
        cluster_width = c_hi - c_lo
        defl = (cluster_width / wilson_width) if wilson_width > 0 else float("nan")
        per_level[str(lvl)] = {
            "n": n,
            "n_success": k,
            "n_clusters": n_clusters,
            "p_hat": float(k / n) if n > 0 else None,
            "wilson_95_ci": [w_lo, w_hi],
            "wilson_95_width": wilson_width,
            "cluster_bootstrap_95_ci": [c_lo, c_hi],
            "cluster_bootstrap_95_width": cluster_width,
            "deflation_factor": defl,
        }
    return per_level


def main():
    stage4 = analyse_stage4()
    abl = {axis: analyse_ablation(axis) for axis in ["T", "branching", "obs_noise", "mut_rate"]}

    # Aggregate deflation summary — only "informative" cells with 0 < p̂ < 1.
    def _informative(s):
        return (s["p_hat"] is not None and 0.0 < s["p_hat"] < 1.0
                and np.isfinite(s["deflation_factor"])
                and s["wilson_95_width"] > 0)

    s4_def = np.array([s["deflation_factor"] for s in stage4.values() if _informative(s)])
    abl_defs = {}
    all_abl_defs = []
    for axis, info in abl.items():
        vals = [s["deflation_factor"] for s in info.values() if _informative(s)]
        abl_defs[axis] = vals
        all_abl_defs.extend(vals)

    out = {
        "generated_jst": jst_now(),
        "analysis": "T2e_cluster_robust_bootstrap_CI",
        "convention": {
            "stage4": "archetype = task_index // 10, instance = task_index % 10 (10×10 design per cell)",
            "ablations": "archetype = task_index // 5, instance = task_index % 5 (5×5 design per level)",
            "ci_type": "Cluster percentile bootstrap, resample archetypes with replacement",
            "B": B,
            "ci_pct": 0.95,
        },
        "stage4_per_cell": stage4,
        "ablations_per_level": abl,
        "summary": {
            "stage4_median_deflation": float(np.median(s4_def)) if len(s4_def) else None,
            "stage4_mean_deflation":   float(np.mean(s4_def))   if len(s4_def) else None,
            "stage4_p25_deflation":    float(np.percentile(s4_def, 25)) if len(s4_def) else None,
            "stage4_p75_deflation":    float(np.percentile(s4_def, 75)) if len(s4_def) else None,
            "ablation_median_deflation_by_axis": {
                axis: float(np.median(vs)) if vs else None for axis, vs in abl_defs.items()
            },
            "overall_ablation_median_deflation": float(np.median(all_abl_defs)) if all_abl_defs else None,
        },
        "notes": (
            "Cluster bootstrap on cells where p_hat = 0 or 1 cannot deflate or inflate "
            "the CI in the standard sense (the bootstrap concentrates at 0 or 1 with all "
            "weight). For such cells we still report the width as 0 and a non-finite "
            "deflation factor; they are excluded from the summary statistics. The "
            "informative cells are those with intermediate p_hat where the Wilson CI is "
            "wide enough for the cluster structure to matter."
        ),
    }
    write_json(ANALYSIS / "cluster_robust_ci.json", out)

    md = []
    md.append("# T2e — Cluster-Robust Bootstrap CI (Archetype-Level Resampling)\n")
    md.append(f"Generated: {jst_now()}\n\n")
    md.append("## Convention\n")
    md.append("- Stage 4: each cell has 100 episodes laid out as 10 archetype × 10 instance "
              "(per EXP_PLAN §6.5). `archetype = task_index // 10`.\n")
    md.append("- Stage 5b ablations: each level has 25 episodes laid out as 5 archetype × 5 "
              "instance (`archetype = task_index // 5`).\n")
    md.append(f"- B = {B:,} bootstrap reps, 95% percentile CI.\n")
    md.append("- Deflation factor = cluster_CI_width / Wilson_CI_width. >1 = Wilson under-states; <1 = Wilson over-states.\n\n")
    md.append("## Stage 4 per-cell\n\n")
    md.append("| cell | n | k | p̂ | Wilson 95% | Cluster 95% | Defl. factor |\n|---|---:|---:|---:|---:|---:|---:|")
    for ck, s in sorted(stage4.items()):
        w_lo, w_hi = s["wilson_95_ci"]
        c_lo, c_hi = s["cluster_bootstrap_95_ci"]
        defl = s["deflation_factor"]
        defl_s = f"{defl:.2f}" if np.isfinite(defl) else "n/a"
        md.append(
            f"| {ck} | {s['n']} | {s['n_success']} | {s['p_hat']:.2f} | "
            f"[{w_lo:.3f}, {w_hi:.3f}] | [{c_lo:.3f}, {c_hi:.3f}] | {defl_s} |"
        )
    md.append(f"\nStage 4 median deflation factor (informative cells only): **{out['summary']['stage4_median_deflation']}**")
    if out['summary']['stage4_median_deflation'] is not None:
        md.append(f"  (mean={out['summary']['stage4_mean_deflation']:.3f}; IQR=[{out['summary']['stage4_p25_deflation']:.3f}, {out['summary']['stage4_p75_deflation']:.3f}])\n")
    else:
        md.append("\n")

    for axis, info in abl.items():
        md.append(f"\n## Stage 5b ablation: {axis}\n\n")
        md.append("| level | n | k | p̂ | Wilson 95% | Cluster 95% | Defl. factor |\n|---|---:|---:|---:|---:|---:|---:|")
        for lvl, s in info.items():
            w_lo, w_hi = s["wilson_95_ci"]
            c_lo, c_hi = s["cluster_bootstrap_95_ci"]
            defl = s["deflation_factor"]
            defl_s = f"{defl:.2f}" if np.isfinite(defl) else "n/a"
            md.append(
                f"| {lvl} | {s['n']} | {s['n_success']} | {s['p_hat']:.2f} | "
                f"[{w_lo:.3f}, {w_hi:.3f}] | [{c_lo:.3f}, {c_hi:.3f}] | {defl_s} |"
            )
        if abl_defs[axis]:
            md.append(f"\nMedian deflation factor: **{np.median(abl_defs[axis]):.3f}**\n")

    md.append("\n## Interpretation\n")
    md.append("- Deflation factors close to 1.0 indicate that the i.i.d. Wilson CI used in "
              "the paper is well-calibrated under archetype-level clustering — episodes "
              "within an archetype carry essentially independent information beyond the "
              "shared structural template.\n")
    md.append("- Factors > 1 indicate residual intra-archetype correlation (Wilson CI too narrow).\n")
    md.append("- Cells with p̂ ∈ {0, 1} are degenerate (both Wilson and cluster CIs collapse "
              "to a half-open interval) and are not informative; they are excluded from the "
              "summary statistics.\n")
    (ANALYSIS / "cluster_robust_ci.md").write_text("\n".join(md))

    print("Wrote cluster_robust_ci.json + .md")
    print(f"  Stage 4 median deflation (informative cells): "
          f"{out['summary']['stage4_median_deflation']}")
    print(f"  Ablation median deflation by axis: {out['summary']['ablation_median_deflation_by_axis']}")


if __name__ == "__main__":
    main()
