#!/usr/bin/env python3
"""T2c — Lee-Drift Baseline Comparison.

For each Stage-4 episode, fit two univariate models to WSA(t):
  Model L (Lee linear drift):       WSA(t) = α + β · t              (k=2 params, σ² estimated)
  Model P (phase step):             WSA(t) = α + β · 1[t ≥ τ]       (k=3 params: α, β, τ ∈ {1..T-1})

Compare via Bayesian Information Criterion (BIC). For Gaussian residuals,
  BIC = n·ln(RSS/n) + k·ln(n)
A lower BIC indicates a better model. We report ΔBIC = BIC_Lee − BIC_Phase
(positive ⇒ Phase preferred).

Per-cell aggregate: % of episodes preferring Phase (ΔBIC > 0); also a
"decisive" subset using |ΔBIC| > 6 (Kass & Raftery 1995 strong evidence).

Episodes with constant WSA (all identical) are excluded (no residual signal to
discriminate the two models).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from collections import defaultdict
from _common import (
    ROOT, ANALYSIS, jst_now, parse_stage4_task_id,
    load_step_episodes, write_json, cell_key,
)

MIN_T = 6  # need >= 6 steps to fit the phase model with at least 2 obs each side


def fit_lee(y):
    """OLS linear fit y = α + β t, return RSS."""
    n = len(y)
    t = np.arange(n, dtype=float)
    A = np.column_stack([np.ones(n), t])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    rss = float((resid ** 2).sum())
    return rss, coef


def fit_phase(y):
    """Step model y = α + β·1[t≥τ], MLE over τ ∈ {2..n-2} (need ≥2 obs in
    each segment). Returns (RSS_min, τ_best, α_best, β_best).
    """
    n = len(y)
    best = None
    for tau in range(2, n - 1):  # tau is the first step in the post segment
        pre = y[:tau]
        post = y[tau:]
        if len(pre) < 2 or len(post) < 2:
            continue
        a = float(pre.mean())
        b = float(post.mean() - pre.mean())
        rss = float(((pre - a) ** 2).sum() + ((post - (a + b)) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, tau, a, b)
    return best  # tuple or None


def bic_gauss(rss, n, k):
    """Gaussian BIC up to constants. Lower = better."""
    if rss <= 0:
        # perfect fit: degenerate.
        # Use a small positive RSS proxy so log is finite.
        rss = 1e-12
    return n * np.log(rss / n) + k * np.log(n)


def analyse_episode(steps):
    y = np.array([s["world_state_accuracy"] for s in steps], dtype=float)
    n = len(y)
    if n < MIN_T:
        return None
    if y.std() == 0:
        return None
    rss_L, coef_L = fit_lee(y)
    phase = fit_phase(y)
    if phase is None:
        return None
    rss_P, tau, a, b = phase

    bic_L = bic_gauss(rss_L, n, k=3)   # 2 mean params + 1 σ²
    bic_P = bic_gauss(rss_P, n, k=4)   # 2 mean params + 1 τ + 1 σ²
    delta = bic_L - bic_P  # >0 means phase preferred
    return dict(
        n_steps=n,
        bic_lee=bic_L,
        bic_phase=bic_P,
        delta_bic=float(delta),
        tau_best=int(tau),
        beta_lee=float(coef_L[1]),
        beta_phase=float(b),
        prefer_phase=bool(delta > 0),
        decisive_phase=bool(delta > 6),
        decisive_lee=bool(delta < -6),
    )


def main():
    steps_by_task = load_step_episodes()
    per_episode = []
    skipped_short = 0
    skipped_const = 0
    for tid, steps in steps_by_task.items():
        sc, dd, _ = parse_stage4_task_id(tid)
        res = analyse_episode(steps)
        if res is None:
            # categorize skip reason
            if len(steps) < MIN_T:
                skipped_short += 1
            else:
                skipped_const += 1
            continue
        per_episode.append({"task_id": tid, "sc": sc, "dd": dd, **res})

    # Aggregate per cell
    per_cell = defaultdict(list)
    for ep in per_episode:
        per_cell[cell_key(ep["sc"], ep["dd"])].append(ep)

    per_cell_stats = {}
    for ck, eps in sorted(per_cell.items()):
        n = len(eps)
        deltas = np.array([e["delta_bic"] for e in eps])
        per_cell_stats[ck] = {
            "n_episodes": n,
            "frac_prefer_phase": float(np.mean([e["prefer_phase"] for e in eps])) if n > 0 else None,
            "frac_decisive_phase": float(np.mean([e["decisive_phase"] for e in eps])) if n > 0 else None,
            "frac_decisive_lee":   float(np.mean([e["decisive_lee"]   for e in eps])) if n > 0 else None,
            "median_delta_bic": float(np.median(deltas)) if n > 0 else None,
            "mean_delta_bic":   float(np.mean(deltas))   if n > 0 else None,
            "median_tau_best": float(np.median([e["tau_best"] for e in eps])) if n > 0 else None,
        }

    n_total = sum(s["n_episodes"] for s in per_cell_stats.values())
    n_prefer_phase = sum(int(round(s["n_episodes"] * (s["frac_prefer_phase"] or 0)))
                         for s in per_cell_stats.values())
    n_decisive_phase = sum(int(round(s["n_episodes"] * (s["frac_decisive_phase"] or 0)))
                           for s in per_cell_stats.values())
    n_decisive_lee = sum(int(round(s["n_episodes"] * (s["frac_decisive_lee"] or 0)))
                         for s in per_cell_stats.values())

    overall_frac_phase = n_prefer_phase / max(n_total, 1)
    all_deltas = np.array([e["delta_bic"] for e in per_episode])
    overall_median_dbic = float(np.median(all_deltas)) if n_total > 0 else None
    overall_mean_dbic = float(np.mean(all_deltas)) if n_total > 0 else None

    out = {
        "generated_jst": jst_now(),
        "analysis": "T2c_Lee_drift_baseline",
        "source_file": "data/raw_logs/stage4_step.jsonl",
        "models": {
            "lee_linear":  "WSA(t) = α + β t   (k_eff = 3)",
            "phase_step":  "WSA(t) = α + β·1[t≥τ], τ ∈ {2,…,n−2}   (k_eff = 4)",
            "BIC": "n·ln(RSS/n) + k·ln(n); lower is better",
            "delta_bic_sign": "ΔBIC = BIC_lee − BIC_phase > 0 ⇒ Phase preferred",
            "decisive_threshold": "|ΔBIC| > 6  (Kass & Raftery 1995 strong evidence)",
        },
        "n_episodes_analysed": n_total,
        "n_episodes_skipped_short": skipped_short,
        "n_episodes_skipped_constant_WSA": skipped_const,
        "overall": {
            "frac_prefer_phase": overall_frac_phase,
            "frac_decisive_phase": n_decisive_phase / max(n_total, 1),
            "frac_decisive_lee":   n_decisive_lee   / max(n_total, 1),
            "median_delta_bic": overall_median_dbic,
            "mean_delta_bic": overall_mean_dbic,
        },
        "per_cell": per_cell_stats,
    }

    out_path = ANALYSIS / "stage4_lee_drift_baseline.json"
    write_json(out_path, out)

    md = []
    md.append("# T2c — Lee Linear Drift vs Phase Step Baseline (Stage 4)\n")
    md.append(f"Generated: {jst_now()}\n\n")
    md.append("## Models\n")
    md.append("- **Lee linear drift**: `WSA(t) = α + β·t` — slow, continuous decay over the episode.\n")
    md.append("- **Phase step**:      `WSA(t) = α + β·1[t≥τ]` — discrete transition at a fitted breakpoint τ.\n")
    md.append("- Compared via Gaussian BIC; ΔBIC = BIC_Lee − BIC_Phase > 0 means Phase preferred.\n\n")
    md.append("## Sample sizes\n")
    md.append(f"- Episodes analysed: **{n_total}** / 1600\n")
    md.append(f"- Skipped (T<{MIN_T} steps): {skipped_short}\n")
    md.append(f"- Skipped (constant WSA across episode): {skipped_const}\n\n")
    md.append("## Headline\n")
    md.append(f"- % preferring Phase model: **{overall_frac_phase:.1%}**\n")
    md.append(f"- % decisively Phase (ΔBIC>6): **{out['overall']['frac_decisive_phase']:.1%}**\n")
    md.append(f"- % decisively Lee   (ΔBIC<−6): **{out['overall']['frac_decisive_lee']:.1%}**\n")
    md.append(f"- Median ΔBIC: **{overall_median_dbic:+.2f}** (positive ⇒ Phase favoured)\n")
    md.append(f"- Mean ΔBIC:   **{overall_mean_dbic:+.2f}**\n\n")
    md.append("## Per-cell breakdown\n\n")
    md.append("| cell | n | % phase | % decisive phase | % decisive Lee | median ΔBIC | median τ |\n|---|---:|---:|---:|---:|---:|---:|")
    for ck in sorted(per_cell_stats.keys()):
        s = per_cell_stats[ck]
        if s["n_episodes"] == 0:
            md.append(f"| {ck} | 0 | — | — | — | — | — |")
        else:
            md.append(
                f"| {ck} | {s['n_episodes']} | "
                f"{s['frac_prefer_phase']:.0%} | "
                f"{s['frac_decisive_phase']:.0%} | "
                f"{s['frac_decisive_lee']:.0%} | "
                f"{s['median_delta_bic']:+.2f} | "
                f"{s['median_tau_best']:.1f} |"
            )
    md.append("\n## Interpretation\n")
    if overall_frac_phase >= 0.5:
        md.append("A *majority* of Stage-4 episodes' WSA trajectory is better described by a "
                  "discrete phase transition than by a slow linear drift à la Lee 2026. The phase "
                  "model wins by BIC despite carrying one extra parameter (τ), indicating the "
                  "improvement in fit is large enough to overcome the penalty.\n")
    else:
        md.append("A majority of episodes are better described by the linear Lee-drift baseline.\n")
    out_md = ANALYSIS / "stage4_lee_drift_baseline.md"
    out_md.write_text("\n".join(md))

    print(f"Wrote {out_path.name}, {out_md.name}")
    print(f"  n_analysed={n_total}, frac_prefer_phase={overall_frac_phase:.3f}, median ΔBIC={overall_median_dbic:+.2f}")


if __name__ == "__main__":
    main()
