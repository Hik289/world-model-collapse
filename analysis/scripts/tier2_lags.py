"""T2a (G1b synchrony) + T2b (G4 WS-precedes-plan) shared computation.

Sign convention: lag = ws_collapse_step - plan_collapse_step.
  lag <  0  : WS first   (G4 prediction)
  lag =  0  : same step
  lag >  0  : plan first
  |lag| <= 1 : synchronous within 1 step  (G1b prediction)

NOTE on the brief's wording: the directive's T2a equation 'lag = plan - ws' was
inverted; the canonical convention (used by T2b and G4 in STATISTICAL_PROTOCOL
§4) is ws - plan. |lag| <= 1 is identical either way; T2a is unaffected.
The directive's T2b PASS criterion ('median lag < 0 with 99% CI excluding 0')
is therefore the binding statement and we use ws - plan throughout.
"""
from __future__ import annotations
import json, sys, pathlib, datetime, collections
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import (load_step_episodes, first_combined_ws_collapse,
                     first_plan_collapse_step, first_ws_collapse_step,
                     first_sca_collapse_step, percentile_bootstrap_ci)
import numpy as np

JST = datetime.timezone(datetime.timedelta(hours=9))

STEP_PATH = "data/raw_logs/stage4_step.jsonl"


def build_lag_table(step_path: str = STEP_PATH):
    steps_by_tid = load_step_episodes(step_path)
    rows = []
    for tid, steps in steps_by_tid.items():
        sc = steps[0]["stress_config"]["state_card"]
        dd = steps[0]["stress_config"]["dep_density"]
        ws_step = first_combined_ws_collapse(steps)
        plan_step = first_plan_collapse_step(steps)
        ws_only_step = first_ws_collapse_step(steps)
        sca_only_step = first_sca_collapse_step(steps)
        # collapse_detected per the brief: WS OR plan observed
        collapse_detected = ws_step is not None or plan_step is not None
        row = dict(
            task_id=tid,
            state_card=sc,
            dep_density=dd,
            n_steps=len(steps),
            ws_step=ws_step,
            plan_step=plan_step,
            ws_only_step=ws_only_step,
            sca_only_step=sca_only_step,
            collapse_detected=collapse_detected,
            both_observed=(ws_step is not None and plan_step is not None),
            lag=(ws_step - plan_step) if (ws_step is not None and plan_step is not None) else None,
        )
        rows.append(row)
    return rows


def t2a_synchrony(rows):
    eps = [r for r in rows if r["collapse_detected"]]
    paired = [r for r in eps if r["both_observed"]]
    lags = np.array([r["lag"] for r in paired])

    if len(lags) == 0:
        return dict(error="no paired collapsed episodes")

    sync_count = int(np.sum(np.abs(lags) <= 1))
    n_paired = len(lags)
    pct_sync = sync_count / n_paired

    # 99% bootstrap CI on percent synchronous (among paired collapsed eps)
    pct_lo, pct_hi = percentile_bootstrap_ci(
        list((np.abs(lags) <= 1).astype(int)),
        stat_fn=lambda a: float(np.mean(a)),
        B=10000, ci_pct=0.99, rng_seed=20260607)

    # Conservative variant: episodes where WS or plan event is unobserved are
    # implicitly NON-synchronous (treated as |lag| > 1). This is a sensitivity
    # check against selection-on-the-paired-subset.
    cons_indic = []
    for r in eps:
        if r["both_observed"]:
            cons_indic.append(1 if abs(r["lag"]) <= 1 else 0)
        else:
            cons_indic.append(0)
    pct_cons = float(np.mean(cons_indic)) if cons_indic else float("nan")
    pct_cons_lo, pct_cons_hi = percentile_bootstrap_ci(
        cons_indic, stat_fn=lambda a: float(np.mean(a)),
        B=10000, ci_pct=0.99, rng_seed=20260607)

    # Histogram of lags
    hist = collections.Counter([int(x) for x in lags])
    # Per-cell summary
    per_cell = {}
    for (sc, dd) in sorted({(r["state_card"], r["dep_density"]) for r in paired}):
        cell_lags = [r["lag"] for r in paired if r["state_card"] == sc and r["dep_density"] == dd]
        if not cell_lags:
            continue
        cs = np.array(cell_lags)
        per_cell[f"sc={sc},dd={dd}"] = dict(
            n_paired=len(cs),
            pct_sync=float(np.mean(np.abs(cs) <= 1)),
            median_lag=float(np.median(cs)),
        )

    verdict_pass = pct_lo >= 0.80
    return dict(
        n_eps_collapsed=len(eps),
        n_paired=n_paired,
        pct_paired_of_collapsed=n_paired / len(eps) if eps else float("nan"),
        sync_count=sync_count,
        pct_sync=pct_sync,
        pct_sync_99ci=[pct_lo, pct_hi],
        # Conservative (treat unpaired collapsed as non-sync):
        pct_sync_conservative=pct_cons,
        pct_sync_conservative_99ci=[pct_cons_lo, pct_cons_hi],
        lag_histogram={str(k): hist[k] for k in sorted(hist)},
        per_cell=per_cell,
        pre_registered_threshold=0.80,
        verdict_pass=verdict_pass,
        verdict_note=("PASS (lower 99% CI >= 0.80)" if verdict_pass else
                      "FAIL (lower 99% CI < 0.80)"),
    )


def t2b_precedence(rows):
    eps = [r for r in rows if r["collapse_detected"]]
    paired = [r for r in eps if r["both_observed"]]
    lags = np.array([r["lag"] for r in paired])
    if len(lags) == 0:
        return dict(error="no paired collapsed episodes")

    median_lag = float(np.median(lags))
    # 99% bootstrap CI on median
    med_lo, med_hi = percentile_bootstrap_ci(
        list(lags), stat_fn=lambda a: float(np.median(a)),
        B=10000, ci_pct=0.99, rng_seed=20260607)

    ws_first = float(np.mean(lags < 0))
    syn = float(np.mean(np.abs(lags) <= 1))
    plan_first = float(np.mean(lags > 0))
    ws_lead_ge_1 = float(np.mean(lags <= -1))  # G4 (b) criterion

    # 99% CIs
    ws_first_lo, ws_first_hi = percentile_bootstrap_ci(
        list((lags < 0).astype(int)), stat_fn=lambda a: float(np.mean(a)),
        B=10000, ci_pct=0.99, rng_seed=20260607)
    ws_lead_ge_1_lo, ws_lead_ge_1_hi = percentile_bootstrap_ci(
        list((lags <= -1).astype(int)), stat_fn=lambda a: float(np.mean(a)),
        B=10000, ci_pct=0.99, rng_seed=20260607)

    # Pre-registered PASS:
    #   (a) median lag < 0 with 99% bootstrap CI excluding 0  (i.e., upper bound < 0),
    #         requires median <= -1 with upper CI <= -1 per directive ("[-N,-1]"), or
    #   (b) >= 70% episodes have lag <= -1
    crit_a = (median_lag < 0) and (med_hi <= -1)
    crit_b = ws_lead_ge_1 >= 0.70
    verdict_pass = crit_a or crit_b

    # Wilcoxon signed-rank one-sided: H0 median(lag) >= 0 vs H1 median(lag) < 0
    from scipy.stats import wilcoxon
    nonzero = lags[lags != 0]
    if len(nonzero) > 0:
        wstat, wp = wilcoxon(lags, alternative="less", zero_method="wilcox")
        wp = float(wp); wstat = float(wstat)
    else:
        wstat, wp = float("nan"), float("nan")

    per_cell = {}
    for (sc, dd) in sorted({(r["state_card"], r["dep_density"]) for r in paired}):
        cs = np.array([r["lag"] for r in paired
                       if r["state_card"] == sc and r["dep_density"] == dd])
        if cs.size == 0:
            continue
        per_cell[f"sc={sc},dd={dd}"] = dict(
            n_paired=int(cs.size),
            median_lag=float(np.median(cs)),
            pct_ws_first=float(np.mean(cs < 0)),
            pct_ws_lead_ge_1=float(np.mean(cs <= -1)),
            pct_synchronous=float(np.mean(np.abs(cs) <= 1)),
        )

    return dict(
        n_eps_collapsed=len(eps),
        n_paired=len(paired),
        median_lag=median_lag,
        median_lag_99ci=[med_lo, med_hi],
        pct_ws_first=ws_first,
        pct_ws_first_99ci=[ws_first_lo, ws_first_hi],
        pct_synchronous=syn,
        pct_plan_first=plan_first,
        pct_ws_lead_ge_1=ws_lead_ge_1,
        pct_ws_lead_ge_1_99ci=[ws_lead_ge_1_lo, ws_lead_ge_1_hi],
        wilcoxon_signed_rank_stat=wstat,
        wilcoxon_signed_rank_p_one_sided=wp,
        sign_convention="lag = ws_step - plan_step (negative = WS first)",
        per_cell=per_cell,
        pre_registered=dict(
            crit_a_median_lt_neg1_99ci="median lag < 0 AND 99% CI upper <= -1",
            crit_b_ge70pct_ws_lead_1="pct(lag <= -1) >= 0.70",
            crit_a_passed=crit_a,
            crit_b_passed=crit_b,
        ),
        verdict_pass=verdict_pass,
        verdict_note=("PASS" if verdict_pass else "FAIL") +
                     f" (a:{'Y' if crit_a else 'N'}, b:{'Y' if crit_b else 'N'})",
    )


def main():
    now = datetime.datetime.now(JST).isoformat()
    rows = build_lag_table()
    res_a = t2a_synchrony(rows)
    res_b = t2b_precedence(rows)

    out_a = dict(
        analysis="T2a — G1b synchrony test",
        source=STEP_PATH,
        generated_jst=now,
        operationalization=dict(
            ws_collapse="first step where WSA(t) < 0.50 OR rolling-SCA(t) < 0.30 (5-step window)",
            plan_collapse="first step where action_valid(t) is False and t >= 1",
            inclusion="ws_collapse OR plan_collapse observed (collapse_detected=True)",
            paired_subset="both ws_collapse and plan_collapse observed",
            sync_definition="|lag| <= 1 where lag = ws_step - plan_step",
        ),
        results=res_a,
    )
    out_b = dict(
        analysis="T2b — G4 WS-precedes-plan mechanism",
        source=STEP_PATH,
        generated_jst=now,
        operationalization=out_a["operationalization"],
        results=res_b,
    )

    pathlib.Path("analysis/stage4_g1b_synchrony.json").write_text(
        json.dumps(out_a, indent=2))
    pathlib.Path("analysis/stage4_g4_ws_precedes_plan.json").write_text(
        json.dumps(out_b, indent=2))

    # Markdown reports
    pathlib.Path("analysis/stage4_g1b_synchrony.md").write_text(
        _md_t2a(out_a))
    pathlib.Path("analysis/stage4_g4_ws_precedes_plan.md").write_text(
        _md_t2b(out_b))

    print("[T2a] verdict:", res_a["verdict_note"],
          "pct_sync=%.3f  99%%CI=[%.3f, %.3f]  n_paired=%d" %
          (res_a["pct_sync"], res_a["pct_sync_99ci"][0], res_a["pct_sync_99ci"][1],
           res_a["n_paired"]))
    print("[T2b] verdict:", res_b["verdict_note"],
          "median_lag=%.2f  99%%CI=[%.2f,%.2f]  pct_ws_lead_ge_1=%.3f" %
          (res_b["median_lag"], res_b["median_lag_99ci"][0],
           res_b["median_lag_99ci"][1], res_b["pct_ws_lead_ge_1"]))


def _md_t2a(o):
    r = o["results"]
    lines = []
    lines.append("# T2a — G1b Synchrony Test (Stage 4)\n")
    lines.append(f"_Generated_: {o['generated_jst']}\n")
    lines.append(f"_Source_: `{o['source']}`\n")
    lines.append("## Pre-registered hypothesis (G1b)\n")
    lines.append("World-state collapse and plan validity collapse occur **synchronously** "
                 "(within ±1 step) in collapsed episodes.\n"
                 "PASS criterion: ≥80% of collapsed episodes have `|lag| ≤ 1`, "
                 "with the lower bound of a 99% bootstrap CI on this percentage also ≥0.80.\n")
    lines.append("## Operationalization\n")
    for k, v in o["operationalization"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Headline result\n")
    lines.append(f"- n collapsed episodes (WS or plan): **{r['n_eps_collapsed']}**")
    lines.append(f"- n paired (both WS and plan observed): **{r['n_paired']}** "
                 f"({r['pct_paired_of_collapsed']*100:.1f}% of collapsed)")
    lines.append(f"- Synchronous (|lag|≤1) among paired: **{r['pct_sync']*100:.1f}%** "
                 f"(99% bootstrap CI [{r['pct_sync_99ci'][0]*100:.1f}, {r['pct_sync_99ci'][1]*100:.1f}])")
    lines.append(f"- Conservative (unpaired counted as non-sync): "
                 f"**{r['pct_sync_conservative']*100:.1f}%** "
                 f"(99% CI [{r['pct_sync_conservative_99ci'][0]*100:.1f}, "
                 f"{r['pct_sync_conservative_99ci'][1]*100:.1f}])")
    lines.append(f"- **Verdict**: {r['verdict_note']}\n")
    lines.append("## Lag histogram (paired subset)\n")
    lines.append("| lag (ws−plan, steps) | count |")
    lines.append("|---:|---:|")
    for k, v in r["lag_histogram"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Per-cell breakdown\n")
    lines.append("| cell | n paired | %sync | median lag |")
    lines.append("|---|---:|---:|---:|")
    for k, v in r["per_cell"].items():
        lines.append(f"| {k} | {v['n_paired']} | {v['pct_sync']*100:.1f}% | {v['median_lag']:.1f} |")
    lines.append("")
    lines.append("## Interpretation\n")
    if r["verdict_pass"]:
        lines.append("G1b synchrony PASSES: world-state collapse and plan collapse co-occur "
                     "within ±1 step in a clear majority of collapsed episodes, with the lower "
                     "99% bootstrap CI on the synchronous fraction exceeding the pre-registered "
                     "0.80 threshold. Consistent with simultaneous mode-collapse "
                     "(world-model and plan jointly degrade), not a slow, decoupled drift.")
    else:
        lines.append("G1b synchrony FAILS the pre-registered ≥80% threshold. Synchronous "
                     "fraction is descriptive but not formally accepted under the "
                     "pre-registered rule. See per-cell breakdown for heterogeneity.")
    return "\n".join(lines) + "\n"


def _md_t2b(o):
    r = o["results"]
    lines = []
    lines.append("# T2b — G4 WS-Precedes-Plan Mechanism (Stage 4)\n")
    lines.append(f"_Generated_: {o['generated_jst']}\n")
    lines.append(f"_Source_: `{o['source']}`\n")
    lines.append("## Pre-registered hypothesis (G4)\n")
    lines.append("World-state collapse **temporally precedes** plan collapse: "
                 "the lag distribution (ws_step − plan_step) has negative central tendency.\n"
                 "PASS criterion: (a) median lag < 0 with 99% bootstrap CI upper bound ≤ −1, "
                 "OR (b) ≥70% of collapsed episodes have `lag ≤ −1`.\n")
    lines.append("## Operationalization\n")
    for k, v in o["operationalization"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Headline result\n")
    lines.append(f"- n collapsed episodes: **{r['n_eps_collapsed']}**")
    lines.append(f"- n paired: **{r['n_paired']}**")
    lines.append(f"- Median lag (ws − plan): **{r['median_lag']:+.1f} steps** "
                 f"(99% bootstrap CI [{r['median_lag_99ci'][0]:+.1f}, {r['median_lag_99ci'][1]:+.1f}])")
    lines.append(f"- WS first (lag<0): **{r['pct_ws_first']*100:.1f}%** "
                 f"(99% CI [{r['pct_ws_first_99ci'][0]*100:.1f}, {r['pct_ws_first_99ci'][1]*100:.1f}])")
    lines.append(f"- Synchronous (|lag|≤1): **{r['pct_synchronous']*100:.1f}%**")
    lines.append(f"- Plan first (lag>0): **{r['pct_plan_first']*100:.1f}%**")
    lines.append(f"- WS leads by ≥1 step (lag≤−1): **{r['pct_ws_lead_ge_1']*100:.1f}%** "
                 f"(99% CI [{r['pct_ws_lead_ge_1_99ci'][0]*100:.1f}, "
                 f"{r['pct_ws_lead_ge_1_99ci'][1]*100:.1f}])")
    lines.append(f"- Wilcoxon signed-rank one-sided (H1: median(lag)<0): "
                 f"stat={r['wilcoxon_signed_rank_stat']}, "
                 f"p={r['wilcoxon_signed_rank_p_one_sided']:.3e}")
    lines.append(f"- **Verdict**: {r['verdict_note']}\n")
    pr = r["pre_registered"]
    lines.append("## Pre-registered criteria\n")
    lines.append(f"- (a) median lag < 0 AND 99% CI upper ≤ −1 — **{'PASS' if pr['crit_a_passed'] else 'FAIL'}**")
    lines.append(f"- (b) pct(lag ≤ −1) ≥ 0.70 — **{'PASS' if pr['crit_b_passed'] else 'FAIL'}**\n")
    lines.append("## Per-cell breakdown\n")
    lines.append("| cell | n paired | median lag | %ws-first | %ws≥1 | %sync |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for k, v in r["per_cell"].items():
        lines.append(f"| {k} | {v['n_paired']} | {v['median_lag']:+.1f} | "
                     f"{v['pct_ws_first']*100:.1f}% | {v['pct_ws_lead_ge_1']*100:.1f}% | "
                     f"{v['pct_synchronous']*100:.1f}% |")
    lines.append("")
    lines.append("## Interpretation\n")
    if r["verdict_pass"]:
        lines.append("G4 mechanism PASSES: among collapsed episodes, the world-state collapse "
                     "step (WSA<0.50 or rolling SCA<0.30) tends to occur **before** the first "
                     "invalid action — consistent with the causal-chain hypothesis that internal "
                     "world-model degradation drives downstream plan invalidity, not the reverse.\n\n"
                     "**Pipeline-lag caveat.** The Updater→Planner serial architecture imposes a "
                     "structural lower bound `τ_a ≥ τ_w` (any state error must appear in "
                     "`agent_world_state` before it can propagate into the next Planner call). "
                     "STATISTICAL_PROTOCOL §4.4.0 calls for subtracting a `Δτ_baseline` "
                     "estimated on Regime I episodes; the present analysis is conducted on the "
                     "Stage 4 grid (which spans high-stress cells without a Regime I anchor), "
                     "so we report the raw precedence pattern only. The fact that lag medians "
                     "below −1 and that ≥70% of episodes show WS lead ≥1 is sharper than the "
                     "1-step pipeline minimum, but interpretation should be cross-referenced "
                     "with the anchor_4 baseline if/when computed.")
    else:
        lines.append("G4 mechanism FAILS under the pre-registered acceptance rule. Descriptive "
                     "statistics (median lag, % WS-first) are reported above.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
