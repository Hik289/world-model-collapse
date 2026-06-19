#!/usr/bin/env python3
"""Run Pilot P1 Step 2 — anchor_5 supplementary on L4+L6 pair.

Director Path A2, condition #1 (2026-05-30):
  - L4 + L6 each: +10 NEW task_seeds × 2 NEW decoding_seeds = 20 ep/level
  - Total: 40 ep extra ($4.40 from S7 dual-use budget)
  - Merge with P1 Step 1 data → L4 30 ep + L6 30 ep
  - Wilson 99% CI on Δp̂(L4→L6):
      lower bound > 10pp  → PASS robust → progress to P2
      lower bound ≤ 10pp → PASS but confidence insufficient → report Director

Output: experiments/pilot/p1_step2_results.json (Step 2 only)
        experiments/pilot/p1_merged_results.json (Step 1 + Step 2 merged)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.llm_client import LLMClient  # noqa: E402
from src.runner import (  # noqa: E402
    CellSpec, CostTracker, EpisodeOutcome, run_pilot_slice,
)


P1_BACKDROP = {
    "T": 40, "state_card": 10, "branching": 4,
    "obs_noise": "clean", "mut_rate": "static",
}
MODEL = "gpt-4o-mini"
MEMORY_MODE = "C_struct"

# Step 2 uses NEW task indices (10-19) + NEW decoding seeds (43, 44),
# independent of Step 1's 0-9 + seed 42.
STEP2_LEVELS = [4, 6]
STEP2_TASK_RANGE = range(10, 20)       # 10 new task indices per level
STEP2_DECODING_SEEDS = [43, 44]        # 2 new decoding seeds


def jst_now() -> str:
    return datetime.now(tz=timezone(timedelta(hours=9))).isoformat()


def build_step2_cells() -> list[CellSpec]:
    """L4 + L6 × 10 task × 2 decoding seeds = 40 cells."""
    cells: list[CellSpec] = []
    env = "graph_nav"
    for level in STEP2_LEVELS:
        stress = dict(P1_BACKDROP, dep_density=level)
        for task_i in STEP2_TASK_RANGE:
            for ds in STEP2_DECODING_SEEDS:
                # Same task_seed scheme as P1 Step 1: 300000 + level*10000 + task_i
                # (task_i = 10..19 is the new range disjoint from Step 1's 0..9)
                task_seed = 300000 + level * 10000 + task_i
                cells.append(CellSpec(
                    env_name=env,
                    model=MODEL,
                    stress_config=stress,
                    task_config={"archetype": "pilot_p1_step2_anchor5", "stress_config": stress},
                    task_seed=task_seed,
                    decoding_seed=ds,
                    world_regime="III_coupled_backdrop",
                    task_id=f"p1s2_dep{level}_t{task_i:02d}_ds{ds}",
                    memory_mode=MEMORY_MODE,
                ))
    return cells


# ---------------------------------------------------------------------------
# Wilson 99% CI helpers
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.01) -> tuple[float, float]:
    """Two-sided Wilson score interval (1-alpha CI). Returns (lo, hi)."""
    if n == 0:
        return (0.0, 1.0)
    from math import sqrt
    # 1-alpha two-sided → use z such that Phi(z) = 1 - alpha/2
    # alpha=0.01 → z ≈ 2.5758 (99% two-sided)
    z = 2.5758293035489
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def newcombe_diff_ci(k1: int, n1: int, k2: int, n2: int, alpha: float = 0.01) -> tuple[float, float]:
    """Newcombe hybrid score interval on (p1 - p2). 1-alpha two-sided CI.

    Approximation: combine Wilson CIs via Newcombe's method 10 (hybrid).
    For Δp̂ = p1 - p2, lo = p1_hat - p2_hat - sqrt((p1_hat - lo1)^2 + (hi2 - p2_hat)^2)
                       hi = p1_hat - p2_hat + sqrt((hi1 - p1_hat)^2 + (p2_hat - lo2)^2)
    """
    p1 = k1 / n1 if n1 else 0.0
    p2 = k2 / n2 if n2 else 0.0
    lo1, hi1 = wilson_ci(k1, n1, alpha=alpha)
    lo2, hi2 = wilson_ci(k2, n2, alpha=alpha)
    delta = p1 - p2
    delta_lo = delta - math.sqrt((p1 - lo1) ** 2 + (hi2 - p2) ** 2)
    delta_hi = delta + math.sqrt((hi1 - p1) ** 2 + (p2 - lo2) ** 2)
    return (delta_lo, delta_hi)


# ---------------------------------------------------------------------------
# Merge with P1 Step 1 + compute decision
# ---------------------------------------------------------------------------

def load_p1_step1_episodes() -> list[dict]:
    """Load Step 1 episodes from raw_logs/pilot_p1_episode.jsonl."""
    path = ROOT / "data" / "raw_logs" / "pilot_p1_episode.jsonl"
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def merge_and_decide(step2_outcomes: list[EpisodeOutcome]) -> dict:
    """Merge Step 1 + Step 2 per level; compute Wilson + Newcombe CI."""
    step1_eps = load_p1_step1_episodes()
    # Filter Step 1 to graph_nav only (whole P1 is graph_nav so should match)
    step1_l4 = [e for e in step1_eps if e["stress_config"]["dep_density"] == 4]
    step1_l6 = [e for e in step1_eps if e["stress_config"]["dep_density"] == 6]
    step2_l4 = [o for o in step2_outcomes if o.cell.stress_config["dep_density"] == 4]
    step2_l6 = [o for o in step2_outcomes if o.cell.stress_config["dep_density"] == 6]

    def merge(step1_list: list[dict], step2_list: list[EpisodeOutcome]) -> tuple[int, int]:
        n1 = len(step1_list)
        k1 = sum(1 for e in step1_list if e["final_success"])
        n2 = len(step2_list)
        k2 = sum(1 for o in step2_list if o.success)
        return (k1 + k2, n1 + n2)

    k_l4, n_l4 = merge(step1_l4, step2_l4)
    k_l6, n_l6 = merge(step1_l6, step2_l6)
    p_l4 = k_l4 / n_l4 if n_l4 else 0.0
    p_l6 = k_l6 / n_l6 if n_l6 else 0.0
    delta_p = p_l4 - p_l6   # positive = success drops from L4 to L6 (H0-aligned)

    wilson_l4 = wilson_ci(k_l4, n_l4, alpha=0.01)
    wilson_l6 = wilson_ci(k_l6, n_l6, alpha=0.01)
    delta_ci = newcombe_diff_ci(k_l4, n_l4, k_l6, n_l6, alpha=0.01)

    decision_lower = delta_ci[0] * 100.0  # pp
    if decision_lower > 10.0:
        decision = "PASS_ROBUST"
        rationale = f"Newcombe 99% CI lower bound on Δp̂ = {decision_lower:.1f}pp > 10pp"
    else:
        decision = "PASS_BUT_LOW_CONFIDENCE"
        rationale = f"Newcombe 99% CI lower bound on Δp̂ = {decision_lower:.1f}pp ≤ 10pp; report Director"

    return {
        "level_4": {
            "step1_k": sum(1 for e in step1_l4 if e["final_success"]),
            "step1_n": len(step1_l4),
            "step2_k": sum(1 for o in step2_l4 if o.success),
            "step2_n": len(step2_l4),
            "merged_k": k_l4,
            "merged_n": n_l4,
            "merged_p": p_l4,
            "wilson_99ci": list(wilson_l4),
        },
        "level_6": {
            "step1_k": sum(1 for e in step1_l6 if e["final_success"]),
            "step1_n": len(step1_l6),
            "step2_k": sum(1 for o in step2_l6 if o.success),
            "step2_n": len(step2_l6),
            "merged_k": k_l6,
            "merged_n": n_l6,
            "merged_p": p_l6,
            "wilson_99ci": list(wilson_l6),
        },
        "delta_p_l4_minus_l6_pp": round(delta_p * 100.0, 2),
        "newcombe_99ci_pp": [round(delta_ci[0] * 100.0, 2), round(delta_ci[1] * 100.0, 2)],
        "decision": decision,
        "rationale": rationale,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=4)
    args = ap.parse_args()

    out_dir = ROOT / "experiments" / "pilot"
    log_dir = ROOT / "data" / "raw_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    client = LLMClient()
    ct = CostTracker(
        out_path=log_dir / "cost_tracker.jsonl",
        phase="pilot",
        slice_name="pilot_p1_step2_anchor5_supplementary",
        emit_every=10,
    )
    cells = build_step2_cells()

    print(f"[pilot] === P1 Step 2 START ({len(cells)} ep, L4/L6 × 10 new tasks × 2 new seeds) ===")

    def progress(i, n, o: EpisodeOutcome):
        tag = "OK" if o.error is None else "ERR"
        success = "✓" if o.success else "✗"
        print(f"[P1S2 {i}/{n}] {o.cell.task_id} {success}{tag} "
              f"steps={o.steps} in={o.input_tokens} out={o.output_tokens} cost=${o.cost_usd:.4f}"
              + (f" err={o.error}" if o.error else ""))

    outcomes = run_pilot_slice(
        cells=cells,
        client=client,
        step_jsonl_path=log_dir / "pilot_p1_step2_step.jsonl",
        episode_jsonl_path=log_dir / "pilot_p1_step2_episode.jsonl",
        cost_tracker=ct,
        n_workers=args.n_workers,
        progress_fn=progress,
    )

    # Per-level Step 2 summary
    by_lvl: dict[int, dict] = {}
    for o in outcomes:
        lvl = int(o.cell.stress_config["dep_density"])
        d = by_lvl.setdefault(lvl, {"n": 0, "n_success": 0, "n_error": 0,
                                    "total_in": 0, "total_out": 0, "total_cost": 0.0,
                                    "total_steps": 0})
        d["n"] += 1
        if o.success:
            d["n_success"] += 1
        if o.error:
            d["n_error"] += 1
        d["total_in"] += o.input_tokens
        d["total_out"] += o.output_tokens
        d["total_cost"] += o.cost_usd
        d["total_steps"] += o.steps

    step2_per_lvl = {}
    for lvl, d in by_lvl.items():
        n = d["n"]
        step2_per_lvl[str(lvl)] = {
            "n": n,
            "n_success": d["n_success"],
            "success_rate": d["n_success"] / n if n else 0.0,
            "mean_steps": d["total_steps"] / n if n else 0.0,
            "total_cost_usd": round(d["total_cost"], 6),
        }

    step2_doc = {
        "step": 2,
        "timestamp_jst": jst_now(),
        "per_level": step2_per_lvl,
        "n_workers": args.n_workers,
        "cost_tracker_triggered": ct.is_stopped(),
        "cost_tracker_stop_reason": ct.stop_reason(),
    }
    (out_dir / "p1_step2_results.json").write_text(
        json.dumps(step2_doc, sort_keys=True, indent=2, ensure_ascii=False)
    )

    # Merge with Step 1 & compute Wilson + Newcombe CIs
    merged = merge_and_decide(outcomes)
    merged["meta"] = {
        "timestamp_jst": jst_now(),
        "model": MODEL,
        "memory_mode": MEMORY_MODE,
        "env": "graph_nav",
        "regime_backdrop": "III_coupled",
        "stress_backdrop": P1_BACKDROP,
        "alpha": 0.01,
        "ci_method": "newcombe_hybrid_score_99pct",
    }
    (out_dir / "p1_merged_results.json").write_text(
        json.dumps(merged, sort_keys=True, indent=2, ensure_ascii=False)
    )

    print("\n[pilot] === P1 Step 2 SUMMARY ===")
    for lvl, info in step2_per_lvl.items():
        print(f"  Step 2 dep={lvl}: {info['n_success']}/{info['n']} ({info['success_rate']:.0%}) "
              f"mean_steps={info['mean_steps']:.1f} cost=${info['total_cost_usd']:.3f}")
    print("\n[pilot] === P1 MERGED (Step 1 + Step 2) ===")
    for lvl_key in ("level_4", "level_6"):
        m = merged[lvl_key]
        wci = m["wilson_99ci"]
        print(f"  {lvl_key}: step1={m['step1_k']}/{m['step1_n']}, step2={m['step2_k']}/{m['step2_n']}, "
              f"MERGED={m['merged_k']}/{m['merged_n']} ({m['merged_p']:.0%}); "
              f"Wilson 99% CI [{wci[0]:.3f}, {wci[1]:.3f}]")
    delta_pp = merged["delta_p_l4_minus_l6_pp"]
    nlo, nhi = merged["newcombe_99ci_pp"]
    print(f"\n  Δp̂(L4 - L6) = {delta_pp:+.1f}pp")
    print(f"  Newcombe 99% CI: [{nlo:.1f}pp, {nhi:.1f}pp]")
    print(f"  Decision: {merged['decision']} — {merged['rationale']}")

    print(f"\n[pilot] cost_tracker triggered? {ct.is_stopped()} reason='{ct.stop_reason()}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
