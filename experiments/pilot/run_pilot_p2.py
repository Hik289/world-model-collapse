#!/usr/bin/env python3
"""Run Pilot Slice P2 — cross-env dep_density replication.

Director Path C1 (2026-05-30):
  - tool_dag + stateful_puzzle × dep_density {1,2,4,6} × mini × 10 task × 1 seed = 80 ep
  - Budget $8.8 (per EXP_PLAN §5.3), ~5-6h ETA
  - No STOP rule (any result is informative, just report Director)
  - No auto-launch P3-P5

Outputs:
  - experiments/pilot/p2_results.json
  - data/raw_logs/pilot_p2_{step,episode}.jsonl
  - cost_tracker.jsonl appended

Backdrop: Regime III (state_card=10, branching=4, obs=clean, mut=static, T=40)
  — same as P1 for direct comparability.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.llm_client import LLMClient  # noqa: E402
from src.runner import (  # noqa: E402
    CellSpec, CostTracker, EpisodeOutcome, run_pilot_slice,
)


P2_BACKDROP = {
    "T": 40, "state_card": 10, "branching": 4,
    "obs_noise": "clean", "mut_rate": "static",
}
P2_DEP_LEVELS = [1, 2, 4, 6]
P2_ENVS = ["tool_dag", "stateful_puzzle"]
MODEL = "gpt-4o-mini"
MEMORY_MODE = "C_struct"


def jst_now() -> str:
    return datetime.now(tz=timezone(timedelta(hours=9))).isoformat()


def build_p2_cells(n_task_per_level: int = 10, decoding_seed: int = 42) -> list[CellSpec]:
    cells: list[CellSpec] = []
    for env in P2_ENVS:
        for level in P2_DEP_LEVELS:
            stress = dict(P2_BACKDROP, dep_density=level)
            for i in range(n_task_per_level):
                # Different namespace from P1 task_seeds: 400000 + env_offset + level*10000 + i
                env_offset = 100000 if env == "stateful_puzzle" else 0
                task_seed = 400000 + env_offset + level * 10000 + i
                cells.append(CellSpec(
                    env_name=env,
                    model=MODEL,
                    stress_config=stress,
                    task_config={"archetype": "pilot_p2", "stress_config": stress},
                    task_seed=task_seed,
                    decoding_seed=decoding_seed,
                    world_regime="III_coupled_backdrop",
                    task_id=f"p2_{env}_dep{level}_t{i:02d}",
                    memory_mode=MEMORY_MODE,
                ))
    return cells


def classify_shape(levels_sorted: list[int], rates: list[float]) -> str:
    """Classify dep_density 1D shape (Director condition #3).

    monotone: success strictly decreasing as level increases
    cliff:    one >= 20pp drop between adjacent + others < 10pp drift
    hump:     non-monotone with a local maximum at L>=L2
    flat:     max(|adjacent diff|) < 10pp
    """
    if len(rates) < 2:
        return "insufficient_data"
    diffs = [rates[i + 1] - rates[i] for i in range(len(rates) - 1)]  # signed
    max_abs = max(abs(d) for d in diffs)
    if max_abs < 0.10:
        return "flat"
    n_neg = sum(1 for d in diffs if d < 0)
    n_pos = sum(1 for d in diffs if d > 0)
    # monotone decreasing iff all diffs < 0 (or zero with small abs)
    if all(d <= 0 for d in diffs) and any(d < 0 for d in diffs):
        # Decide cliff vs monotone: cliff = exactly one |d| >= 0.20, others |d| < 0.10
        big = sum(1 for d in diffs if abs(d) >= 0.20)
        small = sum(1 for d in diffs if abs(d) < 0.10)
        if big == 1 and small == len(diffs) - 1:
            return "cliff"
        return "monotone"
    # If signs mixed → hump or u-shape
    # hump = +, then -. u = -, then +.
    if n_pos > 0 and n_neg > 0:
        # find first sign change
        signs = [1 if d > 0 else -1 for d in diffs]
        if signs[0] > 0 and signs[-1] < 0:
            return "hump"
        if signs[0] < 0 and signs[-1] > 0:
            return "u_shape"
        return "irregular"
    return "irregular"


def analyze_p2(outcomes: list[EpisodeOutcome]) -> dict:
    by_env_lvl: dict[str, dict[int, dict]] = {}
    for o in outcomes:
        env = o.cell.env_name
        lvl = int(o.cell.stress_config["dep_density"])
        e = by_env_lvl.setdefault(env, {})
        d = e.setdefault(lvl, {"n": 0, "n_success": 0, "n_error": 0,
                                "total_steps": 0, "total_in": 0,
                                "total_out": 0, "total_cost": 0.0})
        d["n"] += 1
        if o.success:
            d["n_success"] += 1
        if o.error:
            d["n_error"] += 1
        d["total_steps"] += o.steps
        d["total_in"] += o.input_tokens
        d["total_out"] += o.output_tokens
        d["total_cost"] += o.cost_usd

    summary: dict = {"per_env": {}}
    for env in P2_ENVS:
        per_level = {}
        levels = sorted(by_env_lvl.get(env, {}).keys())
        for lvl in levels:
            d = by_env_lvl[env][lvl]
            n = d["n"]
            per_level[str(lvl)] = {
                "n": n,
                "n_success": d["n_success"],
                "success_rate": d["n_success"] / n if n else 0.0,
                "n_error": d["n_error"],
                "mean_steps": d["total_steps"] / n if n else 0.0,
                "total_cost_usd": round(d["total_cost"], 6),
            }

        # Adjacent drops (positive = success FALLS)
        rates = [per_level[str(l)]["success_rate"] for l in levels]
        adjacent_drops = []
        for i in range(len(levels) - 1):
            a, b = levels[i], levels[i + 1]
            ra, rb = rates[i], rates[i + 1]
            adjacent_drops.append({
                "pair_lower_level": a,
                "pair_upper_level": b,
                "success_rate_lower": ra,
                "success_rate_upper": rb,
                "delta_pp": round((ra - rb) * 100.0, 2),
            })

        # Shape classification (using signed rate differences)
        shape = classify_shape(levels, rates)

        max_drop_pp = max((d["delta_pp"] for d in adjacent_drops), default=0.0)

        summary["per_env"][env] = {
            "per_level": per_level,
            "adjacent_drops": adjacent_drops,
            "max_adjacent_delta_pp": round(max_drop_pp, 2),
            "shape": shape,
        }

    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--n-task", type=int, default=10)
    args = ap.parse_args()

    out_dir = ROOT / "experiments" / "pilot"
    log_dir = ROOT / "data" / "raw_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    client = LLMClient()
    ct = CostTracker(
        out_path=log_dir / "cost_tracker.jsonl",
        phase="pilot",
        slice_name="pilot_p2_cross_env_dep_density",
        emit_every=10,
    )
    cells = build_p2_cells(n_task_per_level=args.n_task)
    print(f"[pilot] === P2 START (cross-env: 2 envs × 4 dep_density × {args.n_task} task = {len(cells)} ep) ===")

    def progress(i, n, o: EpisodeOutcome):
        tag = "OK" if o.error is None else "ERR"
        success = "✓" if o.success else "✗"
        print(f"[P2 {i}/{n}] {o.cell.task_id} {success}{tag} "
              f"steps={o.steps} in={o.input_tokens} out={o.output_tokens} cost=${o.cost_usd:.4f}"
              + (f" err={o.error}" if o.error else ""))

    outcomes = run_pilot_slice(
        cells=cells,
        client=client,
        step_jsonl_path=log_dir / "pilot_p2_step.jsonl",
        episode_jsonl_path=log_dir / "pilot_p2_episode.jsonl",
        cost_tracker=ct,
        n_workers=args.n_workers,
        progress_fn=progress,
    )

    summary = analyze_p2(outcomes)
    summary["meta"] = {
        "timestamp_jst": jst_now(),
        "model": MODEL,
        "memory_mode": MEMORY_MODE,
        "envs": P2_ENVS,
        "regime_backdrop": "III_coupled",
        "stress_backdrop": P2_BACKDROP,
        "dep_density_levels": P2_DEP_LEVELS,
        "n_task_per_level": args.n_task,
        "n_workers": args.n_workers,
        "cost_tracker_triggered": ct.is_stopped(),
        "cost_tracker_stop_reason": ct.stop_reason(),
    }
    (out_dir / "p2_results.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False)
    )

    print("\n[pilot] === P2 SUMMARY ===")
    for env in P2_ENVS:
        info = summary["per_env"].get(env, {})
        print(f"\n  -- {env} (shape={info.get('shape','?')}) --")
        for lvl_str, lvl_info in info.get("per_level", {}).items():
            print(f"    dep_density={lvl_str}: {lvl_info['n_success']}/{lvl_info['n']} "
                  f"({lvl_info['success_rate']:.0%}) steps_mean={lvl_info['mean_steps']:.1f}")
        for d in info.get("adjacent_drops", []):
            print(f"    Δp̂(L{d['pair_lower_level']}→L{d['pair_upper_level']}) = {d['delta_pp']:+.1f}pp "
                  f"({d['success_rate_lower']:.0%} → {d['success_rate_upper']:.0%})")
        print(f"    max adjacent Δp̂ = {info.get('max_adjacent_delta_pp',0):.1f}pp")

    # Compare to P1 graph_nav
    print("\n  [comparison] P1 graph_nav (Step 1+2 merged for L4, L6 / Step 1 only for L1, L2):")
    print(f"    graph_nav: L1=40% L2=60% L4=67% L6=47% (shape=hump per STAGE-3-009)")

    print(f"\n[pilot] cost_tracker triggered? {ct.is_stopped()} reason='{ct.stop_reason()}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
