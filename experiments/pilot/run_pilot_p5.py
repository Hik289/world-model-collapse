#!/usr/bin/env python3
"""Run Pilot Slice P5 — gpt-4o cross-tier anchor_3 sanity (EXP_PLAN §5.6).

Spec (EXP_PLAN §5.6):
  - graph_nav × 4 dep_density levels {1,2,4,6} × 5 task × 1 seed × gpt-4o = 20 ep
  - $30 budget (rough estimate ~$1.50/ep on gpt-4o)
  - Acceptance: JSON valid ≥ 95% on gpt-4o specifically
  - NOT counted in G7 sign-consistency (Pilot anchor_3 cross-tier sanity only)

Outputs:
  - experiments/pilot/p5_results.json
  - data/raw_logs/pilot_p5_{step,episode}.jsonl

Director conditions (Path D2 後續):
  - Run while ERR bug fix is applied in src/environments/base.py
  - Report STAGE_3_PILOT_P5_RESULT + ERR_FIX_DONE + STAGE_4_READY_CHECKLIST_DONE
  - NOT auto-launch Stage 4 — Director decides
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.llm_client import LLMClient  # noqa: E402
from src.runner import (  # noqa: E402
    CellSpec, CostTracker, EpisodeOutcome, run_pilot_slice,
)


P5_BACKDROP = {
    "T": 40, "state_card": 10, "branching": 4,
    "obs_noise": "clean", "mut_rate": "static",
}
P5_DEP_LEVELS = [1, 2, 4, 6]
MODEL = "gpt-4o"
MEMORY_MODE = "C_struct"


def jst_now() -> str:
    return datetime.now(tz=timezone(timedelta(hours=9))).isoformat()


def build_p5_cells(n_task_per_level: int = 5, decoding_seed: int = 42) -> list[CellSpec]:
    """graph_nav × 4 dep_density levels × 5 task × gpt-4o (EXP_PLAN §5.6)."""
    cells: list[CellSpec] = []
    env = "graph_nav"
    for level in P5_DEP_LEVELS:
        stress = dict(P5_BACKDROP, dep_density=level)
        for i in range(n_task_per_level):
            # New task_seed namespace 600000-base for P5
            task_seed = 600000 + level * 10000 + i
            cells.append(CellSpec(
                env_name=env,
                model=MODEL,
                stress_config=stress,
                task_config={"archetype": "pilot_p5_gpt4o", "stress_config": stress},
                task_seed=task_seed,
                decoding_seed=decoding_seed,
                world_regime="III_coupled_backdrop",
                task_id=f"p5_{env}_dep{level}_t{i:02d}",
                memory_mode=MEMORY_MODE,
            ))
    return cells


def analyze_p5(outcomes: list[EpisodeOutcome]) -> dict:
    from collections import defaultdict
    by_lvl = defaultdict(lambda: {"n": 0, "n_success": 0, "n_error": 0,
                                  "total_steps": 0, "total_in": 0,
                                  "total_out": 0, "total_cost": 0.0})
    for o in outcomes:
        lvl = int(o.cell.stress_config["dep_density"])
        d = by_lvl[lvl]
        d["n"] += 1
        if o.success: d["n_success"] += 1
        if o.error: d["n_error"] += 1
        d["total_steps"] += o.steps
        d["total_in"] += o.input_tokens
        d["total_out"] += o.output_tokens
        d["total_cost"] += o.cost_usd

    levels = sorted(by_lvl.keys())
    per_level = {}
    for lvl in levels:
        d = by_lvl[lvl]
        n = d["n"]
        per_level[str(lvl)] = {
            "n": n, "n_success": d["n_success"],
            "success_rate": d["n_success"] / n if n else 0.0,
            "n_error": d["n_error"],
            "mean_steps": d["total_steps"] / n if n else 0.0,
            "total_input_tokens": d["total_in"],
            "total_output_tokens": d["total_out"],
            "total_cost_usd": round(d["total_cost"], 4),
        }

    total_ep = sum(d["n"] for d in by_lvl.values())
    total_errors = sum(d["n_error"] for d in by_lvl.values())
    total_cost = sum(d["total_cost"] for d in by_lvl.values())

    rates = [per_level[str(l)]["success_rate"] for l in levels]
    adjacent_drops = []
    for i in range(len(levels) - 1):
        a, b = levels[i], levels[i + 1]
        ra, rb = rates[i], rates[i + 1]
        adjacent_drops.append({
            "pair_lower_level": a, "pair_upper_level": b,
            "success_rate_lower": ra, "success_rate_upper": rb,
            "delta_pp": round((ra - rb) * 100.0, 2),
        })

    return {
        "per_level": per_level,
        "adjacent_drops": adjacent_drops,
        "total_episodes": total_ep,
        "total_errors": total_errors,
        "total_cost_usd": round(total_cost, 4),
        "env": "graph_nav",
        "model": MODEL,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--n-task", type=int, default=5)
    args = ap.parse_args()

    out_dir = ROOT / "experiments" / "pilot"
    log_dir = ROOT / "data" / "raw_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    client = LLMClient()
    ct = CostTracker(
        out_path=log_dir / "cost_tracker.jsonl",
        phase="pilot",
        slice_name="pilot_p5_gpt4o_anchor3",
        emit_every=5,
    )
    cells = build_p5_cells(n_task_per_level=args.n_task)
    print(f"[pilot] === P5 START (gpt-4o × graph_nav × 4 dep × {args.n_task} task = {len(cells)} ep) ===")

    t_start = time.perf_counter()

    def progress(i, n, o: EpisodeOutcome):
        success = "✓" if o.success else "✗"
        tag = "OK" if o.error is None else "ERR"
        print(f"[P5 {i}/{n}] {o.cell.task_id} {success}{tag} "
              f"steps={o.steps} in={o.input_tokens} out={o.output_tokens} cost=${o.cost_usd:.4f}"
              + (f" err={o.error}" if o.error else ""))

    outcomes = run_pilot_slice(
        cells=cells,
        client=client,
        step_jsonl_path=log_dir / "pilot_p5_step.jsonl",
        episode_jsonl_path=log_dir / "pilot_p5_episode.jsonl",
        cost_tracker=ct,
        n_workers=args.n_workers,
        progress_fn=progress,
    )

    elapsed_s = time.perf_counter() - t_start
    summary = analyze_p5(outcomes)
    summary["meta"] = {
        "timestamp_jst": jst_now(),
        "model": MODEL,
        "memory_mode": MEMORY_MODE,
        "env": "graph_nav",
        "regime_backdrop": "III_coupled",
        "stress_backdrop": P5_BACKDROP,
        "dep_density_levels": P5_DEP_LEVELS,
        "n_task_per_level": args.n_task,
        "n_workers": args.n_workers,
        "wall_clock_s": round(elapsed_s, 1),
        "cost_tracker_triggered": ct.is_stopped(),
        "cost_tracker_stop_reason": ct.stop_reason(),
    }
    (out_dir / "p5_results.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False)
    )

    # Compute JSON valid rate from step JSONL post-hoc (anchor_3 acceptance)
    valid = 0
    total = 0
    with open(log_dir / "pilot_p5_step.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            step = json.loads(line)
            for ct_name in ("planner", "updater", "self_diag"):
                co = step.get(f"{ct_name}_call_outcome") or {}
                if "valid_json" in co:
                    total += 1
                    if co["valid_json"]:
                        valid += 1
    json_valid_rate = valid / total if total else 0.0
    summary["anchor_3_json_valid_rate"] = round(json_valid_rate, 4)
    summary["anchor_3_total_calls"] = total
    summary["anchor_3_valid_calls"] = valid
    summary["anchor_3_pass"] = json_valid_rate >= 0.95
    (out_dir / "p5_results.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False)
    )

    print("\n[pilot] === P5 SUMMARY (gpt-4o × graph_nav × dep_density) ===")
    for lvl_str, lvl_info in summary["per_level"].items():
        print(f"  dep_density={lvl_str}: {lvl_info['n_success']}/{lvl_info['n']} "
              f"({lvl_info['success_rate']:.0%}) cost=${lvl_info['total_cost_usd']:.4f} "
              f"n_error={lvl_info['n_error']}")
    print(f"\n  Total: {summary['total_episodes']} ep, ${summary['total_cost_usd']:.4f} cost, "
          f"{summary['total_errors']} errors")
    print(f"  anchor_3 JSON valid rate: {summary['anchor_3_json_valid_rate']:.1%} "
          f"({summary['anchor_3_valid_calls']}/{summary['anchor_3_total_calls']}) "
          f"{'PASS' if summary['anchor_3_pass'] else 'FAIL'} (threshold 95%)")
    print(f"\n[pilot] wall_clock = {elapsed_s/60.0:.1f} min")
    print(f"[pilot] cost_tracker triggered? {ct.is_stopped()} reason='{ct.stop_reason()}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
