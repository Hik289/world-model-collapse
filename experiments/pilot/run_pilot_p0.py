#!/usr/bin/env python3
"""Run Pilot Slice P0 (StatefulPuzzle Regime I baseline).

Director dispatch (Stage 3):
  P0: stateful_puzzle × gpt-4o-mini × Regime I × 30 task = 30 ep
      threshold: gpt-4o-mini final_success ≥ 80% per env (H0.anchor_4)

Outputs:
  experiments/pilot/p0_results.json
  data/raw_logs/pilot_p0_step.jsonl   pilot_p0_episode.jsonl
  data/raw_logs/cost_tracker.jsonl    (per BUDGET_PLAN App. B)

Cost guards (BUDGET_PLAN §10):
  - CostTracker monitors mini $/ep, deviations, JSON retry, token bloat.
  - First check at ep>=10. If triggered → stop dispatching new episodes.
"""

from __future__ import annotations

import argparse
import hashlib
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


def jst_now() -> str:
    return datetime.now(tz=timezone(timedelta(hours=9))).isoformat()


# Configurations -----------------------------------------------------------

# Regime I (Stable): T=40, state_card=5, dep_density=1, branching=2, obs=clean, mut=static
REGIME_I_STRESS = {
    "T": 40, "state_card": 5, "dep_density": 1, "branching": 2,
    "obs_noise": "clean", "mut_rate": "static",
}

ENVS_P0 = ["stateful_puzzle"]
MODEL = "gpt-4o-mini"
MEMORY_MODE = "C_struct"


def _env_hash(env_name: str) -> int:
    """Stable env-name → int (SHA-256-derived), preserving the P0 seed scheme."""
    return int.from_bytes(hashlib.sha256(env_name.encode("utf-8")).digest()[:4], "big") & 0xFFFF


def build_p0_cells(n_task_per_env: int = 30, decoding_seed: int = 42) -> list[CellSpec]:
    cells: list[CellSpec] = []
    for env in ENVS_P0:
        for i in range(n_task_per_env):
            task_seed = 200000 + _env_hash(env) * 1000 + i
            cells.append(CellSpec(
                env_name=env,
                model=MODEL,
                stress_config=REGIME_I_STRESS,
                task_config={"archetype": "pilot_p0", "stress_config": REGIME_I_STRESS},
                task_seed=task_seed,
                decoding_seed=decoding_seed,
                world_regime="I_stable",
                task_id=f"p0_{env}_t{i:03d}",
                memory_mode=MEMORY_MODE,
            ))
    return cells


# ---------------------------------------------------------------------------
# P0 analysis: per-env final_success rate vs anchor_4 ≥ 0.80 threshold
# ---------------------------------------------------------------------------

def analyze_p0(outcomes: list[EpisodeOutcome]) -> dict:
    by_env: dict[str, dict] = {}
    for o in outcomes:
        env = o.cell.env_name
        d = by_env.setdefault(env, {"n": 0, "n_success": 0, "n_error": 0,
                                    "total_steps": 0, "total_in_tok": 0,
                                    "total_out_tok": 0, "total_cost_usd": 0.0})
        d["n"] += 1
        if o.success:
            d["n_success"] += 1
        if o.error:
            d["n_error"] += 1
        d["total_steps"] += o.steps
        d["total_in_tok"] += o.input_tokens
        d["total_out_tok"] += o.output_tokens
        d["total_cost_usd"] += o.cost_usd

    summary: dict = {"per_env": {}, "anchor_4_threshold": 0.80}
    all_pass = True
    for env, d in by_env.items():
        rate = d["n_success"] / d["n"] if d["n"] else 0.0
        passed = rate >= 0.80
        all_pass = all_pass and passed
        summary["per_env"][env] = {
            "n": d["n"],
            "n_success": d["n_success"],
            "success_rate": rate,
            "n_error": d["n_error"],
            "mean_steps": d["total_steps"] / d["n"] if d["n"] else 0.0,
            "total_input_tokens": d["total_in_tok"],
            "total_output_tokens": d["total_out_tok"],
            "total_cost_usd": round(d["total_cost_usd"], 6),
            "anchor_4_passed": passed,
        }
    summary["anchor_4_overall_passed"] = all_pass
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=4,
                    help="Concurrent episode workers (default 4).")
    ap.add_argument("--p0-n-task", type=int, default=30,
                    help="N task per env in P0 (default 30 per Director).")
    args = ap.parse_args()

    out_dir = ROOT / "experiments" / "pilot"
    log_dir = ROOT / "data" / "raw_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cost_tracker_path = log_dir / "cost_tracker.jsonl"

    client = LLMClient()

    print(f"[pilot] === P0 START (StatefulPuzzle × mini × Regime I × {args.p0_n_task} task = {args.p0_n_task} ep) ===")
    ct = CostTracker(
        out_path=cost_tracker_path,
        phase="pilot",
        slice_name="pilot_p0_regime_I",
        emit_every=10,
    )
    cells = build_p0_cells(n_task_per_env=args.p0_n_task)

    def progress(i, n, o: EpisodeOutcome):
        tag = "OK" if o.error is None else "ERR"
        success = "✓" if o.success else "✗"
        print(f"[P0 {i}/{n}] {o.cell.task_id} {success}{tag} steps={o.steps} in={o.input_tokens} out={o.output_tokens} cost=${o.cost_usd:.4f}"
              + (f" err={o.error}" if o.error else ""))

    outcomes_p0 = run_pilot_slice(
        cells=cells,
        client=client,
        step_jsonl_path=log_dir / "pilot_p0_step.jsonl",
        episode_jsonl_path=log_dir / "pilot_p0_episode.jsonl",
        cost_tracker=ct,
        n_workers=args.n_workers,
        progress_fn=progress,
    )

    p0_summary = analyze_p0(outcomes_p0)
    p0_summary["meta"] = {
        "timestamp_jst": jst_now(),
        "model": MODEL,
        "memory_mode": MEMORY_MODE,
        "regime": "I_stable",
        "stress_config": REGIME_I_STRESS,
        "n_workers": args.n_workers,
        "cost_tracker_triggered": ct.is_stopped(),
        "cost_tracker_stop_reason": ct.stop_reason(),
    }
    with (out_dir / "p0_results.json").open("w") as f:
        json.dump(p0_summary, f, sort_keys=True, indent=2, ensure_ascii=False)

    print("\n[pilot] === P0 SUMMARY ===")
    for env, info in p0_summary["per_env"].items():
        print(f"  {env}: {info['n_success']}/{info['n']} ({info['success_rate']:.0%}) "
              f"steps_mean={info['mean_steps']:.1f} cost=${info['total_cost_usd']:.3f} "
              f"anchor_4={'PASS' if info['anchor_4_passed'] else 'FAIL'}")
    print(f"  overall anchor_4 (≥80% per env): {'PASS' if p0_summary['anchor_4_overall_passed'] else 'FAIL'}")
    print(f"  cost_tracker triggered? {p0_summary['meta']['cost_tracker_triggered']} reason={p0_summary['meta']['cost_tracker_stop_reason']}")

    if not p0_summary["anchor_4_overall_passed"]:
        print("\n[pilot] P0 FAILED anchor_4 — env may be too hard. STOPPING. Reporting to Director.")
        return 2

    if ct.is_stopped():
        print(f"\n[pilot] cost_tracker triggered in P0. STOPPING. Reason: {ct.stop_reason()}")
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
