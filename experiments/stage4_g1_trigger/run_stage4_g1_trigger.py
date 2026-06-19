#!/usr/bin/env python3
"""Stage 4 G1-trigger dispatch — stateful_puzzle × 4 sc × 4 dd × 100 task.

Pre-registered in STAGE_4_PREREQUISITE_CHECKLIST.md Item #7.
Director sign-off: 2026-06-01 00:09 UTC.

Reads:
  - experiments/stage4_prep/stage4_task_seeds.json (1600 cells, sha256 seeds)

Writes:
  - data/raw_logs/stage4_step.jsonl
  - data/raw_logs/stage4_episode.jsonl
  - data/raw_logs/cost_tracker.jsonl (incremental)
  - experiments/stage4_g1_trigger/stage4_results.json (on completion)
  - experiments/stage4_g1_trigger/completed_cells.json (every 50 ep)

Resumability:
  - On startup: load completed_cells.json, skip those task_ids
  - Atomic write: write to .tmp + os.replace

Cost-tracker 0 verify:
  - After 100 episodes complete, sum slice cost in cost_tracker.jsonl
  - If > $0.001 → log [STAGE_4_ABORT_COST_NONZERO] and os._exit(13)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.llm_client import LLMClient  # noqa: E402
from src.runner import (  # noqa: E402
    CellSpec, CostTracker, EpisodeOutcome, run_pilot_slice,
)


SLICE_NAME = "stage4_g1_trigger_sp_haiku"
SEEDS_PATH = ROOT / "experiments" / "stage4_prep" / "stage4_task_seeds.json"
OUT_DIR = ROOT / "experiments" / "stage4_g1_trigger"
LOG_DIR = ROOT / "data" / "raw_logs"
COMPLETED_PATH = OUT_DIR / "completed_cells.json"
RESULTS_PATH = OUT_DIR / "stage4_results.json"


def jst_now() -> str:
    return datetime.now(tz=timezone(timedelta(hours=9))).isoformat()


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def load_completed() -> set[str]:
    if not COMPLETED_PATH.exists():
        return set()
    try:
        data = json.loads(COMPLETED_PATH.read_text())
        return set(data.get("task_ids", []))
    except Exception:
        return set()


def save_completed(task_ids: set[str]) -> None:
    payload = {
        "schema_version": "stage4_completed_v1",
        "updated_jst": jst_now(),
        "n_completed": len(task_ids),
        "task_ids": sorted(task_ids),
    }
    atomic_write_json(COMPLETED_PATH, payload)


def cell_from_dict(d: dict) -> CellSpec:
    return CellSpec(
        env_name=d["env"],
        model=d["model"],
        stress_config=d["stress_config"],
        task_config=d["task_config"],
        task_seed=int(d["task_seed"]),
        decoding_seed=int(d["decoding_seed"]),
        world_regime=d["world_regime"],
        task_id=d["task_id"],
        memory_mode=d.get("memory_mode", "C_struct"),
    )


def verify_cost_zero_at_100ep(cost_tracker_path: Path, slice_name: str,
                              tolerance: float = 1e-3) -> tuple[bool, float, int]:
    """Sum episode.cost_usd for our slice across all rows; return (ok, total, n)."""
    total = 0.0
    n = 0
    if not cost_tracker_path.exists():
        return True, 0.0, 0  # no records yet → ok
    with open(cost_tracker_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("slice") != slice_name:
                continue
            ep_records = rec.get("episodes", [])
            if isinstance(ep_records, list):
                for er in ep_records:
                    if isinstance(er, dict):
                        c = float(er.get("cost_usd", 0.0))
                        total += c
                        n += 1
            # Also support legacy aggregate
            elif "total_cost_usd" in rec:
                total = max(total, float(rec.get("total_cost_usd", 0.0)))
                n = max(n, int(rec.get("n_episodes", 0)))
    return (total <= tolerance), total, n


def analyze_stage4(outcomes: list[EpisodeOutcome]) -> dict:
    """Per-cell aggregate over sc × dd × env (single env for Stage 4)."""
    from collections import defaultdict
    by_sc_dd = defaultdict(lambda: {"n": 0, "n_success": 0, "n_error": 0,
                                    "total_steps": 0, "total_in": 0, "total_out": 0,
                                    "total_cost": 0.0, "task_ids": []})
    for o in outcomes:
        sc = int(o.cell.stress_config["state_card"])
        dd = int(o.cell.stress_config["dep_density"])
        key = f"sc={sc},dd={dd}"
        d = by_sc_dd[key]
        d["n"] += 1
        d["task_ids"].append(o.cell.task_id)
        if o.success: d["n_success"] += 1
        if o.error: d["n_error"] += 1
        d["total_steps"] += o.steps
        d["total_in"] += o.input_tokens
        d["total_out"] += o.output_tokens
        d["total_cost"] += o.cost_usd

    out = {"per_cell": {}}
    for key, d in by_sc_dd.items():
        n = d["n"]
        out["per_cell"][key] = {
            "n": n,
            "n_success": d["n_success"],
            "success_rate": d["n_success"] / n if n else 0.0,
            "n_error": d["n_error"],
            "mean_steps": d["total_steps"] / n if n else 0.0,
            "total_input_tokens": d["total_in"],
            "total_output_tokens": d["total_out"],
            "total_cost_usd": round(d["total_cost"], 6),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--cost-verify-after", type=int, default=100)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[stage4] === Stage 4 G1-trigger dispatch ===")
    print(f"[stage4] start: {jst_now()}")

    # Load grid
    grid = json.loads(SEEDS_PATH.read_text())
    all_cells = [cell_from_dict(c) for c in grid["cells"]]

    # Resumability: skip completed
    completed = load_completed()
    if completed:
        cells = [c for c in all_cells if c.task_id not in completed]
        print(f"[stage4] resuming: {len(completed)} done previously, {len(cells)} remaining")
    else:
        cells = all_cells
        print(f"[stage4] fresh start: {len(cells)} cells")

    if not cells:
        print("[stage4] all cells already complete; computing final aggregate")
        return _finalize(all_cells, [])

    client = LLMClient()
    ct = CostTracker(
        out_path=LOG_DIR / "cost_tracker.jsonl",
        phase="stage4",
        slice_name=SLICE_NAME,
        emit_every=args.checkpoint_every,
    )

    completed_lock = threading.Lock()
    cost_verified = {"done": False, "abort": False}
    outcomes: list[EpisodeOutcome] = []
    t_start = time.perf_counter()

    def progress(i, n, o: EpisodeOutcome):
        success = "✓" if o.success else "✗"
        tag = "OK" if o.error is None else f"ERR({o.error[:60]})"
        elapsed = (time.perf_counter() - t_start) / 60.0
        rate = i / max(elapsed, 0.01)
        print(f"[stage4 {i}/{n}] {o.cell.task_id} {success}{tag} "
              f"steps={o.steps} | elapsed={elapsed:.1f}min rate={rate:.2f}ep/min")
        # Checkpoint
        with completed_lock:
            completed.add(o.cell.task_id)
            outcomes.append(o)
            if i % args.checkpoint_every == 0 or i == n:
                save_completed(completed)
                print(f"[stage4]   checkpoint: {len(completed)}/{len(all_cells)} cells done")
        # Cost verify at 100 ep
        if not cost_verified["done"] and i >= args.cost_verify_after:
            cost_verified["done"] = True
            ok, total, ncost = verify_cost_zero_at_100ep(
                LOG_DIR / "cost_tracker.jsonl", SLICE_NAME, tolerance=1e-3,
            )
            print(f"[stage4]   cost-verify @ {i} ep: total=${total:.6f} over {ncost} records, ok={ok}")
            if not ok:
                cost_verified["abort"] = True
                print(f"[STAGE_4_ABORT_COST_NONZERO] total=${total:.4f} > tolerance; aborting")
                save_completed(completed)
                os._exit(13)

    print(f"[stage4] launching run_pilot_slice (n_workers={args.n_workers}) ...")
    outs = run_pilot_slice(
        cells=cells,
        client=client,
        step_jsonl_path=LOG_DIR / "stage4_step.jsonl",
        episode_jsonl_path=LOG_DIR / "stage4_episode.jsonl",
        cost_tracker=ct,
        n_workers=args.n_workers,
        progress_fn=progress,
    )
    outcomes.extend(outs)  # progress() already appends; redundancy safe via dedup later
    return _finalize(all_cells, outcomes)


def _finalize(all_cells: list, outcomes: list) -> int:
    # Dedup outcomes by task_id (progress may have appended each twice)
    seen: dict[str, EpisodeOutcome] = {}
    for o in outcomes:
        seen[o.cell.task_id] = o
    uniq = list(seen.values())

    summary = analyze_stage4(uniq)
    summary["meta"] = {
        "timestamp_jst": jst_now(),
        "slice": SLICE_NAME,
        "model": "claude-haiku-4-5",
        "env": "stateful_puzzle",
        "grid_axes": {
            "state_cards": [5, 10, 20, 40],
            "dep_densities": [1, 2, 4, 6],
            "n_task_per_cell": 100,
            "total_cells_planned": len(all_cells),
        },
        "n_completed": len(uniq),
    }
    atomic_write_json(RESULTS_PATH, summary)
    save_completed({o.cell.task_id for o in uniq})

    print(f"\n[stage4] === DONE === wrote {RESULTS_PATH}")
    print(f"[stage4] completed {len(uniq)}/{len(all_cells)} cells")
    # Compact 4x4 grid display
    print("\n  success_rate grid (rows=state_card, cols=dep_density):")
    print(f"  {'sc\\dd':>6} | {'1':>6} {'2':>6} {'4':>6} {'6':>6}")
    for sc in [5, 10, 20, 40]:
        row = [f"{sc:>6}"]
        for dd in [1, 2, 4, 6]:
            key = f"sc={sc},dd={dd}"
            cell = summary["per_cell"].get(key, {})
            sr = cell.get("success_rate")
            row.append(f"{sr:>6.0%}" if sr is not None else "  N/A")
        print("  " + " ".join(row).replace(" | ", " | ", 1) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
