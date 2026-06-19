#!/usr/bin/env python3
"""Stage 5 A — gpt-4o-mini × stateful_puzzle × 4×4 grid × n=50.

Director Path (Stage 5 A): cross-model G1 replication on mini.
Pre-registered: STAGE_4_PREREQUISITE_CHECKLIST.md Item #9.

Reads:
  - experiments/stage5_a/stage5_a_task_seeds.json (800 cells)

Writes:
  - data/raw_logs/stage5_a_step.jsonl
  - data/raw_logs/stage5_a_episode.jsonl
  - data/raw_logs/cost_tracker.jsonl (incremental)
  - experiments/stage5_a/stage5_a_results.json (on completion)
  - experiments/stage5_a/completed_cells.json (every 50 ep, atomic)

Cost cap: aborts if cumulative cost > $100 (hard cap per Director sign-off).
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


SLICE_NAME = "stage5_a_g1_trigger_sp_mini"
SEEDS_PATH = ROOT / "experiments" / "stage5_a" / "stage5_a_task_seeds.json"
OUT_DIR = ROOT / "experiments" / "stage5_a"
LOG_DIR = ROOT / "data" / "raw_logs"
COMPLETED_PATH = OUT_DIR / "completed_cells.json"
RESULTS_PATH = OUT_DIR / "stage5_a_results.json"


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
        "schema_version": "stage5_a_completed_v1",
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


def analyze_stage5(outcomes: list[EpisodeOutcome]) -> dict:
    from collections import defaultdict
    by_sc_dd = defaultdict(lambda: {"n": 0, "n_success": 0, "n_error": 0,
                                    "total_steps": 0, "total_in": 0, "total_out": 0,
                                    "total_cost": 0.0})
    for o in outcomes:
        sc = int(o.cell.stress_config["state_card"])
        dd = int(o.cell.stress_config["dep_density"])
        d = by_sc_dd[f"sc={sc},dd={dd}"]
        d["n"] += 1
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
    ap.add_argument("--cost-cap", type=float, default=100.0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[stage5a] === Stage 5 A G1-trigger dispatch (mini cross-model) ===")
    print(f"[stage5a] start: {jst_now()}  cost_cap=${args.cost_cap:.2f}")

    grid = json.loads(SEEDS_PATH.read_text())
    all_cells = [cell_from_dict(c) for c in grid["cells"]]

    completed = load_completed()
    if completed:
        cells = [c for c in all_cells if c.task_id not in completed]
        print(f"[stage5a] resuming: {len(completed)} done previously, {len(cells)} remaining")
    else:
        cells = all_cells
        print(f"[stage5a] fresh start: {len(cells)} cells")

    if not cells:
        return _finalize(all_cells, [])

    client = LLMClient()
    ct = CostTracker(
        out_path=LOG_DIR / "cost_tracker.jsonl",
        phase="stage5_a",
        slice_name=SLICE_NAME,
        emit_every=args.checkpoint_every,
    )

    lock = threading.Lock()
    state = {"cumcost": 0.0, "abort": False, "outcomes": [], "completed": completed}
    t_start = time.perf_counter()

    def progress(i, n, o: EpisodeOutcome):
        elapsed = (time.perf_counter() - t_start) / 60.0
        rate = i / max(elapsed, 0.01)
        with lock:
            state["cumcost"] += o.cost_usd
            state["completed"].add(o.cell.task_id)
            state["outcomes"].append(o)
            cum = state["cumcost"]
            success = "✓" if o.success else "✗"
            tag = "OK" if o.error is None else f"ERR({o.error[:50]})"
            print(f"[stage5a {i}/{n}] {o.cell.task_id} {success}{tag} "
                  f"steps={o.steps} cost=${o.cost_usd:.4f} cum=${cum:.2f} "
                  f"| {elapsed:.1f}min rate={rate:.2f}ep/min")
            # Checkpoint
            if i % args.checkpoint_every == 0 or i == n:
                save_completed(state["completed"])
                print(f"[stage5a]   checkpoint: {len(state['completed'])}/{len(all_cells)} cells done; cum_cost=${cum:.2f}")
            # Hard cost cap
            if cum > args.cost_cap and not state["abort"]:
                state["abort"] = True
                print(f"[STAGE_5_ABORT_COST_CAP] cum=${cum:.2f} > cap=${args.cost_cap:.2f}; aborting")
                save_completed(state["completed"])
                # Signal cost_tracker too so bounded-wave stops dispatching
                ct.force_stop(f"cost_cap_exceeded:${cum:.2f}>${args.cost_cap}")
                # Note: in-flight will drain, then runner exits the wait loop

    print(f"[stage5a] launching run_pilot_slice (bounded-wave, n_workers={args.n_workers}) ...")
    outs = run_pilot_slice(
        cells=cells,
        client=client,
        step_jsonl_path=LOG_DIR / "stage5_a_step.jsonl",
        episode_jsonl_path=LOG_DIR / "stage5_a_episode.jsonl",
        cost_tracker=ct,
        n_workers=args.n_workers,
        progress_fn=progress,
    )
    return _finalize(all_cells, state["outcomes"], cost_cap_hit=state["abort"],
                     final_cum_cost=state["cumcost"])


def _finalize(all_cells: list, outcomes: list, cost_cap_hit: bool = False,
              final_cum_cost: float = 0.0) -> int:
    # Dedup
    seen: dict[str, EpisodeOutcome] = {}
    for o in outcomes:
        seen[o.cell.task_id] = o
    uniq = list(seen.values())

    summary = analyze_stage5(uniq)
    summary["meta"] = {
        "timestamp_jst": jst_now(),
        "slice": SLICE_NAME,
        "model": "gpt-4o-mini",
        "env": "stateful_puzzle",
        "grid_axes": {
            "state_cards": [5, 10, 20, 40],
            "dep_densities": [1, 2, 4, 6],
            "n_task_per_cell": 50,
            "total_cells_planned": len(all_cells),
        },
        "n_completed": len(uniq),
        "cost_cap_hit": cost_cap_hit,
        "final_cum_cost_usd": round(final_cum_cost, 4),
    }
    atomic_write_json(RESULTS_PATH, summary)
    save_completed({o.cell.task_id for o in uniq})

    print(f"\n[stage5a] === DONE === wrote {RESULTS_PATH}")
    print(f"[stage5a] completed {len(uniq)}/{len(all_cells)} cells; final_cost=${final_cum_cost:.4f}")
    print("\n  success_rate grid (rows=state_card, cols=dep_density):")
    print(f"  {'sc\\dd':>6} | {'1':>6} {'2':>6} {'4':>6} {'6':>6}")
    for sc in [5, 10, 20, 40]:
        row = [f"{sc:>6}"]
        for dd in [1, 2, 4, 6]:
            key = f"sc={sc},dd={dd}"
            cell = summary["per_cell"].get(key, {})
            sr = cell.get("success_rate")
            row.append(f"{sr:>6.0%}" if sr is not None else "  N/A")
        print("  " + " ".join(row) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
