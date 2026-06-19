"""Shared helpers for Exp B (Mode A) + Exp C1/C2 (cross-model) runners.

All three experiments use the same generate_cells/run/finalize skeleton; only
the model + memory_mode + backdrop grid + seed namespace differ.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.llm_client import LLMClient  # noqa: E402
from src.runner import (  # noqa: E402
    CellSpec, CostTracker, EpisodeOutcome, run_pilot_slice,
)

LOG_DIR = ROOT / "data" / "raw_logs"
CROSS_HARNESS_ROOT = ROOT / "experiments" / "cross_harness"


def jst_now() -> str:
    return datetime.now(tz=timezone(timedelta(hours=9))).isoformat()


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Experiment spec
# ---------------------------------------------------------------------------

@dataclass
class CrossHarnessSpec:
    name: str                       # e.g. "B_mode_a", "C1_llama3", "C2_gpt4o"
    model: str
    memory_mode: str                # "A_free" | "C_struct"
    world_regime: str               # cost_tracker regime tag
    seed_namespace_base: int        # task_seed = base + 1000*cell_idx + task_idx
    grid_cells: list[dict]          # each: {"state_card": int, "dep_density": int}
    n_per_cell: int = 25
    decoding_seed: int = 42
    T: int = 40
    branching: int = 4
    obs_noise: str = "clean"
    mut_rate: str = "static"
    env_name: str = "stateful_puzzle"

    @property
    def subdir(self) -> Path:
        return CROSS_HARNESS_ROOT / self.name

    @property
    def seeds_path(self) -> Path:
        return self.subdir / "task_seeds.json"

    @property
    def completed_path(self) -> Path:
        return self.subdir / "completed_cells.json"

    @property
    def results_path(self) -> Path:
        return CROSS_HARNESS_ROOT / f"{self.name}_results.json"

    @property
    def slice_name(self) -> str:
        return f"cross_harness_{self.name}"


# ---------------------------------------------------------------------------
# Seed generation
# ---------------------------------------------------------------------------

def generate_seeds(spec: CrossHarnessSpec) -> dict:
    spec.subdir.mkdir(parents=True, exist_ok=True)
    cells = []
    for cell_idx, cell in enumerate(spec.grid_cells):
        sc = int(cell["state_card"])
        dd = int(cell["dep_density"])
        stress = {
            "T": spec.T,
            "state_card": sc,
            "branching": spec.branching,
            "dep_density": dd,
            "obs_noise": spec.obs_noise,
            "mut_rate": spec.mut_rate,
        }
        for task_idx in range(spec.n_per_cell):
            seed = spec.seed_namespace_base + 1000 * cell_idx + task_idx
            task_id = (
                f"crossharness_{spec.name}_sc{sc:02d}_dd{dd}_t{task_idx:03d}"
            )
            cells.append({
                "cell_key": (
                    f"crossharness|exp={spec.name}|env={spec.env_name}"
                    f"|sc={sc}|dd={dd}|task_index={task_idx}"
                ),
                "env": spec.env_name,
                "model": spec.model,
                "state_card": sc,
                "dep_density": dd,
                "task_index": task_idx,
                "task_seed": seed,
                "decoding_seed": spec.decoding_seed,
                "world_regime": spec.world_regime,
                "stress_config": stress,
                "task_id": task_id,
                "task_config": {
                    "archetype": "cross_harness",
                    "stress_config": stress,
                },
                "memory_mode": spec.memory_mode,
            })
    out = {
        "generated_jst": jst_now(),
        "schema_version": "cross_harness_v1",
        "exp": {
            "name": spec.name,
            "model": spec.model,
            "memory_mode": spec.memory_mode,
            "world_regime": spec.world_regime,
            "seed_namespace_base": spec.seed_namespace_base,
            "env": spec.env_name,
            "T": spec.T,
            "branching": spec.branching,
            "obs_noise": spec.obs_noise,
            "mut_rate": spec.mut_rate,
            "grid_cells": spec.grid_cells,
            "n_per_cell": spec.n_per_cell,
        },
        "total_cells": len(cells),
        "decoding_seed": spec.decoding_seed,
        "seed_generator": "seed_namespace_base + 1000*cell_idx + task_idx",
        "cells": cells,
    }
    spec.seeds_path.write_text(
        json.dumps(out, sort_keys=True, indent=2, ensure_ascii=False)
    )
    return out


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


def load_completed(spec: CrossHarnessSpec) -> set[str]:
    if not spec.completed_path.exists():
        return set()
    try:
        return set(json.loads(spec.completed_path.read_text()).get("task_ids", []))
    except Exception:
        return set()


def save_completed(spec: CrossHarnessSpec, task_ids: set[str]) -> None:
    payload = {
        "schema_version": "cross_harness_completed_v1",
        "exp": spec.name,
        "updated_jst": jst_now(),
        "n_completed": len(task_ids),
        "task_ids": sorted(task_ids),
    }
    atomic_write_json(spec.completed_path, payload)


def analyze(spec: CrossHarnessSpec, outcomes: list[EpisodeOutcome]) -> dict:
    by_cell = defaultdict(lambda: {
        "n": 0, "n_success": 0, "n_error": 0,
        "total_steps": 0, "total_in": 0, "total_out": 0, "total_cost": 0.0,
    })
    for o in outcomes:
        sc = int(o.cell.stress_config["state_card"])
        dd = int(o.cell.stress_config["dep_density"])
        key = f"sc={sc},dd={dd}"
        d = by_cell[key]
        d["n"] += 1
        if o.success:
            d["n_success"] += 1
        if o.error:
            d["n_error"] += 1
        d["total_steps"] += o.steps
        d["total_in"] += o.input_tokens
        d["total_out"] += o.output_tokens
        d["total_cost"] += o.cost_usd

    per_cell = {}
    for key, d in by_cell.items():
        n = d["n"]
        per_cell[key] = {
            "n": n,
            "n_success": d["n_success"],
            "success_rate": d["n_success"] / n if n else 0.0,
            "n_error": d["n_error"],
            "mean_steps": d["total_steps"] / n if n else 0.0,
            "total_input_tokens": d["total_in"],
            "total_output_tokens": d["total_out"],
            "total_cost_usd": round(d["total_cost"], 6),
        }
    overall_n = sum(d["n"] for d in by_cell.values())
    overall_success = sum(d["n_success"] for d in by_cell.values())
    overall_err = sum(d["n_error"] for d in by_cell.values())
    overall_cost = sum(d["total_cost"] for d in by_cell.values())
    return {
        "per_cell": per_cell,
        "overall": {
            "n": overall_n,
            "n_success": overall_success,
            "success_rate": overall_success / overall_n if overall_n else 0.0,
            "n_error": overall_err,
            "total_cost_usd": round(overall_cost, 6),
        },
    }


def run_cross_harness(spec: CrossHarnessSpec) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=2)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--cost-cap", type=float, default=5.0,
                    help="Hard $ cap for runner (default $5 — override per exp)")
    ap.add_argument("--regenerate-seeds", action="store_true")
    ap.add_argument("--smoke", type=int, default=0,
                    help="If >0, run only this many cells (single cell sampled).")
    args = ap.parse_args()

    spec.subdir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    tag = f"crossharness_{spec.name}"
    print(f"[{tag}] === Cross-harness exp '{spec.name}' dispatch ===")
    print(f"[{tag}] start: {jst_now()}  model={spec.model}  memory_mode={spec.memory_mode}")
    print(f"[{tag}] grid_cells={spec.grid_cells}  n_per_cell={spec.n_per_cell}  cost_cap=${args.cost_cap}")

    if args.regenerate_seeds or not spec.seeds_path.exists():
        generate_seeds(spec)
        print(f"[{tag}] generated seeds → {spec.seeds_path}")

    grid = json.loads(spec.seeds_path.read_text())
    all_cells = [cell_from_dict(c) for c in grid["cells"]]

    if args.smoke > 0:
        # Take the first `smoke` cells from the first grid cell so we hit
        # the collapse backdrop, not a corner.
        cells = all_cells[: args.smoke]
        print(f"[{tag}] SMOKE: running first {len(cells)} cells, skipping checkpoint resume")
        completed = set()
    else:
        completed = load_completed(spec)
        if completed:
            cells = [c for c in all_cells if c.task_id not in completed]
            print(f"[{tag}] resuming: {len(completed)} done, {len(cells)} remaining")
        else:
            cells = all_cells
            print(f"[{tag}] fresh start: {len(cells)} cells")

    if not cells:
        return _finalize(spec, all_cells, [])

    client = LLMClient()
    ct = CostTracker(
        out_path=LOG_DIR / "cost_tracker.jsonl",
        phase=f"cross_harness_{spec.name}",
        slice_name=spec.slice_name,
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
            tag2 = "OK" if o.error is None else f"ERR({o.error[:50]})"
            print(f"[{tag} {i}/{n}] {o.cell.task_id} {success}{tag2} "
                  f"steps={o.steps} cost=${o.cost_usd:.4f} cum=${cum:.2f} "
                  f"| {elapsed:.1f}min rate={rate:.2f}ep/min")
            if (i % args.checkpoint_every == 0 or i == n) and args.smoke == 0:
                save_completed(spec, state["completed"])
                print(f"[{tag}]   checkpoint: {len(state['completed'])}/{len(all_cells)} cells done; "
                      f"cum_cost=${cum:.2f}")
            if cum > args.cost_cap and not state["abort"]:
                state["abort"] = True
                print(f"[CROSSHARNESS_{spec.name.upper()}_ABORT_COST_CAP] "
                      f"cum=${cum:.2f} > cap=${args.cost_cap:.2f}; aborting")
                if args.smoke == 0:
                    save_completed(spec, state["completed"])
                ct.force_stop(f"cost_cap_exceeded:${cum:.2f}>${args.cost_cap}")

    print(f"[{tag}] launching run_pilot_slice (bounded-wave, n_workers={args.n_workers}) ...")
    step_path = LOG_DIR / f"cross_harness_{spec.name}_step.jsonl"
    ep_path = LOG_DIR / f"cross_harness_{spec.name}_episode.jsonl"
    run_pilot_slice(
        cells=cells,
        client=client,
        step_jsonl_path=step_path,
        episode_jsonl_path=ep_path,
        cost_tracker=ct,
        n_workers=args.n_workers,
        progress_fn=progress,
    )
    return _finalize(spec, all_cells, state["outcomes"],
                     cost_cap_hit=state["abort"],
                     final_cum_cost=state["cumcost"],
                     is_smoke=args.smoke > 0)


def _finalize(spec: CrossHarnessSpec, all_cells: list, outcomes: list,
              cost_cap_hit: bool = False, final_cum_cost: float = 0.0,
              is_smoke: bool = False) -> int:
    seen: dict[str, EpisodeOutcome] = {}
    for o in outcomes:
        seen[o.cell.task_id] = o
    uniq = list(seen.values())

    summary = analyze(spec, uniq)
    summary["meta"] = {
        "timestamp_jst": jst_now(),
        "slice": spec.slice_name,
        "exp_name": spec.name,
        "model": spec.model,
        "memory_mode": spec.memory_mode,
        "world_regime": spec.world_regime,
        "env": spec.env_name,
        "grid_cells": spec.grid_cells,
        "n_per_cell": spec.n_per_cell,
        "total_cells_planned": len(all_cells),
        "n_completed": len(uniq),
        "cost_cap_hit": cost_cap_hit,
        "final_cum_cost_usd": round(final_cum_cost, 4),
        "is_smoke": is_smoke,
    }

    if is_smoke:
        out_path = spec.subdir / "smoke_results.json"
    else:
        out_path = spec.results_path
    atomic_write_json(out_path, summary)
    if not is_smoke:
        save_completed(spec, {o.cell.task_id for o in uniq})

    tag = f"crossharness_{spec.name}"
    print(f"\n[{tag}] === DONE === wrote {out_path}")
    print(f"[{tag}] completed {len(uniq)}/{len(all_cells)}  cost=${final_cum_cost:.4f}")
    print(f"\n  per-cell success_rate:")
    for cell in spec.grid_cells:
        sc, dd = int(cell["state_card"]), int(cell["dep_density"])
        key = f"sc={sc},dd={dd}"
        c = summary["per_cell"].get(key, {})
        sr = c.get("success_rate")
        srs = f"{sr:>6.0%}" if sr is not None else "  N/A"
        n = c.get("n", 0)
        print(f"    sc={sc:<2} dd={dd}  sr={srs}  n={n}  err={c.get('n_error', 0)}")
    print(f"\n  overall: success_rate={summary['overall']['success_rate']:.1%} "
          f"n={summary['overall']['n']} err={summary['overall']['n_error']}")
    return 0
