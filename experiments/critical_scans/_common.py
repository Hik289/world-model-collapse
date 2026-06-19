"""Critical-scan fine-grained sweep runners — shared infrastructure.

Stage 5 B follow-up (2026-06-06): zoom-in around phase transitions identified
by Stage 4 G1-trigger grid + Stage 5 B T ablation:

  - Exp-SC-Fine: state_card ∈ {11..19} step=1, dd=1, T=40, branching=4,
                 obs=clean, mut=static; sc★ ≈ 50% crossover.
                 Stage 4 grid: sc=10 (100%) → sc=20 (0%), critical in 10-20.
  - Exp-T-Fine:  T ∈ {22,25,28,30,32,35,38,42,48,55,65}, sc=10, dd=6,
                 branching=4, obs=clean, mut=static; T★ ≈ 50% crossover.
                 T ablation: T=20 (0%) → T=40 (12%) → T=80 (32%), critical in 20-40.

Each sweep:
  - env: stateful_puzzle
  - model: claude-haiku-4-5 via lab proxy (127.0.0.1:18801) → $0
  - n=50 ep per level
  - decoding_seed = 42 (locked)
  - cost_cap = $5 hard cap (defensive only; lab proxy actual = $0)

Task seed namespaces (distinct from Stage 4/5 sha256 + Stage 5 B 900k-base):
  - Exp-SC-Fine: 1_100_000 + 10_000 * level_idx + task_idx
  - Exp-T-Fine:  1_200_000 + 10_000 * level_idx + task_idx
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
SCAN_ROOT = ROOT / "experiments" / "critical_scans"


def jst_now() -> str:
    return datetime.now(tz=timezone(timedelta(hours=9))).isoformat()


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Scan config
# ---------------------------------------------------------------------------

@dataclass
class ScanSpec:
    name: str                    # short tag, e.g. "sc_fine"
    axis_key: str                # stress_config key being varied
    levels: list                 # ordered values
    backdrop: dict               # fixed stress_config values
    seed_base: int               # task_seed = seed_base + 10000*level_idx + task_idx
    world_regime: str            # tag for events / logs (must be in cost_tracker dict)
    n_per_level: int = 50
    decoding_seed: int = 42

    @property
    def subdir(self) -> Path:
        return SCAN_ROOT / self.name

    @property
    def seeds_path(self) -> Path:
        return self.subdir / "task_seeds.json"

    @property
    def completed_path(self) -> Path:
        return self.subdir / "completed_cells.json"

    @property
    def results_path(self) -> Path:
        return self.subdir / "results.json"

    @property
    def slice_name(self) -> str:
        return f"critical_scan_{self.name}_sp_haiku"


# ---------------------------------------------------------------------------
# Seed grid generation
# ---------------------------------------------------------------------------

def _level_token(value) -> str:
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def generate_seeds(spec: ScanSpec) -> dict:
    spec.subdir.mkdir(parents=True, exist_ok=True)
    cells = []
    for level_idx, level in enumerate(spec.levels):
        for task_idx in range(spec.n_per_level):
            seed = spec.seed_base + 10_000 * level_idx + task_idx
            stress = dict(spec.backdrop)
            stress[spec.axis_key] = level
            level_tok = _level_token(level)
            task_id = f"cs_{spec.name}_{level_tok}_t{task_idx:03d}"
            cells.append({
                "cell_key": (
                    f"critical_scan|axis={spec.name}|level={level_tok}"
                    f"|task_index={task_idx}"
                ),
                "env": "stateful_puzzle",
                "model": "claude-haiku-4-5",
                "axis": spec.name,
                "axis_level": level,
                "level_idx": level_idx,
                "task_index": task_idx,
                "task_seed": seed,
                "decoding_seed": spec.decoding_seed,
                "world_regime": spec.world_regime,
                "stress_config": stress,
                "task_id": task_id,
                "task_config": {
                    "archetype": "critical_scan",
                    "stress_config": stress,
                },
                "memory_mode": "C_struct",
            })
    out = {
        "generated_jst": jst_now(),
        "schema_version": "critical_scan_v1",
        "scan": {
            "name": spec.name,
            "axis_key": spec.axis_key,
            "levels": spec.levels,
            "world_regime": spec.world_regime,
            "seed_base": spec.seed_base,
        },
        "backdrop": spec.backdrop,
        "n_per_level": spec.n_per_level,
        "total_cells": len(cells),
        "decoding_seed": spec.decoding_seed,
        "seed_generator": f"{spec.seed_base} + 10000*level_idx + task_idx",
        "cells": cells,
    }
    spec.seeds_path.write_text(
        json.dumps(out, sort_keys=True, indent=2, ensure_ascii=False))
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

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


def load_completed(spec: ScanSpec) -> set[str]:
    if not spec.completed_path.exists():
        return set()
    try:
        return set(json.loads(spec.completed_path.read_text()).get("task_ids", []))
    except Exception:
        return set()


def save_completed(spec: ScanSpec, task_ids: set[str]) -> None:
    payload = {
        "schema_version": "critical_scan_completed_v1",
        "scan": spec.name,
        "updated_jst": jst_now(),
        "n_completed": len(task_ids),
        "task_ids": sorted(task_ids),
    }
    atomic_write_json(spec.completed_path, payload)


def analyze(spec: ScanSpec, outcomes: list[EpisodeOutcome]) -> dict:
    by_level = defaultdict(lambda: {
        "n": 0, "n_success": 0, "n_error": 0,
        "total_steps": 0, "total_in": 0, "total_out": 0,
        "total_cost": 0.0,
    })
    for o in outcomes:
        level = o.cell.stress_config.get(spec.axis_key)
        key = f"{spec.axis_key}={level}"
        d = by_level[key]
        d["n"] += 1
        if o.success: d["n_success"] += 1
        if o.error: d["n_error"] += 1
        d["total_steps"] += o.steps
        d["total_in"] += o.input_tokens
        d["total_out"] += o.output_tokens
        d["total_cost"] += o.cost_usd
    out = {"per_level": {}}
    for key, d in by_level.items():
        n = d["n"]
        out["per_level"][key] = {
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


def run_scan(spec: ScanSpec) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=2)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--cost-cap", type=float, default=5.0)
    ap.add_argument("--regenerate-seeds", action="store_true")
    args = ap.parse_args()

    spec.subdir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    tag = f"crit_scan_{spec.name}"
    print(f"[{tag}] === Critical scan '{spec.name}' dispatch ===")
    print(f"[{tag}] start: {jst_now()}  axis={spec.axis_key} levels={spec.levels}")
    print(f"[{tag}] backdrop={spec.backdrop}")
    print(f"[{tag}] model=claude-haiku-4-5 via lab proxy   cost_cap=${args.cost_cap}")

    if args.regenerate_seeds or not spec.seeds_path.exists():
        generate_seeds(spec)
        print(f"[{tag}] generated seeds → {spec.seeds_path}")

    grid = json.loads(spec.seeds_path.read_text())
    all_cells = [cell_from_dict(c) for c in grid["cells"]]
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
        phase=f"critical_scan_{spec.name}",
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
            if i % args.checkpoint_every == 0 or i == n:
                save_completed(spec, state["completed"])
                print(f"[{tag}]   checkpoint: {len(state['completed'])}/{len(all_cells)} cells done; cum_cost=${cum:.2f}")
            if cum > args.cost_cap and not state["abort"]:
                state["abort"] = True
                print(f"[CRIT_SCAN_{spec.name.upper()}_ABORT_COST_CAP] cum=${cum:.2f} > cap=${args.cost_cap:.2f}; aborting")
                save_completed(spec, state["completed"])
                ct.force_stop(f"cost_cap_exceeded:${cum:.2f}>${args.cost_cap}")

    print(f"[{tag}] launching run_pilot_slice (bounded-wave, n_workers={args.n_workers}) ...")
    step_path = LOG_DIR / f"critical_scan_{spec.name}_step.jsonl"
    ep_path = LOG_DIR / f"critical_scan_{spec.name}_episode.jsonl"
    run_pilot_slice(
        cells=cells,
        client=client,
        step_jsonl_path=step_path,
        episode_jsonl_path=ep_path,
        cost_tracker=ct,
        n_workers=args.n_workers,
        progress_fn=progress,
    )
    return _finalize(spec, all_cells, state["outcomes"], cost_cap_hit=state["abort"],
                     final_cum_cost=state["cumcost"])


def _finalize(spec: ScanSpec, all_cells: list, outcomes: list,
              cost_cap_hit: bool = False, final_cum_cost: float = 0.0) -> int:
    seen: dict[str, EpisodeOutcome] = {}
    for o in outcomes:
        seen[o.cell.task_id] = o
    uniq = list(seen.values())

    summary = analyze(spec, uniq)
    summary["meta"] = {
        "timestamp_jst": jst_now(),
        "slice": spec.slice_name,
        "scan_name": spec.name,
        "scan_axis": spec.axis_key,
        "levels": spec.levels,
        "world_regime": spec.world_regime,
        "backdrop": spec.backdrop,
        "model": "claude-haiku-4-5",
        "env": "stateful_puzzle",
        "n_per_level": spec.n_per_level,
        "total_cells_planned": len(all_cells),
        "n_completed": len(uniq),
        "cost_cap_hit": cost_cap_hit,
        "final_cum_cost_usd": round(final_cum_cost, 4),
    }
    atomic_write_json(spec.results_path, summary)
    save_completed(spec, {o.cell.task_id for o in uniq})

    tag = f"crit_scan_{spec.name}"
    print(f"\n[{tag}] === DONE === wrote {spec.results_path}")
    print(f"[{tag}] completed {len(uniq)}/{len(all_cells)}  cost=${final_cum_cost:.4f}")
    print(f"\n  per-level success_rate ({spec.axis_key}):")
    for level in spec.levels:
        key = f"{spec.axis_key}={level}"
        cell = summary["per_level"].get(key, {})
        sr = cell.get("success_rate")
        srs = f"{sr:>6.0%}" if sr is not None else "  N/A"
        n = cell.get("n", 0)
        print(f"    {key:<24} sr={srs}  n={n}")
    return 0
