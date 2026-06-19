#!/usr/bin/env python3
"""Run Pilot Slice P4 — haiku × 3 env × dep_density replication.

Director Path D2 (2026-05-31):
  - 3 env (graph_nav, tool_dag, stateful_puzzle) × dep_density {1,2,4,6} × haiku × 10 task = 120 ep
  - via lab Anthropic proxy at 127.0.0.1:18801 (no $ cost)
  - Regime III backdrop, same as P1/P2 for direct cross-model comparison
  - No STOP rule; report Director per-env per-model shape table
  - Monitor proxy throughput; <5 ep/min sustained → report Director

Outputs:
  - experiments/pilot/p4_results.json (+ comparison with P1+P2 mini)
  - data/raw_logs/pilot_p4_{step,episode}.jsonl
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


P4_BACKDROP = {
    "T": 40, "state_card": 10, "branching": 4,
    "obs_noise": "clean", "mut_rate": "static",
}
P4_DEP_LEVELS = [1, 2, 4, 6]
P4_ENVS = ["graph_nav", "tool_dag", "stateful_puzzle"]
MODEL = "claude-haiku-4-5"
MEMORY_MODE = "C_struct"


def jst_now() -> str:
    return datetime.now(tz=timezone(timedelta(hours=9))).isoformat()


def build_p4_cells(n_task_per_level: int = 10, decoding_seed: int = 42) -> list[CellSpec]:
    cells: list[CellSpec] = []
    for env in P4_ENVS:
        for level in P4_DEP_LEVELS:
            stress = dict(P4_BACKDROP, dep_density=level)
            for i in range(n_task_per_level):
                env_offset = {"graph_nav": 0, "tool_dag": 200000, "stateful_puzzle": 400000}[env]
                task_seed = 500000 + env_offset + level * 10000 + i
                cells.append(CellSpec(
                    env_name=env,
                    model=MODEL,
                    stress_config=stress,
                    task_config={"archetype": "pilot_p4_haiku", "stress_config": stress},
                    task_seed=task_seed,
                    decoding_seed=decoding_seed,
                    world_regime="III_coupled_backdrop",
                    task_id=f"p4_{env}_dep{level}_t{i:02d}",
                    memory_mode=MEMORY_MODE,
                ))
    return cells


def classify_shape(rates: list[float]) -> str:
    if len(rates) < 2:
        return "insufficient_data"
    diffs = [rates[i + 1] - rates[i] for i in range(len(rates) - 1)]
    max_abs = max(abs(d) for d in diffs)
    if max_abs < 0.10:
        return "flat"
    n_neg = sum(1 for d in diffs if d < 0)
    n_pos = sum(1 for d in diffs if d > 0)
    if all(d <= 0 for d in diffs) and any(d < 0 for d in diffs):
        big = sum(1 for d in diffs if abs(d) >= 0.20)
        small = sum(1 for d in diffs if abs(d) < 0.10)
        if big == 1 and small == len(diffs) - 1:
            return "cliff"
        return "monotone"
    if n_pos > 0 and n_neg > 0:
        signs = [1 if d > 0 else -1 for d in diffs]
        if signs[0] > 0 and signs[-1] < 0:
            return "hump"
        if signs[0] < 0 and signs[-1] > 0:
            return "u_shape"
        return "irregular"
    return "irregular"


def analyze_p4(outcomes: list[EpisodeOutcome]) -> dict:
    from collections import defaultdict
    by_env_lvl = defaultdict(lambda: defaultdict(lambda: {"n": 0, "n_success": 0, "n_error": 0,
                                                          "total_steps": 0, "total_in": 0,
                                                          "total_out": 0}))
    for o in outcomes:
        env = o.cell.env_name
        lvl = int(o.cell.stress_config["dep_density"])
        d = by_env_lvl[env][lvl]
        d["n"] += 1
        if o.success: d["n_success"] += 1
        if o.error: d["n_error"] += 1
        d["total_steps"] += o.steps
        d["total_in"] += o.input_tokens
        d["total_out"] += o.output_tokens

    summary = {"per_env": {}}
    for env in P4_ENVS:
        levels = sorted(by_env_lvl.get(env, {}).keys())
        per_level = {}
        for lvl in levels:
            d = by_env_lvl[env][lvl]
            n = d["n"]
            per_level[str(lvl)] = {
                "n": n, "n_success": d["n_success"],
                "success_rate": d["n_success"] / n if n else 0.0,
                "n_error": d["n_error"],
                "mean_steps": d["total_steps"] / n if n else 0.0,
                "total_input_tokens": d["total_in"],
                "total_output_tokens": d["total_out"],
            }
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
        shape = classify_shape(rates)
        max_drop = max((d["delta_pp"] for d in adjacent_drops), default=0.0)
        summary["per_env"][env] = {
            "per_level": per_level,
            "adjacent_drops": adjacent_drops,
            "max_adjacent_delta_pp": round(max_drop, 2),
            "shape": shape,
        }
    return summary


def cross_model_sign_consistency(p4_summary: dict) -> dict:
    """Compare haiku shapes (P4) vs mini shapes (P1 + P2) per env.

    Loads mini shapes from prior result files.
    """
    out_dir = ROOT / "experiments" / "pilot"
    mini_shapes: dict[str, str] = {}
    # graph_nav: from P1 (hump per STAGE-3-009 v2)
    p1_path = out_dir / "p1_results.json"
    if p1_path.exists():
        p1 = json.loads(p1_path.read_text())
        rates = []
        for lvl in sorted(int(k) for k in p1["per_level"].keys()):
            rates.append(p1["per_level"][str(lvl)]["success_rate"])
        mini_shapes["graph_nav"] = classify_shape(rates)
    # tool_dag + stateful_puzzle: from P2
    p2_path = out_dir / "p2_results.json"
    if p2_path.exists():
        p2 = json.loads(p2_path.read_text())
        for env in ("tool_dag", "stateful_puzzle"):
            info = p2["per_env"].get(env, {})
            mini_shapes[env] = info.get("shape", "?")

    haiku_shapes = {env: info["shape"] for env, info in p4_summary["per_env"].items()}

    rows: list[dict] = []
    for env in P4_ENVS:
        mini_s = mini_shapes.get(env, "?")
        haiku_s = haiku_shapes.get(env, "?")
        # max-Δp̂ sign across model — positive = success drops with dep_density (H0-aligned)
        mini_drop = None
        haiku_drop = p4_summary["per_env"].get(env, {}).get("max_adjacent_delta_pp")
        if env == "graph_nav" and p1_path.exists():
            mini_drop = p1["max_adjacent_delta_pp"]
        elif env in ("tool_dag", "stateful_puzzle") and p2_path.exists():
            mini_drop = p2["per_env"].get(env, {}).get("max_adjacent_delta_pp")
        rows.append({
            "env": env,
            "mini_shape": mini_s,
            "haiku_shape": haiku_s,
            "mini_max_drop_pp": mini_drop,
            "haiku_max_drop_pp": haiku_drop,
            "shapes_match": mini_s == haiku_s,
        })
    # G7-style sign-consistency: same sign of max Δp̂ across both models
    n_consistent = sum(1 for r in rows if r["mini_max_drop_pp"] is not None and r["haiku_max_drop_pp"] is not None
                       and (r["mini_max_drop_pp"] * r["haiku_max_drop_pp"]) >= 0)
    return {
        "table": rows,
        "n_shapes_match": sum(1 for r in rows if r["shapes_match"]),
        "n_sign_consistent_max_drop": n_consistent,
    }


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
        slice_name="pilot_p4_haiku_cross_env_dep_density",
        emit_every=10,
    )
    cells = build_p4_cells(n_task_per_level=args.n_task)
    print(f"[pilot] === P4 START (haiku × 3 env × 4 dep_density × {args.n_task} task = {len(cells)} ep) ===")

    t_start = time.perf_counter()

    def progress(i, n, o: EpisodeOutcome):
        success = "✓" if o.success else "✗"
        tag = "OK" if o.error is None else "ERR"
        print(f"[P4 {i}/{n}] {o.cell.task_id} {success}{tag} "
              f"steps={o.steps} in={o.input_tokens} out={o.output_tokens}"
              + (f" err={o.error}" if o.error else ""))
        # Throughput monitoring
        elapsed_min = (time.perf_counter() - t_start) / 60.0
        if elapsed_min > 1.0:
            rate = i / elapsed_min
            if i % 10 == 0:
                print(f"  [throughput] {rate:.1f} ep/min so far")

    outcomes = run_pilot_slice(
        cells=cells,
        client=client,
        step_jsonl_path=log_dir / "pilot_p4_step.jsonl",
        episode_jsonl_path=log_dir / "pilot_p4_episode.jsonl",
        cost_tracker=ct,
        n_workers=args.n_workers,
        progress_fn=progress,
    )

    elapsed_s = time.perf_counter() - t_start
    summary = analyze_p4(outcomes)
    cross = cross_model_sign_consistency(summary)
    summary["meta"] = {
        "timestamp_jst": jst_now(),
        "model": MODEL,
        "memory_mode": MEMORY_MODE,
        "envs": P4_ENVS,
        "regime_backdrop": "III_coupled",
        "stress_backdrop": P4_BACKDROP,
        "dep_density_levels": P4_DEP_LEVELS,
        "n_task_per_level": args.n_task,
        "n_workers": args.n_workers,
        "wall_clock_s": round(elapsed_s, 1),
        "throughput_ep_per_min": round(len(outcomes) * 60.0 / elapsed_s, 2),
        "cost_tracker_triggered": ct.is_stopped(),
        "cost_tracker_stop_reason": ct.stop_reason(),
    }
    summary["cross_model_sign_consistency"] = cross
    (out_dir / "p4_results.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False)
    )

    print("\n[pilot] === P4 SUMMARY (haiku × 3 env × dep_density) ===")
    for env in P4_ENVS:
        info = summary["per_env"].get(env, {})
        print(f"\n  -- {env} (shape={info.get('shape','?')}) --")
        for lvl_str, lvl_info in info.get("per_level", {}).items():
            print(f"    dep_density={lvl_str}: {lvl_info['n_success']}/{lvl_info['n']} "
                  f"({lvl_info['success_rate']:.0%}) steps_mean={lvl_info['mean_steps']:.1f}")
        for d in info.get("adjacent_drops", []):
            print(f"    Δp̂(L{d['pair_lower_level']}→L{d['pair_upper_level']}) = {d['delta_pp']:+.1f}pp "
                  f"({d['success_rate_lower']:.0%} → {d['success_rate_upper']:.0%})")

    print("\n  [G7-style sign-consistency table — mini (P1+P2) vs haiku (P4)] --")
    print(f"  {'env':20s} | {'mini_shape':10s} {'haiku_shape':12s} | {'mini_max_drop':>14s} {'haiku_max_drop':>14s} | match")
    for row in cross["table"]:
        mini_d = f"{row['mini_max_drop_pp']:.1f}pp" if row['mini_max_drop_pp'] is not None else 'NA'
        haiku_d = f"{row['haiku_max_drop_pp']:.1f}pp" if row['haiku_max_drop_pp'] is not None else 'NA'
        print(f"  {row['env']:20s} | {row['mini_shape']:10s} {row['haiku_shape']:12s} | {mini_d:>14s} {haiku_d:>14s} | {row['shapes_match']}")
    print(f"  shapes match: {cross['n_shapes_match']}/{len(cross['table'])}")
    print(f"  max-Δp̂ sign-consistent: {cross['n_sign_consistent_max_drop']}/{len(cross['table'])}")

    print(f"\n[pilot] wall_clock = {elapsed_s/60.0:.1f} min, throughput = {summary['meta']['throughput_ep_per_min']:.2f} ep/min")
    print(f"[pilot] cost_tracker triggered? {ct.is_stopped()} reason='{ct.stop_reason()}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
