#!/usr/bin/env python3
"""ERR-fix smoke test — Stage-4 Prerequisite Checklist Item #2.

Run 5 haiku × tool_dag × dep=4 to verify that the runner.world_state_facts +
defensive _as_dict/_as_list patch keeps the AttributeError 'str'/'list' crashes
from re-occurring under haiku temp=0.

Use Stage-4-namespace task_seeds (600000+) to:
  (a) probe a different seed space from P4 (catch any cell-specific bug)
  (b) act as a smoke test for Item #3 (sha256 task_seeds for Stage 4 cells)
      that we will produce next.
"""

from __future__ import annotations

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


def jst_now() -> str:
    return datetime.now(tz=timezone(timedelta(hours=9))).isoformat()


def main() -> int:
    out_dir = ROOT / "experiments" / "pilot"
    log_dir = ROOT / "data" / "raw_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    backdrop = {
        "T": 40, "state_card": 10, "branching": 4,
        "obs_noise": "clean", "mut_rate": "static",
        "dep_density": 4,
    }
    cells = []
    for i in range(5):
        # 600000-base namespace = fresh seeds; ensure no overlap with P4 500000
        task_seed = 600000 + 200000 + 4 * 10000 + i  # tool_dag, dep=4, i
        cells.append(CellSpec(
            env_name="tool_dag",
            model="claude-haiku-4-5",
            stress_config=backdrop,
            task_config={"archetype": "err_fix_smoke", "stress_config": backdrop},
            task_seed=task_seed,
            decoding_seed=42,
            world_regime="III_coupled_backdrop",
            task_id=f"err_fix_smoke_tool_dag_dep4_t{i:02d}",
            memory_mode="C_struct",
        ))

    client = LLMClient()
    ct = CostTracker(
        out_path=log_dir / "cost_tracker.jsonl",
        phase="pilot",
        slice_name="err_fix_smoke_tool_dag_dep4_haiku",
        emit_every=5,
    )

    print(f"[smoke] === ERR-fix smoke (haiku × tool_dag × dep=4 × 5 ep) ===")
    t0 = time.perf_counter()

    def progress(i, n, o: EpisodeOutcome):
        success = "✓" if o.success else "✗"
        tag = "OK" if o.error is None else f"ERR({o.error[:80]})"
        print(f"[smoke {i}/{n}] {o.cell.task_id} {success}{tag} steps={o.steps}")

    outcomes = run_pilot_slice(
        cells=cells,
        client=client,
        step_jsonl_path=log_dir / "err_fix_smoke_step.jsonl",
        episode_jsonl_path=log_dir / "err_fix_smoke_episode.jsonl",
        cost_tracker=ct,
        n_workers=2,
        progress_fn=progress,
    )
    elapsed = time.perf_counter() - t0

    n_err = sum(1 for o in outcomes if o.error is not None)
    n_succ = sum(1 for o in outcomes if o.success)
    summary = {
        "timestamp_jst": jst_now(),
        "model": "claude-haiku-4-5",
        "env": "tool_dag",
        "dep_density": 4,
        "n_total": len(outcomes),
        "n_success": n_succ,
        "n_error_crashes": n_err,
        "wall_clock_s": round(elapsed, 1),
        "errors": [{"task_id": o.cell.task_id, "error": o.error}
                   for o in outcomes if o.error],
        "verdict": "PASS" if n_err == 0 else "FAIL",
    }
    (out_dir / "err_fix_smoke_results.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False)
    )
    print(f"\n[smoke] === RESULT ===")
    print(f"  n_total={len(outcomes)}  n_success={n_succ}  n_crashes={n_err}")
    print(f"  verdict: {summary['verdict']}")
    print(f"  wall_clock: {elapsed/60.0:.1f} min")
    print(f"\n  P4 baseline (haiku × tool_dag × dep=4): 6/10 crashes; this smoke must show 0/5")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
