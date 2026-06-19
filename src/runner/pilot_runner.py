"""Pilot runner — drives Pilot Slice P0 (Regime I baseline) and P1 (anchor_5).

Approach:
  - For each (env, model, stress_config, task_seed, decoding_seed) cell, run
    a single episode via `evaluation.run_episode` with the LLM-backed agent.
  - Episode-level parallelism via ThreadPoolExecutor (I/O bound, GIL-friendly
    since 99% of time is API I/O wait).
  - Each finished episode → cost_tracker.record_episode + accumulate result.
  - If cost_tracker triggers stop → cancel remaining futures gracefully and
    write whatever was completed.

Concurrency safety:
  - Each episode creates its own env + agent instance (no shared state).
  - JSONLWriter is shared but guarded by file-append atomicity (OS-level
    append is single-syscall on Linux for short lines; we also `f.flush()`).
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable

from ..agents.llm_agent import build_llm_agent
from ..agents.llm_client import LLMClient
from ..environments import ENV_REGISTRY
from ..evaluation import EpisodeContext, JSONLWriter, new_run_id, run_episode
from .cost_tracker import CostTracker, EpisodeRecord, estimate_cost_usd


# ---------------------------------------------------------------------------
# Cell specification
# ---------------------------------------------------------------------------

@dataclass
class CellSpec:
    """One episode cell to run."""
    env_name: str
    model: str
    stress_config: dict
    task_config: dict
    task_seed: int
    decoding_seed: int
    world_regime: str
    task_id: str
    memory_mode: str = "C_struct"


@dataclass
class EpisodeOutcome:
    cell: CellSpec
    success: bool
    steps: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    json_retries_total: int
    json_calls_total: int
    error: str | None = None


# ---------------------------------------------------------------------------
# Per-episode worker
# ---------------------------------------------------------------------------

_jsonl_lock = threading.Lock()


def _safe_write(writer: JSONLWriter, record):
    """Append a JSON line in a thread-safe way."""
    with _jsonl_lock:
        writer.write_record(record)


def _run_one_episode(
    cell: CellSpec,
    client: LLMClient,
    step_writer: JSONLWriter,
    epi_writer: JSONLWriter,
) -> EpisodeOutcome:
    """Run a single episode. All exceptions are caught and returned as failure."""
    try:
        env_cls = ENV_REGISTRY[cell.env_name]
        env = env_cls()
        # Build agent: action_templates depend on a (probably pre-reset) env.
        # We reset once internally to populate get_meta, then use those templates.
        env_for_meta = env_cls()
        env_for_meta.reset(cell.task_config, seed=cell.task_seed)
        action_templates = env_for_meta.get_meta().action_templates
        agent = build_llm_agent(client, cell.model, action_templates, memory_mode=cell.memory_mode)

        # Build a thread-safe wrapper writer per episode (the safe_write
        # function above is thread-safe; we use the global JSONLWriter directly).
        class _SafeWriter:
            def __init__(self, w): self.w = w
            def write_record(self, rec):
                _safe_write(self.w, rec)

        sw_safe = _SafeWriter(step_writer)
        ew_safe = _SafeWriter(epi_writer)

        ctx = EpisodeContext(
            run_id=new_run_id(),
            task_id=cell.task_id,
            task_seed=cell.task_seed,
            decoding_seed=cell.decoding_seed,
            world_regime=cell.world_regime,
            stress_config=cell.stress_config,
        )
        # Hook into runner to gather per-step retry stats: easiest is to
        # subclass behaviour but for now we recompute from logged JSONL post-hoc.
        # We track tokens via the runner's CallOutcome aggregation (already
        # in the episode summary).
        result = run_episode(env, agent, cell.task_config, ctx, sw_safe, ew_safe)
        summary = result["summary"]

        cost = estimate_cost_usd(cell.model, summary.total_input_tokens, summary.total_output_tokens)
        return EpisodeOutcome(
            cell=cell,
            success=bool(summary.final_success),
            steps=summary.steps_taken,
            input_tokens=summary.total_input_tokens,
            output_tokens=summary.total_output_tokens,
            cost_usd=cost,
            json_retries_total=0,   # populated post-hoc from step JSONL if needed
            json_calls_total=summary.steps_taken * 3,  # 3 calls per step
        )
    except Exception as e:
        return EpisodeOutcome(
            cell=cell,
            success=False,
            steps=0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            json_retries_total=0,
            json_calls_total=0,
            error=f"{type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------------
# Runner entry point
# ---------------------------------------------------------------------------

def run_pilot_slice(
    cells: list[CellSpec],
    client: LLMClient,
    step_jsonl_path: Path,
    episode_jsonl_path: Path,
    cost_tracker: CostTracker,
    n_workers: int = 4,
    progress_fn: Callable[[int, int, EpisodeOutcome], None] | None = None,
) -> list[EpisodeOutcome]:
    """Run a list of cells with bounded-wave concurrency.

    Submits at most ``n_workers * IN_FLIGHT_MULTIPLIER`` futures at a time so
    that ``cost_tracker.is_stopped()`` can effectively halt dispatch within
    one wave (~``n_workers`` extra episodes) instead of executing all queued
    cells regardless of triggers (STAGE-3-017 submit-all-futures flaw fix).

    Returns a list of EpisodeOutcome in completion order. On stop trigger,
    drains in-flight futures and stops submitting new cells; cancels queued
    cells that have not been submitted yet.
    """
    IN_FLIGHT_MULTIPLIER = 4  # max ``n_workers * 4`` futures in flight

    outcomes: list[EpisodeOutcome] = []
    n_total_planned = len(cells)
    cell_iter = iter(cells)
    in_flight: set = set()
    submitted_count = 0
    completed_count = 0
    stop_dispatch = False

    def _try_submit(pool, sw, ew, max_in_flight: int) -> int:
        nonlocal submitted_count
        added = 0
        while len(in_flight) < max_in_flight and not stop_dispatch:
            try:
                cell = next(cell_iter)
            except StopIteration:
                return added
            fut = pool.submit(_run_one_episode, cell, client, sw, ew)
            in_flight.add(fut)
            submitted_count += 1
            added += 1
        return added

    with JSONLWriter(step_jsonl_path) as sw, JSONLWriter(episode_jsonl_path) as ew:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            # Prime the in-flight set
            _try_submit(pool, sw, ew, n_workers * IN_FLIGHT_MULTIPLIER)

            while in_flight:
                # Wait for at least one to complete
                done_set, _pending = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done_set:
                    in_flight.remove(fut)
                    outcome = fut.result()
                    outcomes.append(outcome)
                    completed_count += 1
                    rec = EpisodeRecord(
                        model=outcome.cell.model,
                        input_tokens=outcome.input_tokens,
                        output_tokens=outcome.output_tokens,
                        cost_usd=outcome.cost_usd,
                        json_retries_total=outcome.json_retries_total,
                        json_calls_total=outcome.json_calls_total,
                        n_steps=outcome.steps,
                        success=outcome.success,
                        rerun=False,
                        world_regime=outcome.cell.world_regime,
                    )
                    cost_tracker.record_episode(rec)
                    if progress_fn:
                        progress_fn(completed_count, n_total_planned, outcome)
                # Re-evaluate stop signal each wave
                if cost_tracker.is_stopped() and not stop_dispatch:
                    stop_dispatch = True
                    # Do not submit more; let in-flight drain
                # Submit more if still under cap and not stopped
                if not stop_dispatch:
                    _try_submit(pool, sw, ew, n_workers * IN_FLIGHT_MULTIPLIER)
    cost_tracker.finalize()
    return outcomes
