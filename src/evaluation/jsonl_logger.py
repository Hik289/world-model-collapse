"""Per-step and per-episode JSONL logging (EXP_PLAN §3.3 / §3.4).

Schemas include the P0-C patch fields:
    task_seed, decoding_seed, wallclock_ms,
    input_tokens_this_step, output_tokens_this_step

Append-only, deterministic key ordering (sort_keys=True), one JSON object per line.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _canonical_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Schemas (as dataclasses for type-safety; serialized via asdict).
# Field set mirrors EXP_PLAN §3.3 / §3.4 verbatim.
# ---------------------------------------------------------------------------

@dataclass
class StepLogRecord:
    run_id: str
    task_id: str
    task_seed: int
    decoding_seed: int
    model: str
    memory_mode: str
    world_regime: str
    stress_config: dict
    step: int
    observation: str
    agent_world_state: dict
    gold_world_state: dict
    agent_action: str
    required_preconditions_agent: list[str]
    required_preconditions_gold: list[str]
    expected_effects_agent: list[str]
    actual_effects: list[str]
    action_valid: bool
    world_state_accuracy: float
    world_consistent: bool
    dependency_correct: bool
    goal_retained: bool
    false_progress: bool
    state_staleness: bool
    self_check_valid: bool
    self_check_correct: bool
    confidence: float
    error_labels: list[str]
    wallclock_ms: int
    input_tokens_this_step: int
    output_tokens_this_step: int


@dataclass
class EpisodeLogRecord:
    run_id: str
    task_id: str
    task_seed: int
    decoding_seed: int
    model: str
    memory_mode: str
    world_regime: str
    stress_config: dict
    final_success: bool
    steps_taken: int
    collapse_detected: bool
    tau_o_primary: int | None
    tau_o_v2_strict: int | None
    tau_o_v3_continuous: int | None
    tau_w: int | None
    tau_a: int | None
    mean_world_state_accuracy: float
    mean_action_validity: float
    mean_self_check_accuracy: float
    false_progress_rate: float
    recovery_rate_5: float
    collapse_type: str | None
    fvr_pre: float | None
    fvr_post: float | None
    fvr_pre_pseudo: float | None
    fvr_post_pseudo: float | None
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float


# ---------------------------------------------------------------------------
# JSONL writers (append-only, line-buffered).
# ---------------------------------------------------------------------------

class JSONLWriter:
    """Append-only JSONL writer with canonical key ordering.

    Safe for sequential single-process use. For concurrent writers, use one
    instance per run_id (we do this by run_id-shard in the runner).
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write_record(self, record: dict | StepLogRecord | EpisodeLogRecord) -> None:
        if isinstance(record, (StepLogRecord, EpisodeLogRecord)):
            record = asdict(record)
        line = _canonical_dumps(record)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
