"""Agent 3-call loop interfaces (EXP_PLAN §3.1).

Each step the runner invokes three calls in order:
    Planner Call → World-State Update Call → Self-Diag Call → env.step(action)

Each call returns a structured JSON-like dict, with fields specified in
README §7.1-§7.4 (LOCKED). Concrete implementations (LLM-backed,
oracle-backed, fallback-noop) live in `src/agents/`.

This file defines:
  - The three abstract call interfaces.
  - The `BaseAgent` composition (one Planner + one Updater + one SelfDiag).
  - A `CallOutcome` record carrying parsed JSON + raw text + retry count
    so the runner / logger can record P0-C schema fields.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Output schemas (LOCKED keys; see EXP_PLAN §3.1).
# We do NOT enforce exact value types at this layer; the schema is used by
# the JSON-parser/validator in `src/agents/json_parser.py` (Stage 2).
# ---------------------------------------------------------------------------

PLANNER_SCHEMA_KEYS = (
    "next_action",
    "required_preconditions",
    "expected_effects",
    "confidence",
)

UPDATER_SCHEMA_KEYS = (
    "changed_facts",
    "removed_facts",
    "full_world_state",
)

SELF_DIAG_SCHEMA_KEYS = (
    "self_check_valid",
    "missing_preconditions",
    "should_replan",
)


@dataclass
class CallOutcome:
    """Result of a single LLM call (planner/updater/self-diag).

    Carries enough metadata to satisfy logging schema §3.3 (token counts,
    wallclock, retry status) and Stage-2 anchor_3 validation.
    """

    parsed: dict
    raw_text: str = ""
    valid_json: bool = True
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wallclock_ms: int = 0
    fallback_used: bool = False
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract call interfaces
# ---------------------------------------------------------------------------

class PlannerCall(ABC):
    @abstractmethod
    def __call__(
        self,
        observation_text: str,
        agent_world_state: dict,
        history: list[dict],
        decoding_seed: int,
    ) -> CallOutcome: ...


class UpdaterCall(ABC):
    @abstractmethod
    def __call__(
        self,
        observation_text: str,
        observation_partial_state: dict,
        prev_agent_world_state: dict,
        last_action: str | None,
        last_action_outcome: dict | None,
        decoding_seed: int,
    ) -> CallOutcome: ...


class SelfDiagCall(ABC):
    @abstractmethod
    def __call__(
        self,
        agent_world_state: dict,
        proposed_action: str,
        required_preconditions: list[str],
        decoding_seed: int,
    ) -> CallOutcome: ...


# ---------------------------------------------------------------------------
# BaseAgent composition
# ---------------------------------------------------------------------------

@dataclass
class BaseAgent:
    name: str
    model_id: str
    memory_mode: str  # "A_full" / "B_summary" / "C_struct" / "D_oracle"
    planner: PlannerCall
    updater: UpdaterCall
    self_diag: SelfDiagCall
