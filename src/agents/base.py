from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
PLANNER_SCHEMA_KEYS = ('next_action', 'required_preconditions', 'expected_effects', 'confidence')
UPDATER_SCHEMA_KEYS = ('changed_facts', 'removed_facts', 'full_world_state')
SELF_DIAG_SCHEMA_KEYS = ('self_check_valid', 'missing_preconditions', 'should_replan')

@dataclass
class CallOutcome:
    parsed: dict
    raw_text: str = ''
    valid_json: bool = True
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wallclock_ms: int = 0
    fallback_used: bool = False
    extra: dict = field(default_factory=dict)

class PlannerCall(ABC):

    @abstractmethod
    def __call__(self, observation_text: str, agent_world_state: dict, history: list[dict], decoding_seed: int) -> CallOutcome:
        ...

class UpdaterCall(ABC):

    @abstractmethod
    def __call__(self, observation_text: str, observation_partial_state: dict, prev_agent_world_state: dict, last_action: str | None, last_action_outcome: dict | None, decoding_seed: int) -> CallOutcome:
        ...

class SelfDiagCall(ABC):

    @abstractmethod
    def __call__(self, agent_world_state: dict, proposed_action: str, required_preconditions: list[str], decoding_seed: int) -> CallOutcome:
        ...

@dataclass
class BaseAgent:
    name: str
    model_id: str
    memory_mode: str
    planner: PlannerCall
    updater: UpdaterCall
    self_diag: SelfDiagCall
