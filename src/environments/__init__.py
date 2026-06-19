"""Rule-based environments for worldmodelphase."""

from .base import (
    Environment,
    Observation,
    StepResult,
    ValidityCheck,
    EnvMeta,
    canonical_json,
    canonical_hash,
    canonicalize_world_state,
    empty_world_state,
)
from .graph_nav import GraphNavEnv
from .tool_dag import ToolDAGEnv
from .stateful_puzzle import StatefulPuzzleEnv

ENV_REGISTRY = {
    "graph_nav": GraphNavEnv,
    "tool_dag": ToolDAGEnv,
    "stateful_puzzle": StatefulPuzzleEnv,
}

__all__ = [
    "Environment",
    "Observation",
    "StepResult",
    "ValidityCheck",
    "EnvMeta",
    "canonical_json",
    "canonical_hash",
    "canonicalize_world_state",
    "empty_world_state",
    "GraphNavEnv",
    "ToolDAGEnv",
    "StatefulPuzzleEnv",
    "ENV_REGISTRY",
]
