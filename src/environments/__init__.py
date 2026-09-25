"""StatefulPuzzle environment and shared API for worldmodelphase."""

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
from .stateful_puzzle import StatefulPuzzleEnv

ENV_REGISTRY = {
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
    "StatefulPuzzleEnv",
    "ENV_REGISTRY",
]
