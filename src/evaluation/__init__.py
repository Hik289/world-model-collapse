"""Evaluation layer — JSONL logging + episode runner.

Stage-2 modules (collapse_detection.py, collapse_type_classifier.py,
step_metrics.py) will land here next.
"""

from .jsonl_logger import (
    EpisodeLogRecord,
    JSONLWriter,
    StepLogRecord,
)
from .runner import (
    EpisodeContext,
    jaccard,
    new_run_id,
    run_episode,
    world_state_facts,
)

__all__ = [
    "EpisodeLogRecord",
    "JSONLWriter",
    "StepLogRecord",
    "EpisodeContext",
    "jaccard",
    "new_run_id",
    "run_episode",
    "world_state_facts",
]
