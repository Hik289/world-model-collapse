from __future__ import annotations
import json
import threading
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

def _canonical_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

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
    env_name: str = ''
    gold_world_state_before: dict = field(default_factory=dict)
    call_diagnostics: dict = field(default_factory=dict)

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
    env_name: str = ''

class JSONLWriter:

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = self.path.open('a', encoding='utf-8')

    def write_record(self, record: dict | StepLogRecord | EpisodeLogRecord) -> None:
        if isinstance(record, (StepLogRecord, EpisodeLogRecord)):
            record = asdict(record)
        line = _canonical_dumps(record)
        with self._lock:
            self._fh.write(line + '\n')
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

def read_records(path):
    import gzip
    path = Path(path)
    if not path.exists() and path.suffix != '.gz':
        path = path.with_suffix(path.suffix + '.gz')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(f'Malformed JSONL at line {number}') from None

def load_completed_traces(step_path, episode_path, exclude_empty=True):
    metadata = {}
    for record in read_records(episode_path):
        identity = (record['task_id'], record.get('model', ''), record.get('decoding_seed'))
        metadata[identity] = record
    selected = {}
    for identity, record in metadata.items():
        key = (record['task_id'], record.get('run_id'), record.get('model', ''), record.get('decoding_seed'))
        selected[key] = (identity, {})
    for record in read_records(step_path):
        key = (record['task_id'], record.get('run_id'), record.get('model', ''), record.get('decoding_seed'))
        if key not in selected:
            continue
        identity, steps = selected[key]
        index = record['step']
        if index == 0 and steps:
            steps.clear()
        steps[index] = record
    completed, excluded = ([], [])
    for identity, steps_by_index in selected.values():
        meta = metadata[identity]
        steps = [steps_by_index[i] for i in sorted(steps_by_index)]
        expected = meta.get('steps_taken', len(steps))
        reason = None
        if not steps or len(steps) != expected or [s['step'] for s in steps] != list(range(len(steps))):
            reason = 'incomplete_steps'
        elif len(steps) > meta.get('stress_config', {}).get('T', len(steps)):
            reason = 'steps_exceed_horizon'
        elif exclude_empty and any((s.get('output_tokens_this_step') == 0 for s in steps)):
            reason = 'empty_model_response'
        if reason:
            excluded.append({'task_id': meta['task_id'], 'model': meta.get('model'), 'stress_config': meta.get('stress_config', {}), 'reason': reason})
        else:
            completed.append((meta, steps))
    return (completed, excluded)
