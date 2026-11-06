from __future__ import annotations
from typing import Callable
from .base import CallOutcome, PlannerCall, SelfDiagCall, UpdaterCall

class OraclePlanner(PlannerCall):

    def __init__(self, env_factory: Callable[[], 'object']):
        self.env_factory = env_factory

    def __call__(self, observation_text, agent_world_state, history, decoding_seed):
        env = self.env_factory()
        meta = env.get_meta()
        chosen = 'noop'
        preconds: list[str] = []
        for cand in meta.action_templates:
            v = env.check_action_validity(cand)
            if v.valid:
                chosen = cand
                break
        return CallOutcome(parsed={'next_action': chosen, 'required_preconditions': preconds, 'expected_effects': [], 'confidence': 1.0}, raw_text='', valid_json=True, retries=0, input_tokens=0, output_tokens=0, wallclock_ms=0, fallback_used=False)

class OracleUpdater(UpdaterCall):

    def __init__(self, env_factory: Callable[[], 'object']):
        self.env_factory = env_factory

    def __call__(self, observation_text, observation_partial_state, prev_agent_world_state, last_action, last_action_outcome, decoding_seed):
        env = self.env_factory()
        gold = env.get_gold_state()
        return CallOutcome(parsed={'changed_facts': [], 'removed_facts': [], 'full_world_state': gold}, raw_text='', valid_json=True, retries=0)

class OracleSelfDiag(SelfDiagCall):

    def __init__(self, env_factory: Callable[[], 'object']):
        self.env_factory = env_factory

    def __call__(self, agent_world_state, proposed_action, required_preconditions, decoding_seed):
        env = self.env_factory()
        v = env.check_action_validity(proposed_action)
        return CallOutcome(parsed={'self_check_valid': bool(v.valid), 'missing_preconditions': sorted(v.missing) if v.missing else [], 'should_replan': not bool(v.valid)}, raw_text='', valid_json=True, retries=0)
