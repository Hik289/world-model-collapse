"""Oracle agent calls — for anchor_2 toy-trace verification + smoke tests.

The Oracle calls cheat by reading directly from the env (passed via closure or
env reference). They are NEVER used in pilot/full-grid; only for:
  (a) anchor_2 evaluator-on-gold-traces verification (Stage 2)
  (b) end-to-end logger/runner smoke tests (Stage 1 closeout)
  (c) Mode D (oracle) memory-mode pilot slice
"""

from __future__ import annotations

from typing import Callable

from .base import CallOutcome, PlannerCall, SelfDiagCall, UpdaterCall


class OraclePlanner(PlannerCall):
    """Picks a valid action from the env's current legal-action set.

    Greedy strategy:
      1. If finish action available → use it.
      2. Else pick any valid action template (deterministic by step index).
    """

    def __init__(self, env_factory: Callable[[], "object"]):
        # env_factory returns the live env instance (closure over the runner).
        self.env_factory = env_factory

    def __call__(self, observation_text, agent_world_state, history, decoding_seed):
        env = self.env_factory()
        meta = env.get_meta()
        # Find first valid candidate (deterministic walk)
        chosen = "noop"
        preconds: list[str] = []
        for cand in meta.action_templates:
            v = env.check_action_validity(cand)
            if v.valid:
                chosen = cand
                break
        return CallOutcome(
            parsed={
                "next_action": chosen,
                "required_preconditions": preconds,
                "expected_effects": [],
                "confidence": 1.0,
            },
            raw_text="",
            valid_json=True,
            retries=0,
            input_tokens=0,
            output_tokens=0,
            wallclock_ms=0,
            fallback_used=False,
        )


class OracleUpdater(UpdaterCall):
    """Mirror the env's gold world state into agent_world_state."""

    def __init__(self, env_factory: Callable[[], "object"]):
        self.env_factory = env_factory

    def __call__(
        self,
        observation_text,
        observation_partial_state,
        prev_agent_world_state,
        last_action,
        last_action_outcome,
        decoding_seed,
    ):
        env = self.env_factory()
        gold = env.get_gold_state()
        return CallOutcome(
            parsed={
                "changed_facts": [],
                "removed_facts": [],
                "full_world_state": gold,
            },
            raw_text="",
            valid_json=True,
            retries=0,
        )


class OracleSelfDiag(SelfDiagCall):
    """Agree with env validity."""

    def __init__(self, env_factory: Callable[[], "object"]):
        self.env_factory = env_factory

    def __call__(self, agent_world_state, proposed_action, required_preconditions, decoding_seed):
        env = self.env_factory()
        v = env.check_action_validity(proposed_action)
        return CallOutcome(
            parsed={
                "self_check_valid": bool(v.valid),
                "missing_preconditions": sorted(v.missing) if v.missing else [],
                "should_replan": not bool(v.valid),
            },
            raw_text="",
            valid_json=True,
            retries=0,
        )
