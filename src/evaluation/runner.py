"""Episode runner: drives the 3-call agent loop + per-step JSONL logging.

EXP_PLAN §3.1 — each step:
    Planner Call → Updater Call → Self-Diag Call → env.step(action)

This runner is used for:
  (Stage 1) end-to-end smoke test with OracleAgent → produces JSONL data
  (Stage 2) LLM-backed agents
  (Stage 3+) Pilot + Full Grid

The runner does NOT compute statistical metrics (collapse onset, fvr, …);
those are computed offline by `src/evaluation/collapse_detection.py` (Stage 2)
from the per-step JSONL. The runner records the raw measurements needed
by the evaluator.

Per-step `world_state_accuracy` is computed here as continuous Jaccard between
canonicalized agent_world_state and gold_world_state fact sets. anchor_2 toy
oracle verification (Stage 2) uses the same Jaccard implementation to check
that gold-trace-as-agent yields metric == 1.0 across the board.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from ..agents.base import BaseAgent
from ..environments.base import Environment, canonicalize_world_state
from .jsonl_logger import EpisodeLogRecord, JSONLWriter, StepLogRecord


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _as_dict(x) -> dict:
    """Defensive: return x if dict-like, else {}.

    Tolerates LLM output drift (haiku temp=0 occasionally returns
    JSON top-level non-dict at sub-keys, e.g. ``"objects": "no objects"``).
    """
    return x if isinstance(x, dict) else {}


def _as_list(x) -> list:
    """Defensive: return x if list-like, else []."""
    return x if isinstance(x, list) else []


def world_state_facts(ws: dict) -> set[str]:
    """Convert a canonicalized world state into a set of fact strings.

    Continuous-Jaccard friendly: each fact contributes 1 unit to denominator.
    Fact taxonomy (deterministic, sorted, exhaustive within the schema):
      obj:<id>:type=<t>
      obj_prop:<id>:<k>=<v>
      loc:<id>:type=<t>
      loc_contains:<lid>:<oid>
      rel:<subj>:<rel>:<obj>
      inv:<oid>
      open_sg:<sg>
      done_sg:<sg>
      blocked_dep:<action>:<missing_str>
      belief:<prop>:c=<conf>

    Defensive against LLM JSON drift (e.g. haiku returning ``"objects": "no
    objects"`` or arrays where dicts are expected).  Anything off-shape is
    treated as empty and produces zero facts for that key.
    """
    facts: set[str] = set()
    if not isinstance(ws, dict):
        return facts
    for oid, meta in _as_dict(ws.get("objects")).items():
        meta_d = _as_dict(meta)
        facts.add(f"obj:{oid}:type={meta_d.get('type', '')}")
        for k, v in _as_dict(meta_d.get("props")).items():
            facts.add(f"obj_prop:{oid}:{k}={v}")
    for lid, meta in _as_dict(ws.get("locations")).items():
        meta_d = _as_dict(meta)
        facts.add(f"loc:{lid}:type={meta_d.get('type', '')}")
        for c in _as_list(meta_d.get("contents")):
            facts.add(f"loc_contains:{lid}:{c}")
    for r in _as_list(ws.get("relations")):
        r_d = _as_dict(r)
        facts.add(f"rel:{r_d.get('subj','')}:{r_d.get('rel','')}:{r_d.get('obj','')}")
    for i in _as_list(ws.get("inventory")):
        facts.add(f"inv:{i}")
    for sg in _as_list(ws.get("open_subgoals")):
        facts.add(f"open_sg:{sg}")
    for sg in _as_list(ws.get("completed_subgoals")):
        facts.add(f"done_sg:{sg}")
    for b in _as_list(ws.get("blocked_dependencies")):
        b_d = _as_dict(b)
        m = ",".join(sorted(_as_list(b_d.get("missing"))))
        facts.add(f"blocked_dep:{b_d.get('action','')}:{m}")
    for b in _as_list(ws.get("beliefs")):
        b_d = _as_dict(b)
        facts.add(f"belief:{b_d.get('prop','')}:c={b_d.get('confidence',0.0)}")
    return facts


def jaccard(agent: set[str], gold: set[str]) -> float:
    if not agent and not gold:
        return 1.0
    inter = len(agent & gold)
    union = len(agent | gold)
    return inter / union if union else 1.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class EpisodeContext:
    """Identifies an episode in the JSONL log."""
    run_id: str
    task_id: str
    task_seed: int
    decoding_seed: int
    world_regime: str
    stress_config: dict


def run_episode(
    env: Environment,
    agent: BaseAgent,
    task_config: dict,
    ctx: EpisodeContext,
    step_writer: JSONLWriter,
    episode_writer: JSONLWriter,
    max_history: int = 5,
) -> dict:
    """Run one episode through the 3-call loop.

    Returns the summary dict that was written to the episode JSONL.
    """

    obs = env.reset(task_config, seed=ctx.task_seed)
    # Initial agent world state — empty; Updater Call will populate.
    agent_world_state: dict = {
        "objects": {}, "locations": {}, "relations": [],
        "inventory": [], "open_subgoals": [], "completed_subgoals": [],
        "blocked_dependencies": [], "beliefs": [],
    }

    history: list[dict] = []
    last_action: str | None = None
    last_outcome: dict | None = None
    t = 0
    t_max = int(task_config.get("stress_config", {}).get("T", 40))

    total_in = 0
    total_out = 0

    # Trackers for episode-level summary
    wsa_list: list[float] = []
    av_list: list[bool] = []
    sca_list: list[bool] = []
    fp_count = 0
    successful = False

    while t < t_max:
        # ---- 1) Updater Call ----
        t0 = time.perf_counter()
        upd_out = agent.updater(
            observation_text=obs.text,
            observation_partial_state=obs.partial_state,
            prev_agent_world_state=agent_world_state,
            last_action=last_action,
            last_action_outcome=last_outcome,
            decoding_seed=ctx.decoding_seed,
        )
        upd_ms = int((time.perf_counter() - t0) * 1000)
        # Updater output overwrites the structured agent state.
        new_state = upd_out.parsed.get("full_world_state") or agent_world_state
        agent_world_state = canonicalize_world_state(new_state)

        # ---- 2) Planner Call ----
        t0 = time.perf_counter()
        plan_out = agent.planner(
            observation_text=obs.text,
            agent_world_state=agent_world_state,
            history=history[-max_history:],
            decoding_seed=ctx.decoding_seed,
        )
        plan_ms = int((time.perf_counter() - t0) * 1000)
        next_action: str = plan_out.parsed.get("next_action") or "noop"
        req_pre_agent: list[str] = list(plan_out.parsed.get("required_preconditions") or [])
        exp_eff_agent: list[str] = list(plan_out.parsed.get("expected_effects") or [])
        confidence: float = float(plan_out.parsed.get("confidence", 0.0))

        # ---- 3) Self-Diag Call ----
        t0 = time.perf_counter()
        sd_out = agent.self_diag(
            agent_world_state=agent_world_state,
            proposed_action=next_action,
            required_preconditions=req_pre_agent,
            decoding_seed=ctx.decoding_seed,
        )
        sd_ms = int((time.perf_counter() - t0) * 1000)
        self_check_valid = bool(sd_out.parsed.get("self_check_valid", False))

        # ---- 4) env.step ----
        gold_before = env.get_gold_state()
        validity_gold = env.check_action_validity(next_action)
        gold_required = sorted(validity_gold.missing)
        error_labels = env.compute_error_labels(agent_world_state, next_action)
        step_res = env.step(next_action)
        gold_after = env.get_gold_state()

        # ---- compute per-step metrics ----
        # agent_world_state was produced by the Updater Call BEFORE env.step,
        # so it is the agent's view of the pre-step state. We compare against
        # gold_before (same time slice) for world_state_accuracy.
        agent_facts = world_state_facts(agent_world_state)
        gold_before_facts = world_state_facts(gold_before)
        gold_facts = world_state_facts(gold_after)
        wsa = jaccard(agent_facts, gold_before_facts)
        wsa_list.append(wsa)

        action_valid = bool(validity_gold.valid)
        av_list.append(action_valid)
        self_check_correct = (self_check_valid == action_valid)
        sca_list.append(self_check_correct)

        dependency_correct = sorted(req_pre_agent) == gold_required

        # goal retention: agent's open_subgoals matches gold open_subgoals
        ag_open = sorted(agent_world_state.get("open_subgoals") or [])
        gd_open = sorted(gold_after.get("open_subgoals") or [])
        goal_retained = ag_open == gd_open

        # false progress: agent claims completed subgoal not actually completed
        ag_done = set(agent_world_state.get("completed_subgoals") or [])
        gd_done = set(gold_after.get("completed_subgoals") or [])
        false_progress = bool(ag_done - gd_done)
        if false_progress:
            fp_count += 1

        # state staleness: agent state ≠ gold_before (any fact differs at the
        # time slice the agent reasoned about)
        state_staleness = (agent_facts != gold_before_facts)

        actual_effects: list[str] = []  # filled in optionally; left empty for now
        world_consistent = not state_staleness

        # ---- log ----
        record = StepLogRecord(
            run_id=ctx.run_id,
            task_id=ctx.task_id,
            task_seed=ctx.task_seed,
            decoding_seed=ctx.decoding_seed,
            model=agent.model_id,
            memory_mode=agent.memory_mode,
            world_regime=ctx.world_regime,
            stress_config=dict(ctx.stress_config),
            step=t,
            observation=obs.text,
            agent_world_state=agent_world_state,
            gold_world_state=gold_after,
            agent_action=next_action,
            required_preconditions_agent=sorted(req_pre_agent),
            required_preconditions_gold=gold_required,
            expected_effects_agent=sorted(exp_eff_agent),
            actual_effects=actual_effects,
            action_valid=action_valid,
            world_state_accuracy=wsa,
            world_consistent=world_consistent,
            dependency_correct=dependency_correct,
            goal_retained=goal_retained,
            false_progress=false_progress,
            state_staleness=state_staleness,
            self_check_valid=self_check_valid,
            self_check_correct=self_check_correct,
            confidence=confidence,
            error_labels=sorted(error_labels),
            wallclock_ms=upd_ms + plan_ms + sd_ms,
            input_tokens_this_step=(upd_out.input_tokens + plan_out.input_tokens + sd_out.input_tokens),
            output_tokens_this_step=(upd_out.output_tokens + plan_out.output_tokens + sd_out.output_tokens),
        )
        step_writer.write_record(record)

        total_in += record.input_tokens_this_step
        total_out += record.output_tokens_this_step

        last_action = next_action
        last_outcome = {"valid": action_valid, "validity_reason": gold_before is not None}
        history.append({"step": t, "action": next_action, "valid": action_valid})

        obs = step_res.observation
        t += 1
        if step_res.done:
            successful = (not gold_after.get("open_subgoals")) and (step_res.reward > 0)
            break

    # Episode summary (collapse onset / fvr / collapse_type filled in offline by
    # evaluator scripts in Stage 2; here we emit placeholders so that the
    # episode JSONL schema is complete and consumers don't choke on missing keys).
    n = max(1, len(wsa_list))
    summary = EpisodeLogRecord(
        run_id=ctx.run_id,
        task_id=ctx.task_id,
        task_seed=ctx.task_seed,
        decoding_seed=ctx.decoding_seed,
        model=agent.model_id,
        memory_mode=agent.memory_mode,
        world_regime=ctx.world_regime,
        stress_config=dict(ctx.stress_config),
        final_success=bool(successful),
        steps_taken=t,
        collapse_detected=False,           # filled by collapse_detection (Stage 2)
        tau_o_primary=None,
        tau_o_v2_strict=None,
        tau_o_v3_continuous=None,
        tau_w=None,
        tau_a=None,
        mean_world_state_accuracy=sum(wsa_list) / n,
        mean_action_validity=sum(1.0 for v in av_list if v) / n,
        mean_self_check_accuracy=sum(1.0 for v in sca_list if v) / n,
        false_progress_rate=fp_count / n,
        recovery_rate_5=0.0,               # filled by collapse_detection (Stage 2)
        collapse_type=None,
        fvr_pre=None,
        fvr_post=None,
        fvr_pre_pseudo=None,
        fvr_post_pseudo=None,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        total_cost_usd=0.0,
    )
    episode_writer.write_record(summary)
    return {"summary": summary, "steps": t, "success": successful}


def new_run_id() -> str:
    return str(uuid.uuid4())
