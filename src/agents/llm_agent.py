"""LLM-backed Agent composition (used in Pilot + Full Grid).

Wires the Planner / Updater / SelfDiag call interfaces to `LLMClient.call_typed`
with retry + JSON validation. Stateless per call; the runner holds context.

The agent itself does NOT carry conversation history — `BaseAgent` is just a
container. The runner passes observation + agent_world_state on each step.
"""

from __future__ import annotations

import re

from .base import BaseAgent, CallOutcome, PlannerCall, SelfDiagCall, UpdaterCall
from .llm_client import LLMClient
from .prompts import planner_user, self_diag_user, updater_user
from .prompts import PLANNER_SYS, SELF_DIAG_SYS, UPDATER_SYS
from .prompts import (
    MODE_A_PLANNER_SYS, MODE_A_SELF_DIAG_SYS, MODE_A_UPDATER_SYS,
    planner_user_mode_a, self_diag_user_mode_a, updater_user_mode_a,
)


# Token defaults for Mode A free-form (slightly larger than Mode C planner /
# self_diag because we now have prose, but smaller for updater because there's
# no full_world_state JSON).
MODE_A_MAX_TOKENS = {
    "planner": 400,
    "updater": 500,
    "self_diag": 250,
}


# Mode A natural-language memory is carried inside the agent_world_state dict
# under this key. The runner already passes the dict between steps unchanged
# (see run_episode), so this is the cleanest piggy-back.
NL_MEMORY_KEY = "_natural_understanding"


# ---------------------------------------------------------------------------
# LLM-backed call implementations
# ---------------------------------------------------------------------------

class LLMPlanner(PlannerCall):
    def __init__(self, client: LLMClient, model: str, action_templates: list[str]):
        self.client = client
        self.model = model
        self.action_templates = action_templates

    def __call__(self, observation_text, agent_world_state, history, decoding_seed):
        usr = planner_user(observation_text, agent_world_state, self.action_templates, history)
        return self.client.call_typed(
            model=self.model,
            call_type="planner",
            system_prompt=PLANNER_SYS,
            user_prompt=usr,
            seed=decoding_seed,
        )


class LLMUpdater(UpdaterCall):
    def __init__(self, client: LLMClient, model: str):
        self.client = client
        self.model = model

    def __call__(
        self,
        observation_text,
        observation_partial_state,
        prev_agent_world_state,
        last_action,
        last_action_outcome,
        decoding_seed,
    ):
        usr = updater_user(
            observation_text=observation_text,
            observation_partial_state=observation_partial_state,
            prev_agent_world_state=prev_agent_world_state,
            last_action=last_action,
            last_action_outcome=last_action_outcome,
        )
        return self.client.call_typed(
            model=self.model,
            call_type="updater",
            system_prompt=UPDATER_SYS,
            user_prompt=usr,
            seed=decoding_seed,
        )


class LLMSelfDiag(SelfDiagCall):
    def __init__(self, client: LLMClient, model: str):
        self.client = client
        self.model = model

    def __call__(self, agent_world_state, proposed_action, required_preconditions, decoding_seed):
        usr = self_diag_user(
            agent_world_state=agent_world_state,
            proposed_action=proposed_action,
            required_preconditions=required_preconditions,
        )
        return self.client.call_typed(
            model=self.model,
            call_type="self_diag",
            system_prompt=SELF_DIAG_SYS,
            user_prompt=usr,
            seed=decoding_seed,
        )


# ---------------------------------------------------------------------------
# Mode A free-form call implementations (Exp B, 2026-06-07).
#
# The Updater outputs a natural-language paragraph; the Planner reads that
# paragraph and emits a 2-line REASONING/ACTION response; the Self-Diag emits
# a 2-line ASSESSMENT/VERDICT response. No JSON validation, no retries (we
# fall back to "noop" if parsing fails — same safety net as Mode C fallback).
# ---------------------------------------------------------------------------

_EMPTY_WS_STUB = {
    "objects": {}, "locations": {}, "relations": [], "inventory": [],
    "open_subgoals": [], "completed_subgoals": [], "blocked_dependencies": [], "beliefs": [],
}


def _extract_nl_memory(agent_world_state: dict) -> str:
    if not isinstance(agent_world_state, dict):
        return ""
    return str(agent_world_state.get(NL_MEMORY_KEY) or "")


def _wrap_nl_memory(text: str) -> dict:
    """Build a runner-compatible world-state dict that carries NL memory."""
    out = dict(_EMPTY_WS_STUB)
    out["objects"] = {}
    out["locations"] = {}
    out["relations"] = []
    out["inventory"] = []
    out["open_subgoals"] = []
    out["completed_subgoals"] = []
    out["blocked_dependencies"] = []
    out["beliefs"] = []
    # NB: this key is silently dropped by `canonicalize_world_state` (it only
    # copies known keys). We re-inject it after canonicalisation by stashing it
    # back in the runner — actually we can't, runner is read-only. Workaround:
    # carry NL memory in `beliefs` (which the canonicaliser preserves).
    out["beliefs"] = [{"prop": f"NL::{text}", "confidence": 1.0}]
    return out


def _extract_nl_from_canonical_ws(ws: dict) -> str:
    """Pull the NL memory back out of canonicalised world-state.beliefs."""
    if not isinstance(ws, dict):
        return ""
    for b in ws.get("beliefs") or []:
        if not isinstance(b, dict):
            continue
        prop = str(b.get("prop", ""))
        if prop.startswith("NL::"):
            return prop[len("NL::"):]
    return ""


_ACTION_LINE_RE = re.compile(r"(?im)^\s*ACTION\s*[:=]\s*(.+?)\s*$")
_VERDICT_LINE_RE = re.compile(r"(?im)^\s*VERDICT\s*[:=]\s*([A-Za-z_]+)\s*$")


def _parse_planner_mode_a(text: str, action_templates: list[str]) -> tuple[str, str]:
    """Return (next_action, reasoning_text). Falls back to 'noop'.

    Tries:
      1. The `ACTION: ...` line.
      2. Any line in the raw text that exactly matches one of the templates.
    """
    reasoning = ""
    # capture REASONING content for diagnostics
    m_reason = re.search(r"(?im)^\s*REASONING\s*[:=]\s*(.+?)$", text)
    if m_reason:
        reasoning = m_reason.group(1).strip()

    m = _ACTION_LINE_RE.search(text or "")
    candidate = m.group(1).strip().strip("'\"`") if m else ""
    # Strip trailing punctuation, but preserve balanced parens (e.g. move(n3))
    while candidate and candidate[-1] in ".,;":
        candidate = candidate[:-1]
    # If trailing ')' is unbalanced, strip it; otherwise keep.
    if candidate.endswith(")") and candidate.count("(") < candidate.count(")"):
        candidate = candidate[:-1]
    # Strip wrapping markdown
    if candidate.startswith("`") and candidate.endswith("`"):
        candidate = candidate.strip("`")
    if candidate:
        return candidate, reasoning

    # Fallback: scan for any template literal
    if action_templates:
        for tmpl in action_templates:
            if tmpl and tmpl in (text or ""):
                return tmpl, reasoning
    return "noop", reasoning


def _parse_self_diag_mode_a(text: str) -> tuple[bool, str]:
    """Return (self_check_valid, assessment_text). Defaults to True (execute)."""
    assessment = ""
    m_a = re.search(r"(?im)^\s*ASSESSMENT\s*[:=]\s*(.+?)$", text or "")
    if m_a:
        assessment = m_a.group(1).strip()
    m = _VERDICT_LINE_RE.search(text or "")
    if m:
        v = m.group(1).strip().upper()
        if v in ("REPLAN", "ABORT", "NO", "FALSE", "INVALID"):
            return False, assessment
        if v in ("EXECUTE", "RUN", "GO", "YES", "TRUE", "VALID"):
            return True, assessment
    return True, assessment  # default to execute on unparseable


class ModeAUpdater(UpdaterCall):
    def __init__(self, client: LLMClient, model: str):
        self.client = client
        self.model = model

    def __call__(
        self,
        observation_text,
        observation_partial_state,
        prev_agent_world_state,
        last_action,
        last_action_outcome,
        decoding_seed,
    ):
        prev_nl = _extract_nl_from_canonical_ws(prev_agent_world_state)
        usr = updater_user_mode_a(
            observation_text=observation_text,
            prev_current_understanding=prev_nl,
            last_action=last_action,
            last_action_outcome=last_action_outcome,
        )
        raw = self.client.call_raw(
            model=self.model,
            system_prompt=MODE_A_UPDATER_SYS,
            user_prompt=usr,
            seed=decoding_seed if (self.model.startswith("gpt-") or self.model.startswith("azure:")) else None,
            temperature=0.0,
            max_tokens=MODE_A_MAX_TOKENS["updater"],
        )
        # Keep what fits the contract: full_world_state dict + delta lists.
        new_understanding = (raw.text or "").strip()
        if not new_understanding and prev_nl:
            new_understanding = prev_nl  # carry forward if model returned empty
        parsed = {
            "changed_facts": [],
            "removed_facts": [],
            "full_world_state": _wrap_nl_memory(new_understanding),
        }
        return CallOutcome(
            parsed=parsed,
            raw_text=raw.text,
            valid_json=True if raw.text else False,
            retries=0,
            input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens,
            wallclock_ms=raw.wallclock_ms,
            fallback_used=not bool(raw.text),
            extra={"system_fingerprint": raw.system_fingerprint, "mode": "A_free"},
        )


class ModeAPlanner(PlannerCall):
    def __init__(self, client: LLMClient, model: str, action_templates: list[str]):
        self.client = client
        self.model = model
        self.action_templates = action_templates

    def __call__(self, observation_text, agent_world_state, history, decoding_seed):
        nl_mem = _extract_nl_from_canonical_ws(agent_world_state)
        usr = planner_user_mode_a(
            observation_text=observation_text,
            current_understanding=nl_mem,
            action_templates=self.action_templates,
            history=history,
        )
        raw = self.client.call_raw(
            model=self.model,
            system_prompt=MODE_A_PLANNER_SYS,
            user_prompt=usr,
            seed=decoding_seed if (self.model.startswith("gpt-") or self.model.startswith("azure:")) else None,
            temperature=0.0,
            max_tokens=MODE_A_MAX_TOKENS["planner"],
        )
        action, reasoning = _parse_planner_mode_a(raw.text or "", self.action_templates)
        parsed = {
            "next_action": action,
            "required_preconditions": [],
            "expected_effects": [],
            "confidence": 0.5,
        }
        return CallOutcome(
            parsed=parsed,
            raw_text=raw.text,
            valid_json=True if raw.text else False,
            retries=0,
            input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens,
            wallclock_ms=raw.wallclock_ms,
            fallback_used=not bool(raw.text),
            extra={"system_fingerprint": raw.system_fingerprint, "mode": "A_free", "reasoning": reasoning},
        )


class ModeASelfDiag(SelfDiagCall):
    def __init__(self, client: LLMClient, model: str):
        self.client = client
        self.model = model

    def __call__(self, agent_world_state, proposed_action, required_preconditions, decoding_seed):
        nl_mem = _extract_nl_from_canonical_ws(agent_world_state)
        usr = self_diag_user_mode_a(
            current_understanding=nl_mem,
            proposed_action=proposed_action,
        )
        raw = self.client.call_raw(
            model=self.model,
            system_prompt=MODE_A_SELF_DIAG_SYS,
            user_prompt=usr,
            seed=decoding_seed if (self.model.startswith("gpt-") or self.model.startswith("azure:")) else None,
            temperature=0.0,
            max_tokens=MODE_A_MAX_TOKENS["self_diag"],
        )
        valid, assessment = _parse_self_diag_mode_a(raw.text or "")
        parsed = {
            "self_check_valid": valid,
            "missing_preconditions": [],
            "should_replan": not valid,
        }
        return CallOutcome(
            parsed=parsed,
            raw_text=raw.text,
            valid_json=True if raw.text else False,
            retries=0,
            input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens,
            wallclock_ms=raw.wallclock_ms,
            fallback_used=not bool(raw.text),
            extra={"system_fingerprint": raw.system_fingerprint, "mode": "A_free", "assessment": assessment},
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_llm_agent(
    client: LLMClient,
    model: str,
    action_templates: list[str],
    memory_mode: str = "C_struct",
) -> BaseAgent:
    """Build a BaseAgent that uses LLM-backed Planner/Updater/SelfDiag.

    memory_mode:
      - "C_struct" (default): structured JSON world-state, 3 retries (Stage 4 default).
      - "A_free":  natural-language understanding, no JSON validation (Exp B).
    """
    if memory_mode == "A_free":
        return BaseAgent(
            name=f"llm:{model}:A_free",
            model_id=model,
            memory_mode=memory_mode,
            planner=ModeAPlanner(client, model, action_templates),
            updater=ModeAUpdater(client, model),
            self_diag=ModeASelfDiag(client, model),
        )
    return BaseAgent(
        name=f"llm:{model}:{memory_mode}",
        model_id=model,
        memory_mode=memory_mode,
        planner=LLMPlanner(client, model, action_templates),
        updater=LLMUpdater(client, model),
        self_diag=LLMSelfDiag(client, model),
    )
