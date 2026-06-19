"""Prompt templates for the 3-call loop (anchor_3 warm-up + production).

Design principles:
  - System prompt establishes role + STRICT JSON output (no prose, no fences).
  - User prompt provides observation + current agent state + (for self-diag)
    the proposed action.
  - Schema is described in plain English alongside an example.
  - Output: ONLY the JSON object, no commentary.

Schema must match `src/agents/json_parser.py` (PLANNER_SCHEMA etc.).
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

PLANNER_SYS = (
    "You are the Planner of a tool-using agent operating in a structured rule-based environment. "
    "On each step, given an observation and the agent's current structured world-state, "
    "you output the next concrete action to take. "
    "OUTPUT FORMAT: a single JSON object only. No prose. No markdown fences. "
    "The object must contain exactly these keys: "
    '{"next_action": str, "required_preconditions": list[str], "expected_effects": list, "confidence": number in [0,1]}. '
    "next_action must match one of the action templates exactly (e.g. 'move(n3)', 'call(tool_2)', 'go(room_1)', 'noop'). "
    "required_preconditions lists facts that must hold for the action to succeed (e.g. 'hold(key_0)'). "
    "expected_effects lists facts produced (e.g. 'at(n3)'). "
    "confidence is your subjective certainty in [0,1]."
)

UPDATER_SYS = (
    "You are the World-State Updater of a tool-using agent. "
    "Given an observation (text + partial structured state from the env) and the prior agent world-state, "
    "you output the new structured world-state. "
    "OUTPUT FORMAT: a single JSON object only. No prose. No markdown fences. "
    "The object must contain exactly these keys: "
    '{"changed_facts": list, "removed_facts": list, "full_world_state": dict}. '
    "full_world_state must be a dict with the keys: objects, locations, relations, inventory, "
    "open_subgoals, completed_subgoals, blocked_dependencies, beliefs. "
    "Unused keys may be empty lists/dicts. "
    "changed_facts and removed_facts are lists of fact-strings describing the delta from prior state."
)

SELF_DIAG_SYS = (
    "You are the Self-Diagnosis module of a tool-using agent. "
    "Given the agent's current world-state and a proposed action with its claimed required preconditions, "
    "you decide whether the action is valid to execute now. "
    "OUTPUT FORMAT: a single JSON object only. No prose. No markdown fences. "
    "The object must contain exactly these keys: "
    '{"self_check_valid": bool, "missing_preconditions": list[str], "should_replan": bool}. '
    "self_check_valid is true iff all required_preconditions are satisfied by the current world-state. "
    "missing_preconditions lists any unmet preconditions. "
    "should_replan is true iff the action should NOT be executed and a new plan is needed."
)


# ---------------------------------------------------------------------------
# Mode A free-form prompts (cross-harness test, Exp B 2026-06-07).
#
# The agent maintains a NATURAL LANGUAGE "current understanding" instead of a
# structured JSON world-state. The action must still match an enum template
# because env.step() expects a parseable action string. All other inference
# (preconditions, effects, beliefs) lives inside the prose.
# ---------------------------------------------------------------------------

MODE_A_UPDATER_SYS = (
    "You are the memory updater of a tool-using agent operating in a structured environment. "
    "Given a new observation, the prior natural-language 'current understanding', and the last action + outcome, "
    "you write the new 'current understanding': a concise English paragraph (max ~200 words) summarising "
    "what the agent now knows about the world, what subgoals are open, what is blocked, and any beliefs. "
    "Do NOT output JSON. Do NOT use bullet lists. Just one paragraph of plain English prose."
)

MODE_A_PLANNER_SYS = (
    "You are the planner of a tool-using agent. "
    "Given an observation, the natural-language 'current understanding' of the world, and a list of available action templates, "
    "you must choose the next concrete action. "
    "Your output MUST be exactly two lines:\n"
    "  REASONING: <one or two sentences explaining your choice>\n"
    "  ACTION: <one action string that matches one of the action templates exactly, e.g. move(n3), call(tool_2), open(ctr_0), noop>\n"
    "The ACTION line is mandatory. It MUST be a single template instance, nothing else, no quotes, no JSON."
)

MODE_A_SELF_DIAG_SYS = (
    "You are the self-check module of a tool-using agent. "
    "Given the natural-language 'current understanding' and a proposed action, decide if the action is safe to execute now. "
    "Your output MUST be exactly two lines:\n"
    "  ASSESSMENT: <one or two sentences>\n"
    "  VERDICT: <one of EXECUTE | REPLAN>\n"
    "EXECUTE means run the action. REPLAN means abandon the action and re-plan."
)


def planner_user_mode_a(
    observation_text: str,
    current_understanding: str,
    action_templates: list[str],
    history: list[dict] | None = None,
) -> str:
    parts = [
        "OBSERVATION:",
        observation_text,
        "",
        "CURRENT UNDERSTANDING (your own prose memory from prior step):",
        current_understanding or "(no prior understanding — this is the first step)",
        "",
        f"AVAILABLE ACTION TEMPLATES (pick exactly one): {action_templates[:60]}",
    ]
    if history:
        recent = history[-5:]
        parts.extend([
            "",
            "RECENT HISTORY (last 5 actions):",
            _json_short(recent, limit=800),
        ])
    parts.extend([
        "",
        "Reply in the exact 2-line REASONING/ACTION format.",
    ])
    return "\n".join(parts)


def updater_user_mode_a(
    observation_text: str,
    prev_current_understanding: str,
    last_action: str | None,
    last_action_outcome: dict | None,
) -> str:
    parts = [
        "NEW OBSERVATION:",
        observation_text,
        "",
        "PRIOR UNDERSTANDING (the paragraph you wrote last step):",
        prev_current_understanding or "(none — first step)",
        "",
        f"LAST ACTION: {last_action!r}",
        f"LAST ACTION OUTCOME: {_json_short(last_action_outcome or {}, limit=400)}",
        "",
        "Write the new 'current understanding' paragraph (English prose, ~200 words max). No JSON.",
    ]
    return "\n".join(parts)


def self_diag_user_mode_a(
    current_understanding: str,
    proposed_action: str,
) -> str:
    parts = [
        "CURRENT UNDERSTANDING:",
        current_understanding or "(empty)",
        "",
        f"PROPOSED ACTION: {proposed_action!r}",
        "",
        "Reply in the exact 2-line ASSESSMENT/VERDICT format.",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# User-prompt formatters
# ---------------------------------------------------------------------------

def _json_short(obj: Any, limit: int = 4000) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if len(s) <= limit:
        return s
    # Truncate but keep valid-ish marker
    return s[:limit] + "...[TRUNCATED]"


def planner_user(
    observation_text: str,
    agent_world_state: dict,
    action_templates: list[str],
    history: list[dict] | None = None,
) -> str:
    parts = [
        "OBSERVATION:",
        observation_text,
        "",
        "CURRENT WORLD STATE (structured):",
        _json_short(agent_world_state, limit=4000),
        "",
        f"AVAILABLE ACTION TEMPLATES (use exactly one of these patterns): {action_templates[:60]}",
    ]
    if history:
        recent = history[-5:]
        parts.extend([
            "",
            "RECENT HISTORY (last 5 actions):",
            _json_short(recent, limit=800),
        ])
    parts.extend([
        "",
        "Output JSON only.",
    ])
    return "\n".join(parts)


def updater_user(
    observation_text: str,
    observation_partial_state: dict,
    prev_agent_world_state: dict,
    last_action: str | None,
    last_action_outcome: dict | None,
) -> str:
    parts = [
        "OBSERVATION TEXT:",
        observation_text,
        "",
        "OBSERVATION STRUCTURED (what the env revealed this step):",
        _json_short(observation_partial_state, limit=4000),
        "",
        "PRIOR AGENT WORLD STATE:",
        _json_short(prev_agent_world_state, limit=4000),
        "",
        f"LAST ACTION: {last_action!r}",
        f"LAST ACTION OUTCOME: {_json_short(last_action_outcome or {}, limit=400)}",
        "",
        "Produce the new agent world-state reflecting the observation. Output JSON only.",
    ]
    return "\n".join(parts)


def self_diag_user(
    agent_world_state: dict,
    proposed_action: str,
    required_preconditions: list[str],
) -> str:
    parts = [
        "CURRENT WORLD STATE:",
        _json_short(agent_world_state, limit=4000),
        "",
        f"PROPOSED ACTION: {proposed_action!r}",
        f"REQUIRED PRECONDITIONS: {required_preconditions}",
        "",
        "Decide whether the proposed action is valid. Output JSON only.",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Anchor_3 fixture prompts (5 templates × 10 repeats per call_type).
# Each template embeds 1 scenario: normal / missing precondition / nested array.
# Covered envs: graph_nav (gn), tool_dag (td), stateful_puzzle (sp).
# ---------------------------------------------------------------------------

# Fixture observation+state pairs (kept small to bound token cost).
_FIX_GN_NORMAL = {
    "obs_text": "At n0. Inventory: ['key_0']. Open subgoals: ['reach(n2)'].",
    "obs_partial": {
        "objects": {"n0": {"type": "node", "props": {}}, "n2": {"type": "node", "props": {"is_goal": True}}},
        "locations": {"n0": {"type": "node", "contents": []}},
        "relations": [{"subj": "agent", "rel": "at", "obj": "n0"}, {"subj": "n0", "rel": "adjacent", "obj": "n1"}],
        "inventory": ["key_0"], "open_subgoals": ["reach(n2)"], "completed_subgoals": [],
        "blocked_dependencies": [], "beliefs": [],
    },
    "templates": ["move(n0)", "move(n1)", "move(n2)", "pick(key_0)", "drop(key_0)", "noop"],
}

_FIX_GN_MISSING = {
    "obs_text": "At n0. Door door_0 between n0 and n1 is locked. Need switch_0 on AND hold(key_0). Inventory: [].",
    "obs_partial": {
        "objects": {"door_0": {"type": "door", "props": {"unlocked": False}}, "key_0": {"type": "key", "props": {"location": "n2"}}},
        "locations": {"n0": {"type": "node", "contents": ["switch_0"]}},
        "relations": [{"subj": "agent", "rel": "at", "obj": "n0"}],
        "inventory": [], "open_subgoals": ["reach(n1)"], "completed_subgoals": [],
        "blocked_dependencies": [{"action": "unlock(door_0)", "missing": ["hold(key_0)", "switch_on(switch_0)"]}],
        "beliefs": [],
    },
    "templates": ["unlock(door_0)", "turn_on(switch_0)", "move(n2)", "pick(key_0)", "noop"],
}

_FIX_GN_NESTED = {
    "obs_text": "At n0. Multiple decoys: ['n3','n7']. Subgoals chain: ['reach(n5)','pickup(key_2)'].",
    "obs_partial": {
        "objects": {
            "n0": {"type": "node", "props": {}},
            "n3": {"type": "node", "props": {"decoy": True}},
            "n7": {"type": "node", "props": {"decoy": True}},
        },
        "locations": {"n0": {"type": "node", "contents": []}},
        "relations": [{"subj": "agent", "rel": "at", "obj": "n0"}],
        "inventory": [],
        "open_subgoals": ["reach(n5)", "pickup(key_2)"],
        "completed_subgoals": [],
        "blocked_dependencies": [],
        "beliefs": [{"prop": "n3_is_decoy", "confidence": 0.9}],
    },
    "templates": ["move(n3)", "move(n7)", "move(n2)", "inspect(n3)", "noop"],
}

_FIX_TD_NORMAL = {
    "obs_text": "Active vars: ['v0','v1']. Target var: TARGET (type TypeC). Open subgoals: ['finish(TARGET)'].",
    "obs_partial": {
        "objects": {
            "v0": {"type": "variable", "props": {"data_type": "TypeA"}},
            "v1": {"type": "variable", "props": {"data_type": "TypeB"}},
            "tool_0": {"type": "tool", "props": {"output_type": "TypeC", "layer": 0, "input_types_str": "TypeA,TypeB"}},
        },
        "locations": {},
        "relations": [],
        "inventory": ["v0", "v1"],
        "open_subgoals": ["finish(TARGET)"],
        "completed_subgoals": [],
        "blocked_dependencies": [],
        "beliefs": [],
    },
    "templates": ["call(tool_0)", "inspect_var(v0)", "finish(TARGET)", "noop"],
}

_FIX_SP_NORMAL = {
    "obs_text": "In room_0. Inventory: ['item_0']. Open subgoals: ['sg_0'].",
    "obs_partial": {
        "objects": {
            "room_0": {"type": "room", "props": {}},
            "item_0": {"type": "item", "props": {"location": "inventory"}},
            "ctr_0": {"type": "container", "props": {"location": "room_0", "open": False}},
        },
        "locations": {"room_0": {"type": "room", "contents": ["ctr_0"]}},
        "relations": [{"subj": "agent", "rel": "at", "obj": "room_0"}],
        "inventory": ["item_0"],
        "open_subgoals": ["sg_0"],
        "completed_subgoals": [],
        "blocked_dependencies": [{"action": "finish_subgoal(sg_0)", "missing": ["open(ctr_0)"]}],
        "beliefs": [],
    },
    "templates": ["open(ctr_0)", "close(ctr_0)", "finish_subgoal(sg_0)", "go(room_1)", "noop"],
}


FIXTURES = [
    ("gn_normal", _FIX_GN_NORMAL),
    ("gn_missing", _FIX_GN_MISSING),
    ("gn_nested", _FIX_GN_NESTED),
    ("td_normal", _FIX_TD_NORMAL),
    ("sp_normal", _FIX_SP_NORMAL),
]


def build_anchor3_prompt(call_type: str, fixture_id: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for an anchor_3 warm-up call.

    Picks one of the 5 fixtures and renders the appropriate user prompt.
    """
    fix_map = dict(FIXTURES)
    if fixture_id not in fix_map:
        raise ValueError(f"unknown fixture {fixture_id}")
    fx = fix_map[fixture_id]

    if call_type == "planner":
        sys_prompt = PLANNER_SYS
        usr_prompt = planner_user(
            observation_text=fx["obs_text"],
            agent_world_state=fx["obs_partial"],
            action_templates=fx["templates"],
            history=None,
        )
    elif call_type == "updater":
        sys_prompt = UPDATER_SYS
        usr_prompt = updater_user(
            observation_text=fx["obs_text"],
            observation_partial_state=fx["obs_partial"],
            prev_agent_world_state={
                "objects": {}, "locations": {}, "relations": [], "inventory": [],
                "open_subgoals": [], "completed_subgoals": [], "blocked_dependencies": [], "beliefs": [],
            },
            last_action=None,
            last_action_outcome=None,
        )
    elif call_type == "self_diag":
        sys_prompt = SELF_DIAG_SYS
        # Pick the first plausible action template as the proposed action.
        proposed = fx["templates"][0]
        # Pull preconditions from blocked deps if present.
        preconds: list[str] = []
        for b in fx["obs_partial"].get("blocked_dependencies", []):
            if b.get("action") == proposed:
                preconds = list(b.get("missing", []))
                break
        usr_prompt = self_diag_user(
            agent_world_state=fx["obs_partial"],
            proposed_action=proposed,
            required_preconditions=preconds,
        )
    else:
        raise ValueError(f"unknown call_type {call_type}")

    return sys_prompt, usr_prompt
