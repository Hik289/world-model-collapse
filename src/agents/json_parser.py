"""Robust JSON parser for LLM-call outputs (anchor_3 + production).

Returns (parsed_dict_or_none, valid_bool, error_str). Stripping markdown fences
and finding the first balanced JSON object; required-key + light type
validation per call_type.
"""

from __future__ import annotations

import json
import re
from typing import Any


PLANNER_SCHEMA = {
    "next_action": "str",
    "required_preconditions": "list[str]",
    "expected_effects": "list",
    "confidence": "number_in_0_1",
}

UPDATER_SCHEMA = {
    "changed_facts": "list",
    "removed_facts": "list",
    "full_world_state": "dict",
}

SELF_DIAG_SCHEMA = {
    "self_check_valid": "bool",
    "missing_preconditions": "list[str]",
    "should_replan": "bool",
}

SCHEMAS = {
    "planner": PLANNER_SCHEMA,
    "updater": UPDATER_SCHEMA,
    "self_diag": SELF_DIAG_SCHEMA,
}

_FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*\n?|\n?```\s*$", re.MULTILINE)


def _strip_fence(text: str) -> str:
    s = text.strip()
    s = _FENCE_RE.sub("", s).strip()
    return s


def _find_first_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _check_type(val: Any, spec: str) -> bool:
    if spec == "str":
        return isinstance(val, str)
    if spec == "bool":
        return isinstance(val, bool)
    if spec == "number":
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    if spec == "number_in_0_1":
        return (
            isinstance(val, (int, float))
            and not isinstance(val, bool)
            and 0.0 <= float(val) <= 1.0
        )
    if spec == "list":
        return isinstance(val, list)
    if spec == "list[str]":
        return isinstance(val, list) and all(isinstance(x, str) for x in val)
    if spec == "dict":
        return isinstance(val, dict)
    return True


def parse_call_output(raw_text: str, call_type: str) -> tuple[dict | None, bool, str]:
    if call_type not in SCHEMAS:
        raise ValueError(f"Unknown call_type {call_type}")
    schema = SCHEMAS[call_type]

    stripped = _strip_fence(raw_text)
    candidate = _find_first_object(stripped)
    if candidate is None:
        return None, False, "no_json_object_found"
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as e:
        return None, False, f"json_decode_error:{e.msg}"
    if not isinstance(obj, dict):
        return None, False, "not_a_dict"

    for key, type_spec in schema.items():
        if key not in obj:
            return obj, False, f"missing_key:{key}"
        if not _check_type(obj[key], type_spec):
            return obj, False, f"bad_type:{key}:{type_spec}"

    return obj, True, ""
