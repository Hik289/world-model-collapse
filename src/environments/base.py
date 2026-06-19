"""Base Environment contract — EXP_PLAN v2.1 §2.1.

All three rule-based environments (graph_nav, tool_dag, stateful_puzzle)
implement this contract. Deterministic by construction:

  - All randomness flows through a per-environment `random.Random(seed)` instance
    created in `reset()`. No global RNG access. No wallclock. No uuid.
  - All exposed dict/list structures are canonicalized via `canonical_json()`
    before hashing.
  - List ordering with no inherent semantics is sorted on emission.

This file is the single source of truth for the env API. anchor_1 determinism
test (experiments/verification_specs/verify_anchor_1.py) imports from here.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Canonical serialization helpers
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    """Deterministic JSON dump used for hashing.

    Matches anchor_1 spec: ``sort_keys=True, separators=(',', ':'),
    ensure_ascii=False``.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(obj: Any) -> str:
    """SHA256 over canonical JSON of obj."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Data classes (lightweight; canonical via .to_dict())
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """Single observation handed back to the agent.

    `text` is a natural-language rendering used by Planner prompts; `partial_state`
    is the structured info the env reveals (may differ from gold based on
    `obs_noise`).
    """

    text: str
    partial_state: dict
    done: bool = False
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "partial_state": self.partial_state,
            "done": self.done,
            "info": self.info,
        }


@dataclass
class StepResult:
    """Return value of `env.step(action)`."""

    observation: Observation
    reward: float
    done: bool
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        # Deterministic numeric formatting via JSON round-trip; reward kept as float.
        return {
            "observation": self.observation.to_dict(),
            "reward": self.reward,
            "done": self.done,
            "info": self.info,
        }


@dataclass
class ValidityCheck:
    """Output of `env.check_action_validity(action)`."""

    valid: bool
    missing: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "missing": sorted(self.missing),  # list of strings, no semantic order
            "reason": self.reason,
        }


@dataclass
class EnvMeta:
    """Static description of an env instance returned by `get_meta()`.

    Includes action template list used by the deterministic policy in anchor_1.
    """

    name: str
    stress_config: dict
    task_config: dict
    action_templates: list[str]  # canonical list of legal action shapes
    object_ids: list[str]
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "stress_config": self.stress_config,
            "task_config": self.task_config,
            "action_templates": list(self.action_templates),
            "object_ids": list(self.object_ids),
            "extra": self.extra,
        }


# ---------------------------------------------------------------------------
# WorldState canonicalization
# ---------------------------------------------------------------------------

EMPTY_WORLD_STATE: dict = {
    "objects": {},
    "locations": {},
    "relations": [],
    "inventory": [],
    "open_subgoals": [],
    "completed_subgoals": [],
    "blocked_dependencies": [],
    "beliefs": [],
}


def empty_world_state() -> dict:
    """Return a fresh canonical empty WorldState (§2.2)."""
    # Deep copy via JSON round-trip to avoid shared references.
    return json.loads(canonical_json(EMPTY_WORLD_STATE))


def canonicalize_world_state(ws: dict) -> dict:
    """Return a sorted/canonical copy of a WorldState dict.

    Sorting rules:
      - `objects`, `locations` dicts: keys sorted lex.
      - `inventory`, `open_subgoals`, `completed_subgoals`: sorted lex.
      - `relations`: sorted by (subj, rel, obj).
      - `blocked_dependencies`: sorted by (action, sorted(missing)).
      - `beliefs`: sorted by prop string.
    Inside each object/location, `contents`/`props` likewise canonicalized.
    """
    out: dict = {}
    # Defensive: ws may not be a dict if upstream returned malformed payload
    # (e.g., haiku occasionally emits a top-level list/string for
    # full_world_state. anchor_3 + json_parser validate top-level dict, but
    # nested values like objects[oid] or relations[i] can still be wrong type.
    # We coerce silently and drop malformed items rather than crash.
    # P4 ERR bug fix (2026-05-31): replace .get() on potentially non-dict.
    if not isinstance(ws, dict):
        ws = {}

    objs = ws.get("objects", {})
    if not isinstance(objs, dict):
        objs = {}
    out["objects"] = {
        oid: {
            "type": (meta.get("type", "") if isinstance(meta, dict) else ""),
            "props": dict(sorted(((meta.get("props") if isinstance(meta, dict) else None) or {}).items())),
        }
        for oid, meta in sorted(objs.items())
    }

    locs = ws.get("locations", {})
    if not isinstance(locs, dict):
        locs = {}
    out["locations"] = {
        lid: {
            "type": (meta.get("type", "") if isinstance(meta, dict) else ""),
            "contents": sorted((meta.get("contents") if isinstance(meta, dict) else None) or []),
        }
        for lid, meta in sorted(locs.items())
    }

    rels_raw = ws.get("relations") or []
    if not isinstance(rels_raw, list):
        rels_raw = []
    rels = [r for r in rels_raw if isinstance(r, dict)]
    out["relations"] = sorted(
        ({"subj": str(r.get("subj", "")), "rel": str(r.get("rel", "")), "obj": str(r.get("obj", ""))} for r in rels),
        key=lambda r: (r["subj"], r["rel"], r["obj"]),
    )

    inv = ws.get("inventory") or []
    out["inventory"] = sorted([str(x) for x in inv if isinstance(x, (str, int, float))]) if isinstance(inv, list) else []
    osg = ws.get("open_subgoals") or []
    out["open_subgoals"] = sorted([str(x) for x in osg if isinstance(x, (str, int, float))]) if isinstance(osg, list) else []
    csg = ws.get("completed_subgoals") or []
    out["completed_subgoals"] = sorted([str(x) for x in csg if isinstance(x, (str, int, float))]) if isinstance(csg, list) else []

    bd_raw = ws.get("blocked_dependencies") or []
    if not isinstance(bd_raw, list):
        bd_raw = []
    bd = [b for b in bd_raw if isinstance(b, dict)]
    out["blocked_dependencies"] = sorted(
        (
            {
                "action": str(b.get("action", "")),
                "missing": sorted([str(m) for m in (b.get("missing") or []) if isinstance(m, (str, int, float))]) if isinstance(b.get("missing"), list) else [],
            }
            for b in bd
        ),
        key=lambda b: (b["action"], tuple(b["missing"])),
    )

    bel_raw = ws.get("beliefs") or []
    if not isinstance(bel_raw, list):
        bel_raw = []
    bel = [b for b in bel_raw if isinstance(b, dict)]
    out["beliefs"] = sorted(
        (
            {"prop": str(b.get("prop", "")), "confidence": float(b.get("confidence", 0.0)) if isinstance(b.get("confidence"), (int, float)) and not isinstance(b.get("confidence"), bool) else 0.0}
            for b in bel
        ),
        key=lambda b: b["prop"],
    )

    return out


# ---------------------------------------------------------------------------
# Abstract Environment
# ---------------------------------------------------------------------------

class Environment(ABC):
    """Common rule-based environment interface (EXP_PLAN §2.1)."""

    name: str = "base"

    @abstractmethod
    def reset(self, task_config: dict, seed: int) -> Observation:
        """Initialize state from (task_config, seed). Must be fully deterministic."""

    @abstractmethod
    def step(self, action: str) -> StepResult:
        """Apply action. Returns StepResult; updates internal state deterministically."""

    @abstractmethod
    def get_gold_state(self) -> dict:
        """Canonicalized current gold WorldState."""

    @abstractmethod
    def check_action_validity(self, action: str) -> ValidityCheck:
        """Whether `action` would be valid *now*."""

    @abstractmethod
    def compute_error_labels(self, agent_state: dict, action: str) -> list[str]:
        """Diagnostic labels for (agent_state, action). Sorted lex on emission."""

    @abstractmethod
    def get_meta(self) -> EnvMeta:
        """Static instance descriptor (used by deterministic policies + logging)."""
