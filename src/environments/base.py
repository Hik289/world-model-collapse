from __future__ import annotations
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode('utf-8')).hexdigest()

@dataclass
class Observation:
    text: str
    partial_state: dict
    done: bool = False
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {'text': self.text, 'partial_state': self.partial_state, 'done': self.done, 'info': self.info}

@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {'observation': self.observation.to_dict(), 'reward': self.reward, 'done': self.done, 'info': self.info}

@dataclass
class ValidityCheck:
    valid: bool
    missing: list[str] = field(default_factory=list)
    reason: str = ''

    def to_dict(self) -> dict:
        return {'valid': self.valid, 'missing': sorted(self.missing), 'reason': self.reason}

@dataclass
class EnvMeta:
    name: str
    stress_config: dict
    task_config: dict
    action_templates: list[str]
    object_ids: list[str]
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {'name': self.name, 'stress_config': self.stress_config, 'task_config': self.task_config, 'action_templates': list(self.action_templates), 'object_ids': list(self.object_ids), 'extra': self.extra}
EMPTY_WORLD_STATE: dict = {'objects': {}, 'locations': {}, 'relations': [], 'inventory': [], 'open_subgoals': [], 'completed_subgoals': [], 'blocked_dependencies': [], 'beliefs': []}

def empty_world_state() -> dict:
    return json.loads(canonical_json(EMPTY_WORLD_STATE))

def canonicalize_world_state(ws: dict) -> dict:
    out: dict = {}
    if not isinstance(ws, dict):
        ws = {}
    objs = ws.get('objects', {})
    if not isinstance(objs, dict):
        objs = {}
    out['objects'] = {oid: {'type': meta.get('type', '') if isinstance(meta, dict) else '', 'props': dict(sorted(((meta.get('props') if isinstance(meta, dict) else None) or {}).items()))} for oid, meta in sorted(objs.items())}
    locs = ws.get('locations', {})
    if not isinstance(locs, dict):
        locs = {}
    out['locations'] = {lid: {'type': meta.get('type', '') if isinstance(meta, dict) else '', 'contents': sorted((meta.get('contents') if isinstance(meta, dict) else None) or [])} for lid, meta in sorted(locs.items())}
    rels_raw = ws.get('relations') or []
    if not isinstance(rels_raw, list):
        rels_raw = []
    rels = [r for r in rels_raw if isinstance(r, dict)]
    out['relations'] = sorted(({'subj': str(r.get('subj', '')), 'rel': str(r.get('rel', '')), 'obj': str(r.get('obj', ''))} for r in rels), key=lambda r: (r['subj'], r['rel'], r['obj']))
    inv = ws.get('inventory') or []
    out['inventory'] = sorted([str(x) for x in inv if isinstance(x, (str, int, float))]) if isinstance(inv, list) else []
    osg = ws.get('open_subgoals') or []
    out['open_subgoals'] = sorted([str(x) for x in osg if isinstance(x, (str, int, float))]) if isinstance(osg, list) else []
    csg = ws.get('completed_subgoals') or []
    out['completed_subgoals'] = sorted([str(x) for x in csg if isinstance(x, (str, int, float))]) if isinstance(csg, list) else []
    bd_raw = ws.get('blocked_dependencies') or []
    if not isinstance(bd_raw, list):
        bd_raw = []
    bd = [b for b in bd_raw if isinstance(b, dict)]
    out['blocked_dependencies'] = sorted(({'action': str(b.get('action', '')), 'missing': sorted([str(m) for m in b.get('missing') or [] if isinstance(m, (str, int, float))]) if isinstance(b.get('missing'), list) else []} for b in bd), key=lambda b: (b['action'], tuple(b['missing'])))
    bel_raw = ws.get('beliefs') or []
    if not isinstance(bel_raw, list):
        bel_raw = []
    bel = [b for b in bel_raw if isinstance(b, dict)]
    out['beliefs'] = sorted(({'prop': str(b.get('prop', '')), 'confidence': float(b.get('confidence', 0.0)) if isinstance(b.get('confidence'), (int, float)) and (not isinstance(b.get('confidence'), bool)) else 0.0} for b in bel), key=lambda b: b['prop'])
    return out

class Environment(ABC):
    name: str = 'base'

    @abstractmethod
    def reset(self, task_config: dict, seed: int) -> Observation:
        pass

    @abstractmethod
    def step(self, action: str) -> StepResult:
        pass

    @abstractmethod
    def get_gold_state(self) -> dict:
        pass

    @abstractmethod
    def check_action_validity(self, action: str) -> ValidityCheck:
        pass

    @abstractmethod
    def compute_error_labels(self, agent_state: dict, action: str) -> list[str]:
        pass

    @abstractmethod
    def get_meta(self) -> EnvMeta:
        pass
