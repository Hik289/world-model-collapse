from __future__ import annotations
import os
import sys
_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)
from src.environments.base import canonicalize_world_state
_SKIP_NAMES = {'air', 'agent', 'inventory'}
_SKIP_PREFIXES = ('terminal', 'wire ', 'black wire', 'red wire', 'yellow wire', 'blue wire', 'green wire')

def _ref(node: dict) -> str:
    return f"{node.get('name')}#{node.get('uuid')}"

def _is_skippable(node: dict) -> bool:
    name = (node.get('name') or '').strip().lower()
    if name in _SKIP_NAMES:
        return True
    for p in _SKIP_PREFIXES:
        if name.startswith(p):
            return True
    return False

def _is_container(node: dict) -> bool:
    pc = node.get('propContainer')
    return isinstance(pc, dict) and str(pc.get('isContainer')).lower() == 'true'

def _is_open(node: dict) -> bool:
    pc = node.get('propContainer')
    if not isinstance(pc, dict):
        return True
    return bool(pc.get('isOpen'))

def _obj_props(node: dict) -> dict:
    props: dict = {}
    pc = node.get('propContainer')
    if isinstance(pc, dict) and str(pc.get('isClosable')).lower() == 'true':
        props['isOpen'] = bool(pc.get('isOpen'))
    if node.get('propLife') is not None:
        props['isLiving'] = True
    pd = node.get('propDevice')
    if isinstance(pd, dict) and str(pd.get('isActivable')).lower() == 'true':
        props['isActivated'] = bool(pd.get('isActivated'))
    return props

def _room_nodes(tree: dict) -> dict:
    out = {}
    for _, child in (tree.get('contents') or {}).items():
        if _is_skippable(child):
            continue
        out[_ref(child)] = child
    return out

def _agent_room_ref(tree: dict) -> str | None:
    for room_ref, room in _room_nodes(tree).items():
        for _, child in (room.get('contents') or {}).items():
            if (child.get('name') or '').strip().lower() == 'agent':
                return room_ref
    return None

def _agent_inventory(tree: dict) -> list[str]:
    inv: list[str] = []
    for _, room in _room_nodes(tree).items():
        for _, child in (room.get('contents') or {}).items():
            if (child.get('name') or '').strip().lower() != 'agent':
                continue
            for _, sub in (child.get('contents') or {}).items():
                if (sub.get('name') or '').strip().lower() == 'inventory':
                    for _, item in (sub.get('contents') or {}).items():
                        if _is_skippable(item):
                            continue
                        inv.append(_ref(item))
    return inv

def _walk_object(node: dict, parent_ref: str, objects: dict, locations: dict, respect_open: bool) -> None:
    if _is_skippable(node):
        return
    ref = _ref(node)
    locations.setdefault(parent_ref, {'type': '', 'contents': []})
    if ref not in locations[parent_ref]['contents']:
        locations[parent_ref]['contents'].append(ref)
    objects[ref] = {'type': node.get('name') or '', 'props': _obj_props(node)}
    if _is_container(node):
        locations.setdefault(ref, {'type': 'container', 'contents': []})
        if respect_open and (not _is_open(node)):
            return
        for _, child in (node.get('contents') or {}).items():
            if (child.get('name') or '').strip().lower() == 'inventory':
                continue
            _walk_object(child, ref, objects, locations, respect_open)

def _flatten_room(room_ref: str, room: dict, objects: dict, locations: dict, respect_open: bool) -> None:
    locations.setdefault(room_ref, {'type': 'room', 'contents': []})
    for _, child in (room.get('contents') or {}).items():
        name = (child.get('name') or '').strip().lower()
        if name == 'agent':
            continue
        if _is_skippable(child):
            continue
        _walk_object(child, room_ref, objects, locations, respect_open)

def build_full_gold(tree: dict) -> dict:
    objects: dict = {}
    locations: dict = {}
    for room_ref, room in _room_nodes(tree).items():
        _flatten_room(room_ref, room, objects, locations, respect_open=False)
    relations = []
    agent_room = _agent_room_ref(tree)
    if agent_room is not None:
        relations.append({'subj': 'agent', 'rel': 'at', 'obj': agent_room})
    inventory = sorted(_agent_inventory(tree))
    held = set(inventory)
    for lid, meta in locations.items():
        meta['contents'] = sorted((c for c in meta['contents'] if c not in held))
    ws = {'objects': objects, 'locations': locations, 'relations': relations, 'inventory': inventory, 'open_subgoals': [], 'completed_subgoals': [], 'blocked_dependencies': [], 'beliefs': []}
    return canonicalize_world_state(ws)

def build_partial_obs(tree: dict, agent_room: str | None=None, object_loc_memory: dict | None=None, re_emit_sticky: bool=False) -> dict:
    if agent_room is None:
        agent_room = _agent_room_ref(tree)
    objects: dict = {}
    locations: dict = {}
    rooms = _room_nodes(tree)
    if agent_room is not None and agent_room in rooms:
        _flatten_room(agent_room, rooms[agent_room], objects, locations, respect_open=True)
    inventory = sorted(_agent_inventory(tree))
    held = set(inventory)
    for lid, meta in locations.items():
        meta['contents'] = sorted((c for c in meta['contents'] if c not in held))
    if object_loc_memory is not None:
        for oid in held:
            object_loc_memory.pop(oid, None)
        for lid, meta in locations.items():
            for oid in meta['contents']:
                object_loc_memory[oid] = lid
        if re_emit_sticky:
            for oid, lid in sorted(object_loc_memory.items()):
                if oid in held:
                    continue
                locations.setdefault(lid, {'type': '', 'contents': []})
                if oid not in locations[lid]['contents']:
                    locations[lid]['contents'] = sorted(set(locations[lid]['contents']) | {oid})
                objects.setdefault(oid, {'type': oid.split('#', 1)[0], 'props': {}})
    relations = []
    if agent_room is not None:
        relations.append({'subj': 'agent', 'rel': 'at', 'obj': agent_room})
    ws = {'objects': objects, 'locations': locations, 'relations': relations, 'inventory': inventory, 'open_subgoals': [], 'completed_subgoals': [], 'blocked_dependencies': [], 'beliefs': []}
    return canonicalize_world_state(ws)

def compute_sc(tree: dict) -> dict:
    from src.evaluation.runner import world_state_facts
    full = build_full_gold(tree)
    n_facts = len(world_state_facts(full))
    n_rooms = len(_room_nodes(tree))
    n_objects = len(full.get('objects', {}))
    n_loc = len(full.get('locations', {}))
    n_contains = sum((len(m.get('contents', [])) for m in full.get('locations', {}).values()))
    return {'sc_facts': n_facts, 'n_rooms': n_rooms, 'n_objects': n_objects, 'n_locations': n_loc, 'n_containment': n_contains, 'sc_entities': n_objects + n_rooms}

def agent_room_ref(tree: dict) -> str | None:
    return _agent_room_ref(tree)
