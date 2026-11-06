from __future__ import annotations
import os
import sys
_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)
from src.environments.base import canonicalize_world_state
TASK_TYPE_KEYS = ['pick_and_place_simple', 'look_at_obj_in_light', 'pick_clean_then_place_in_recep', 'pick_heat_then_place_in_recep', 'pick_cool_then_place_in_recep', 'pick_two_obj_and_place']
TASK_TYPE_SHORT = {'pick_and_place_simple': 'pick_and_place', 'look_at_obj_in_light': 'examine', 'pick_clean_then_place_in_recep': 'clean', 'pick_heat_then_place_in_recep': 'heat', 'pick_cool_then_place_in_recep': 'cool', 'pick_two_obj_and_place': 'pick_two'}
_OBJ_STATE_PROP = {'ishot': 'isHot', 'iscool': 'isCool', 'isclean': 'isClean', 'issliced': 'isSliced', 'istoggled': 'isToggled', 'ison': 'isOn'}

def task_type_from_gamefile(path: str) -> str:
    base = path
    for k in TASK_TYPE_KEYS:
        if k in base:
            return TASK_TYPE_SHORT[k]
    return 'unknown'

def _facts_to_tuples(facts) -> list[tuple[str, tuple[str, ...]]]:
    out = []
    for f in facts:
        try:
            args = tuple((a.name.strip() for a in f.arguments))
        except AttributeError:
            args = tuple((str(a).strip() for a in f[1]))
            out.append((f[0], args))
            continue
        out.append((f.name, args))
    return out

def parse_facts(facts):
    tuples = _facts_to_tuples(facts)
    obj_type: dict[str, str] = {}
    recep_type: dict[str, str] = {}
    in_recep: dict[str, set] = {}
    holds: set = set()
    opened: set = set()
    checked: set = set()
    obj_props: dict[str, dict] = {}
    agent_loc = None
    for name, args in tuples:
        n = name.lower()
        if n == 'objecttype' and len(args) == 2:
            obj_type[args[0]] = args[1].replace('type', '').strip() or args[1]
        elif n == 'receptacletype' and len(args) == 2:
            recep_type[args[0]] = args[1].replace('type', '').strip() or args[1]
        elif n == 'inreceptacle' and len(args) == 2:
            in_recep.setdefault(args[1], set()).add(args[0])
        elif n == 'holds' and len(args) == 2:
            holds.add(args[1])
        elif n == 'opened' and len(args) == 1:
            opened.add(args[0])
        elif n == 'checked' and len(args) == 1:
            checked.add(args[0])
        elif n == 'atlocation' and len(args) == 2 and args[0].startswith('agent'):
            agent_loc = args[1]
        elif n in _OBJ_STATE_PROP and len(args) == 1:
            obj_props.setdefault(args[0], {})[_OBJ_STATE_PROP[n]] = True
    return {'obj_type': obj_type, 'recep_type': recep_type, 'in_recep': in_recep, 'holds': holds, 'opened': opened, 'checked': checked, 'obj_props': obj_props, 'agent_loc': agent_loc, 'receptacles': set(recep_type), 'objects': set(obj_type)}

def _obj_meta(oid: str, p: dict) -> dict:
    props = dict(p['obj_props'].get(oid, {}))
    return {'type': p['obj_type'].get(oid, ''), 'props': props}

def _recep_obj_meta(rid: str, p: dict, reveal_state: bool) -> dict:
    props: dict = {}
    if reveal_state:
        if rid in p['opened']:
            props['open'] = True
        elif rid in p['recep_type']:
            props['open'] = False
        if rid in p['checked']:
            props['visited'] = True
    return {'type': p['recep_type'].get(rid, ''), 'props': props}

def build_full_gold(facts) -> dict:
    p = parse_facts(facts)
    objects: dict = {}
    locations: dict = {}
    for oid in p['objects']:
        objects[oid] = _obj_meta(oid, p)
    for rid in p['receptacles']:
        objects[rid] = _recep_obj_meta(rid, p, reveal_state=True)
        contents = sorted(p['in_recep'].get(rid, set()) - p['holds'])
        locations[rid] = {'type': p['recep_type'][rid], 'contents': contents}
    relations = []
    if p['agent_loc'] is not None:
        relations.append({'subj': 'agent', 'rel': 'at', 'obj': p['agent_loc']})
    inventory = sorted(p['holds'])
    ws = {'objects': objects, 'locations': locations, 'relations': relations, 'inventory': inventory, 'open_subgoals': [], 'completed_subgoals': [], 'blocked_dependencies': [], 'beliefs': []}
    return canonicalize_world_state(ws)

def build_partial_obs(facts, observed_receptacles: set, object_loc_memory: dict | None=None) -> dict:
    p = parse_facts(facts)
    objects: dict = {}
    locations: dict = {}
    sticky = object_loc_memory is not None
    if sticky:
        held = set(p['holds'])
        currently_visible_obj_loc: dict = {}
        for rid in p['receptacles']:
            if rid in observed_receptacles and (rid in p['opened'] or rid not in _openable(p)):
                for oid in p['in_recep'].get(rid, set()) - held:
                    currently_visible_obj_loc[oid] = rid
        for oid, rid in currently_visible_obj_loc.items():
            object_loc_memory[oid] = rid
        for oid in held:
            object_loc_memory.pop(oid, None)
    objects: dict = {}
    locations: dict = {}
    for rid in p['receptacles']:
        observed = rid in observed_receptacles
        objects[rid] = _recep_obj_meta(rid, p, reveal_state=observed)
        if observed and (rid in p['opened'] or rid not in _openable(p)):
            contents = sorted(p['in_recep'].get(rid, set()) - p['holds'])
            locations[rid] = {'type': p['recep_type'][rid], 'contents': contents}
            for oid in contents:
                objects[oid] = _obj_meta(oid, p)
        else:
            locations[rid] = {'type': p['recep_type'][rid], 'contents': []}
    if sticky:
        for oid, rid in sorted(object_loc_memory.items()):
            if rid not in locations:
                locations[rid] = {'type': p['recep_type'].get(rid, 'receptacle'), 'contents': []}
            if oid not in locations[rid]['contents']:
                locations[rid]['contents'] = sorted(set(locations[rid]['contents']) | {oid})
            if oid not in objects:
                objects[oid] = _obj_meta(oid, p)
    for oid in sorted(p['holds']):
        objects[oid] = _obj_meta(oid, p)
    relations = []
    if p['agent_loc'] is not None:
        relations.append({'subj': 'agent', 'rel': 'at', 'obj': p['agent_loc']})
    ws = {'objects': objects, 'locations': locations, 'relations': relations, 'inventory': sorted(p['holds']), 'open_subgoals': [], 'completed_subgoals': [], 'blocked_dependencies': [], 'beliefs': []}
    return canonicalize_world_state(ws)

def _openable(p: dict) -> set:
    OPENABLE_TYPES = {'Fridge', 'Microwave', 'Cabinet', 'Drawer', 'Safe', 'Box'}
    out = set(p['opened'])
    for rid, rtype in p['recep_type'].items():
        if rtype in OPENABLE_TYPES:
            out.add(rid)
    return out

def compute_sc(facts) -> dict:
    p = parse_facts(facts)
    full = build_full_gold(facts)
    from src.evaluation.runner import world_state_facts
    n_facts = len(world_state_facts(full))
    n_objects = len(p['objects'])
    n_receptacles = len(p['receptacles'])
    n_in_recep = sum((len(v) for v in p['in_recep'].values()))
    return {'sc_facts': n_facts, 'n_objects': n_objects, 'n_receptacles': n_receptacles, 'n_in_receptacle': n_in_recep, 'sc_entities': n_objects + n_receptacles}
