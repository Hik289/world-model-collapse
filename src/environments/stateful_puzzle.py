"""Stateful Puzzle environment (EXP_PLAN §2.6).

state_card → rooms + containers + items + subgoals.
dep_density → number of preconditions per subgoal.
"""

from __future__ import annotations

import random
from typing import Any

from .base import (
    EnvMeta,
    Environment,
    Observation,
    StepResult,
    ValidityCheck,
    canonicalize_world_state,
    empty_world_state,
)


# state_card → (rooms, containers, items, subgoals)
STATE_CARD_TABLE = {
    # sc=11..19 added 2026-06-06 for Exp-SC-Fine (Zekun-authorized).
    # Linear interpolation between sc=10 (3,3,4,3) and sc=20 (5,6,9,4), rounded to ints.
    # Monotonic non-decreasing; distinct from endpoints (no exact duplicates of sc=10/20).
    5: (2, 1, 2, 2),
    10: (3, 3, 4, 3),
    11: (3, 3, 4, 3),   # rounded; same shape as sc=10 but distinct enum (smallest jump)
    12: (3, 4, 5, 3),
    13: (4, 4, 5, 3),
    14: (4, 4, 6, 3),
    15: (4, 5, 6, 4),   # midpoint
    16: (4, 5, 7, 4),
    17: (4, 5, 7, 4),
    18: (5, 5, 8, 4),
    19: (5, 6, 8, 4),   # close to sc=20
    20: (5, 6, 9, 4),
    40: (8, 12, 20, 6),
}


class StatefulPuzzleEnv(Environment):
    name = "stateful_puzzle"

    def __init__(self) -> None:
        self._task_config: dict = {}
        self._stress: dict = {}
        self._rng: random.Random = random.Random(0)
        self._mut_rng: random.Random = random.Random(0)
        self._obs_rng: random.Random = random.Random(0)

        self._rooms: list[str] = []
        self._room_adj: dict[str, list[str]] = {}
        self._containers: dict[str, dict] = {}  # ctr_id -> {location, open, contents}
        self._items: dict[str, dict] = {}  # item_id -> {location, kind} ; location is room id, ctr id, or "inventory"
        self._switches: dict[str, dict] = {}  # sw -> {location, on}
        self._subgoals: dict[str, dict] = {}  # sg -> {preconds: [...]}
        self._open_subgoals: list[str] = []
        self._completed_subgoals: list[str] = []
        self._inventory: list[str] = []
        self._current_room: str = ""
        self._t: int = 0
        self._t_max: int = 0
        self._action_templates: list[str] = []
        self._object_ids: list[str] = []

    # -------------------------------------------------------------------
    def reset(self, task_config: dict, seed: int) -> Observation:
        self._task_config = dict(task_config)
        self._stress = dict(task_config.get("stress_config") or {})
        self._rng = random.Random(f"{seed}:topology")
        self._mut_rng = random.Random(f"{seed}:mutation")
        self._obs_rng = random.Random(f"{seed}:observation")

        state_card = int(self._stress.get("state_card", 5))
        if state_card not in STATE_CARD_TABLE:
            raise ValueError(f"stateful_puzzle: unknown state_card={state_card}")
        n_rooms, n_ctrs, n_items, n_subgoals = STATE_CARD_TABLE[state_card]

        dep_density = int(self._stress.get("dep_density", 1))
        self._t_max = int(self._stress.get("T", 40))
        self._t = 0
        self._inventory = []
        self._completed_subgoals = []

        # Rooms in a path graph (deterministic)
        self._rooms = [f"room_{i}" for i in range(n_rooms)]
        self._room_adj = {r: [] for r in self._rooms}
        for i in range(n_rooms - 1):
            self._room_adj[self._rooms[i]].append(self._rooms[i + 1])
            self._room_adj[self._rooms[i + 1]].append(self._rooms[i])
        # Add a small number of shortcut edges deterministically
        extras = max(0, n_rooms // 4)
        for _ in range(extras):
            a, b = self._rng.sample(self._rooms, 2)
            if b not in self._room_adj[a]:
                self._room_adj[a].append(b)
                self._room_adj[b].append(a)
        for r in self._rooms:
            self._room_adj[r] = sorted(self._room_adj[r])

        # Containers placed in rooms
        self._containers = {}
        for i in range(n_ctrs):
            r = self._rooms[self._rng.randrange(len(self._rooms))]
            self._containers[f"ctr_{i}"] = {"location": r, "open": False, "contents": []}

        # Items placed in rooms or in containers
        self._items = {}
        ctr_keys = sorted(self._containers.keys())
        for i in range(n_items):
            if ctr_keys and self._rng.random() < 0.5:
                c = self._rng.choice(ctr_keys)
                loc = c
                self._containers[c]["contents"].append(f"item_{i}")
            else:
                loc = self._rooms[self._rng.randrange(len(self._rooms))]
            self._items[f"item_{i}"] = {"location": loc, "kind": "object"}

        # Sort container contents
        for c in self._containers.values():
            c["contents"].sort()

        # Switches (1 per 5 rooms, at least 1 if rooms exist)
        n_sw = max(1, n_rooms // 5)
        self._switches = {}
        for i in range(n_sw):
            r = self._rooms[self._rng.randrange(len(self._rooms))]
            self._switches[f"sw_{i}"] = {"location": r, "on": False}

        # Subgoals: each needs `dep_density` preconditions drawn from primitive pools
        self._subgoals = {}
        sg_pool: list[str] = []
        sg_pool.extend([f"hold({i})" for i in self._items])
        sg_pool.extend([f"open({c})" for c in self._containers])
        sg_pool.extend([f"switch_on({s})" for s in self._switches])
        sg_pool.extend([f"visited({r})" for r in self._rooms])
        for i in range(n_subgoals):
            k = min(dep_density, len(sg_pool))
            preconds = self._rng.sample(sg_pool, k) if k else []
            self._subgoals[f"sg_{i}"] = {"preconds": sorted(preconds)}
        self._open_subgoals = sorted(self._subgoals.keys())

        # Start room
        self._current_room = self._rooms[0]
        # Track visited (used by visited(r) preconds)
        self._visited: set[str] = {self._current_room}

        # Action templates
        self._object_ids = sorted(
            list(self._rooms)
            + list(self._containers.keys())
            + list(self._items.keys())
            + list(self._switches.keys())
            + list(self._subgoals.keys())
        )
        templates: list[str] = []
        for r in self._rooms:
            templates.append(f"go({r})")
        for it in self._items:
            templates.append(f"take({it})")
        for c in self._containers:
            templates.append(f"open({c})")
            templates.append(f"close({c})")
        for s in self._switches:
            templates.append(f"activate({s})")
        for sg in self._subgoals:
            templates.append(f"finish_subgoal({sg})")
        templates.append("noop")
        self._action_templates = sorted(templates)

        return self._make_observation(last_action=None, valid=None)

    # -------------------------------------------------------------------
    def step(self, action: str) -> StepResult:
        self._t += 1
        validity = self.check_action_validity(action)
        info: dict[str, Any] = {"validity": validity.to_dict()}
        if validity.valid:
            self._apply_action(action)
            info["applied"] = True
        else:
            info["applied"] = False

        self._apply_mutation()

        done = (not self._open_subgoals) or (self._t >= self._t_max)
        reward = 1.0 if (not self._open_subgoals) else 0.0
        obs = self._make_observation(last_action=action, valid=validity.valid)
        obs.done = done
        return StepResult(observation=obs, reward=reward, done=done, info=info)

    # -------------------------------------------------------------------
    def get_gold_state(self) -> dict:
        ws = empty_world_state()

        for r in self._rooms:
            ws["objects"][r] = {"type": "room", "props": {}}
        for c, meta in self._containers.items():
            ws["objects"][c] = {
                "type": "container",
                "props": {"location": meta["location"], "open": meta["open"]},
            }
        for it, meta in self._items.items():
            ws["objects"][it] = {
                "type": "item",
                "props": {"location": meta["location"]},
            }
        for s, meta in self._switches.items():
            ws["objects"][s] = {
                "type": "switch",
                "props": {"location": meta["location"], "on": meta["on"]},
            }
        for sg, meta in self._subgoals.items():
            ws["objects"][sg] = {
                "type": "subgoal",
                "props": {"preconds_str": ";".join(meta["preconds"])},
            }

        # Locations
        for r in self._rooms:
            contents: list[str] = []
            for c, m in self._containers.items():
                if m["location"] == r:
                    contents.append(c)
            for it, m in self._items.items():
                if m["location"] == r:
                    contents.append(it)
            for s, m in self._switches.items():
                if m["location"] == r:
                    contents.append(s)
            ws["locations"][r] = {"type": "room", "contents": contents}

        # Container contents also surfaced as locations (so agent can inspect them)
        for c, m in self._containers.items():
            ws["locations"][c] = {"type": "container", "contents": list(m["contents"])}

        # Relations: room adjacency + agent at
        for r, nbrs in self._room_adj.items():
            for m in nbrs:
                if r < m:
                    ws["relations"].append({"subj": r, "rel": "adjacent", "obj": m})
        ws["relations"].append({"subj": "agent", "rel": "at", "obj": self._current_room})

        ws["inventory"] = list(self._inventory)
        ws["open_subgoals"] = list(self._open_subgoals)
        ws["completed_subgoals"] = list(self._completed_subgoals)

        # Blocked deps
        for sg, meta in self._subgoals.items():
            if sg in self._completed_subgoals:
                continue
            missing = [p for p in meta["preconds"] if not self._precond_holds(p)]
            if missing:
                ws["blocked_dependencies"].append({"action": f"finish_subgoal({sg})", "missing": missing})

        return canonicalize_world_state(ws)

    # -------------------------------------------------------------------
    def check_action_validity(self, action: str) -> ValidityCheck:
        verb, args = self._parse(action)
        if verb is None:
            return ValidityCheck(valid=False, missing=[], reason="parse_error")

        if verb == "noop":
            return ValidityCheck(valid=True)

        if verb == "go":
            if not args or args[0] not in self._rooms:
                return ValidityCheck(valid=False, missing=[], reason="unknown_room")
            if args[0] not in self._room_adj.get(self._current_room, []):
                return ValidityCheck(valid=False, missing=[], reason="not_adjacent")
            return ValidityCheck(valid=True)

        if verb == "take":
            if not args or args[0] not in self._items:
                return ValidityCheck(valid=False, missing=[], reason="unknown_item")
            it = args[0]
            loc = self._items[it]["location"]
            if loc == "inventory":
                return ValidityCheck(valid=False, missing=[], reason="already_held")
            # Item is in current room directly, or in an open container in current room
            if loc == self._current_room:
                return ValidityCheck(valid=True)
            if loc in self._containers:
                c = self._containers[loc]
                if c["location"] == self._current_room and c["open"]:
                    return ValidityCheck(valid=True)
                if c["location"] != self._current_room:
                    return ValidityCheck(valid=False, missing=[], reason="wrong_room")
                if not c["open"]:
                    return ValidityCheck(valid=False, missing=[loc], reason="container_closed")
            return ValidityCheck(valid=False, missing=[], reason="wrong_room")

        if verb in ("open", "close"):
            if not args or args[0] not in self._containers:
                return ValidityCheck(valid=False, missing=[], reason="unknown_container")
            c = args[0]
            if self._containers[c]["location"] != self._current_room:
                return ValidityCheck(valid=False, missing=[], reason="wrong_room")
            want_open = verb == "open"
            if self._containers[c]["open"] == want_open:
                return ValidityCheck(valid=False, missing=[], reason="already_in_state")
            return ValidityCheck(valid=True)

        if verb == "activate":
            if not args or args[0] not in self._switches:
                return ValidityCheck(valid=False, missing=[], reason="unknown_switch")
            s = args[0]
            if self._switches[s]["location"] != self._current_room:
                return ValidityCheck(valid=False, missing=[], reason="wrong_room")
            return ValidityCheck(valid=True)

        if verb == "finish_subgoal":
            if not args or args[0] not in self._subgoals:
                return ValidityCheck(valid=False, missing=[], reason="unknown_subgoal")
            sg = args[0]
            if sg in self._completed_subgoals:
                return ValidityCheck(valid=False, missing=[], reason="already_completed")
            missing = [p for p in self._subgoals[sg]["preconds"] if not self._precond_holds(p)]
            if missing:
                return ValidityCheck(valid=False, missing=sorted(missing), reason="precondition_violation")
            return ValidityCheck(valid=True)

        return ValidityCheck(valid=False, missing=[], reason="parse_error")

    # -------------------------------------------------------------------
    def compute_error_labels(self, agent_state: dict, action: str) -> list[str]:
        labels: list[str] = []
        verb, args = self._parse(action)
        if verb is None:
            return ["parse_error"]

        validity = self.check_action_validity(action)
        if not validity.valid:
            if validity.reason == "precondition_violation":
                labels.append("precondition_violation")
            if validity.reason in ("wrong_room", "container_closed"):
                labels.append("object_location_error")
            if validity.reason == "already_completed":
                labels.append("repeated_completed_action")
            if validity.reason == "parse_error":
                labels.append("parse_error")
            if validity.reason in ("already_in_state",):
                labels.append("container_state_error")

        # stale_room_state proxy: agent_state objects' location for current room differs from real
        try:
            ag_objs = agent_state.get("objects", {}) or {}
            cur = self._current_room
            ag_at = ""
            for r in (agent_state.get("relations") or []):
                if r.get("subj") == "agent" and r.get("rel") == "at":
                    ag_at = r.get("obj", "")
            if ag_at and ag_at != cur:
                labels.append("stale_room_state")
        except (AttributeError, TypeError):
            pass

        # incorrect_inventory
        try:
            ag_inv = sorted(agent_state.get("inventory", []) or [])
            if ag_inv != sorted(self._inventory):
                labels.append("incorrect_inventory")
        except (AttributeError, TypeError):
            pass

        # goal_drift: agent declared subgoal not real
        try:
            ag_done = set(agent_state.get("completed_subgoals", []) or [])
            real_done = set(self._completed_subgoals)
            if ag_done - real_done:
                labels.append("goal_drift")
        except (AttributeError, TypeError):
            pass

        return sorted(set(labels))

    # -------------------------------------------------------------------
    def get_meta(self) -> EnvMeta:
        return EnvMeta(
            name=self.name,
            stress_config=dict(self._stress),
            task_config=dict(self._task_config),
            action_templates=list(self._action_templates),
            object_ids=list(self._object_ids),
            extra={
                "rooms": list(self._rooms),
                "containers": sorted(self._containers.keys()),
                "items": sorted(self._items.keys()),
                "switches": sorted(self._switches.keys()),
                "subgoals": sorted(self._subgoals.keys()),
            },
        )

    # -------------------------------------------------------------------
    def _parse(self, action: str) -> tuple[str | None, list[str]]:
        if not isinstance(action, str):
            return None, []
        a = action.strip()
        if a == "noop":
            return "noop", []
        if "(" not in a or not a.endswith(")"):
            return None, []
        verb, rest = a.split("(", 1)
        verb = verb.strip()
        inside = rest[:-1].strip()
        args = [x.strip() for x in inside.split(",")] if inside else []
        return verb, args

    def _precond_holds(self, p: str) -> bool:
        if p.startswith("hold(") and p.endswith(")"):
            return p[5:-1] in self._inventory
        if p.startswith("open(") and p.endswith(")"):
            c = p[5:-1]
            return self._containers.get(c, {}).get("open", False)
        if p.startswith("switch_on(") and p.endswith(")"):
            s = p[10:-1]
            return self._switches.get(s, {}).get("on", False)
        if p.startswith("visited(") and p.endswith(")"):
            return p[8:-1] in self._visited
        return False

    def _apply_action(self, action: str) -> bool:
        verb, args = self._parse(action)
        if verb is None:
            return False
        if verb == "noop":
            return True
        if verb == "go":
            self._current_room = args[0]
            self._visited.add(self._current_room)
            return True
        if verb == "take":
            it = args[0]
            old_loc = self._items[it]["location"]
            if old_loc in self._containers:
                self._containers[old_loc]["contents"] = sorted(
                    [x for x in self._containers[old_loc]["contents"] if x != it]
                )
            self._items[it]["location"] = "inventory"
            self._inventory.append(it)
            self._inventory.sort()
            return True
        if verb == "open":
            self._containers[args[0]]["open"] = True
            return True
        if verb == "close":
            self._containers[args[0]]["open"] = False
            return True
        if verb == "activate":
            self._switches[args[0]]["on"] = not self._switches[args[0]]["on"]
            return True
        if verb == "finish_subgoal":
            sg = args[0]
            self._completed_subgoals.append(sg)
            self._completed_subgoals.sort()
            self._open_subgoals = sorted([g for g in self._open_subgoals if g != sg])
            return True
        return False

    def _apply_mutation(self) -> None:
        mut = self._stress.get("mut_rate", "static")
        if mut == "static":
            return
        if mut == "low":
            if self._mut_rng.random() < 0.20 and self._switches:
                s = self._mut_rng.choice(sorted(self._switches.keys()))
                self._switches[s]["on"] = not self._switches[s]["on"]
            return
        if mut == "medium":
            if self._mut_rng.random() < 0.20 and self._switches:
                s = self._mut_rng.choice(sorted(self._switches.keys()))
                self._switches[s]["on"] = not self._switches[s]["on"]
            if self._mut_rng.random() < 0.05 and self._containers:
                c = self._mut_rng.choice(sorted(self._containers.keys()))
                self._containers[c]["open"] = not self._containers[c]["open"]
            return
        if mut == "high":
            options = []
            if self._switches:
                options.append("switch")
            if self._containers:
                options.append("container")
            if options:
                kind = self._mut_rng.choice(sorted(options))
                if kind == "switch":
                    s = self._mut_rng.choice(sorted(self._switches.keys()))
                    self._switches[s]["on"] = not self._switches[s]["on"]
                else:
                    c = self._mut_rng.choice(sorted(self._containers.keys()))
                    self._containers[c]["open"] = not self._containers[c]["open"]

    def _make_observation(self, last_action: str | None, valid: bool | None) -> Observation:
        obs_mode = self._stress.get("obs_noise", "clean")
        gold = self.get_gold_state()
        if obs_mode == "clean":
            partial = gold
        else:
            partial = self._build_partial_obs(gold, obs_mode)
        text_parts = [
            f"In {self._current_room}.",
            f"Inventory: {sorted(self._inventory)}.",
            f"Open subgoals: {sorted(self._open_subgoals)}.",
        ]
        if last_action is not None:
            text_parts.append(f"Last action: {last_action} (valid={valid}).")
        text = " ".join(text_parts)
        return Observation(text=text, partial_state=partial, done=False, info={"step": self._t})

    def _build_partial_obs(self, gold: dict, mode: str) -> dict:
        partial = empty_world_state()
        cur = self._current_room
        partial["locations"][cur] = gold["locations"].get(cur, {"type": "room", "contents": []})
        # Objects visible: room itself and its contents
        for oid in [cur] + list(gold["locations"].get(cur, {}).get("contents", [])):
            partial["objects"][oid] = gold["objects"].get(oid, {"type": "unknown", "props": {}})
        partial["inventory"] = list(gold["inventory"])
        partial["open_subgoals"] = list(gold["open_subgoals"])
        partial["completed_subgoals"] = list(gold["completed_subgoals"])
        partial["relations"].append({"subj": "agent", "rel": "at", "obj": cur})

        if mode == "partial":
            return canonicalize_world_state(partial)

        if mode in ("distractor", "conflict") and self._obs_rng.random() < 0.30 and self._rooms:
            other = self._obs_rng.choice([r for r in self._rooms if r != cur] or [cur])
            partial["locations"][other] = gold["locations"].get(other, {"type": "room", "contents": []})
            partial["objects"][other] = gold["objects"].get(other, {"type": "room", "props": {}})

        if mode == "conflict" and self._obs_rng.random() < 0.10 and self._items:
            # Lie: place a non-here item into current room's contents
            fake_item = self._obs_rng.choice(sorted(self._items.keys()))
            if self._items[fake_item]["location"] not in (cur, "inventory"):
                contents = list(partial["locations"][cur]["contents"])
                if fake_item not in contents:
                    contents.append(fake_item)
                    partial["locations"][cur] = {"type": "room", "contents": sorted(contents)}

        return canonicalize_world_state(partial)


GRAPH_NAV_STATE_CARD_TABLE = {
    5: (5, 1, 1, 1, 0.0),
    10: (10, 2, 2, 2, 0.20),
    20: (20, 4, 4, 4, 0.35),
    40: (40, 8, 6, 6, 0.50),
}

GRAPH_NAV_OBS_NOISE_MODES = {"clean", "partial", "distractor", "conflict"}
GRAPH_NAV_MUT_RATE_MODES = {"static", "low", "medium", "high"}


class GraphNavEnv(Environment):
    name = "graph_nav"


    def __init__(self) -> None:
        self._task_config: dict = {}
        self._stress: dict = {}
        self._rng: random.Random = random.Random(0)
        self._mut_rng: random.Random = random.Random(0)
        self._obs_rng: random.Random = random.Random(0)
        self._nodes: list[str] = []
        self._edges: dict[str, list[str]] = {}
        self._decoys: set[str] = set()
        self._keys: dict[str, str] = {}
        self._switches: dict[str, str] = {}
        self._switch_state: dict[str, bool] = {}
        self._doors: dict[str, dict] = {}
        self._current_node: str = ""
        self._inventory: list[str] = []
        self._visited: set[str] = set()
        self._goal_node: str = ""
        self._completed_subgoals: list[str] = []
        self._open_subgoals: list[str] = []
        self._t: int = 0
        self._t_max: int = 0
        self._action_templates: list[str] = []
        self._object_ids: list[str] = []


    def reset(self, task_config: dict, seed: int) -> Observation:
        self._task_config = dict(task_config)
        self._stress = dict(task_config.get("stress_config") or {})

        self._rng = random.Random(f"{seed}:topology")
        self._mut_rng = random.Random(f"{seed}:mutation")
        self._obs_rng = random.Random(f"{seed}:observation")

        state_card = int(self._stress.get("state_card", 5))
        if state_card not in GRAPH_NAV_STATE_CARD_TABLE:
            raise ValueError(f"graph_nav: unknown state_card={state_card}")
        n_nodes, n_keys, n_switches, n_doors, decoy_pct = GRAPH_NAV_STATE_CARD_TABLE[state_card]

        dep_density = int(self._stress.get("dep_density", 1))
        self._t_max = int(self._stress.get("T", 40))
        self._t = 0
        self._inventory = []
        self._visited = set()
        self._completed_subgoals = []
        self._switch_state = {}


        self._nodes = [f"n{i}" for i in range(n_nodes)]


        self._edges = {n: [] for n in self._nodes}
        order = list(self._nodes)
        self._rng.shuffle(order)
        for i in range(1, len(order)):
            parent = order[self._rng.randrange(i)]
            self._edges[parent].append(order[i])
            self._edges[order[i]].append(parent)

        extra_edges = max(0, n_nodes // 4)
        for _ in range(extra_edges):
            a, b = self._rng.sample(self._nodes, 2)
            if b not in self._edges[a]:
                self._edges[a].append(b)
                self._edges[b].append(a)

        self._edges = {n: sorted(self._edges[n]) for n in self._nodes}


        n_decoy = int(round(n_nodes * decoy_pct))
        decoy_nodes = self._rng.sample(self._nodes, n_decoy) if n_decoy else []
        self._decoys = set(decoy_nodes)


        non_decoy = [n for n in self._nodes if n not in self._decoys] or list(self._nodes)
        self._keys = {f"key_{i}": non_decoy[self._rng.randrange(len(non_decoy))] for i in range(n_keys)}
        self._switches = {f"sw_{i}": non_decoy[self._rng.randrange(len(non_decoy))] for i in range(n_switches)}
        for sw in self._switches:
            self._switch_state[sw] = False


        self._doors = {}
        for i in range(n_doors):

            a, b = self._rng.sample(self._nodes, 2)
            preconds: list[str] = []
            pool: list[str] = []
            pool.extend([f"hold({k})" for k in self._keys])
            pool.extend([f"switch_on({s})" for s in self._switches])
            pool.extend([f"visited({n})" for n in self._nodes])

            k = min(dep_density, len(pool))
            preconds = self._rng.sample(pool, k)
            self._doors[f"door_{i}"] = {
                "between": tuple(sorted([a, b])),
                "preconds": sorted(preconds),
                "unlocked": False,
            }


        self._current_node = non_decoy[0]

        self._goal_node = non_decoy[-1]
        self._open_subgoals = [f"reach({self._goal_node})"]


        self._object_ids = list(self._nodes) + list(self._keys) + list(self._switches) + list(self._doors)
        self._object_ids = sorted(self._object_ids)

        templates: list[str] = []
        for n in self._nodes:
            templates.append(f"move({n})")
        for k in self._keys:
            templates.append(f"pick({k})")
            templates.append(f"drop({k})")
        for s in self._switches:
            templates.append(f"turn_on({s})")
            templates.append(f"turn_off({s})")
        for d in self._doors:
            templates.append(f"unlock({d})")
        for o in self._object_ids:
            templates.append(f"inspect({o})")
        templates.append("noop")
        self._action_templates = sorted(templates)


        self._visited.add(self._current_node)

        return self._make_observation(last_action=None, valid=None)


    def step(self, action: str) -> StepResult:
        self._t += 1
        validity = self.check_action_validity(action)
        applied_ok = False
        info: dict[str, Any] = {"validity": validity.to_dict()}

        if validity.valid:
            applied_ok = self._apply_action(action)
            info["applied"] = bool(applied_ok)
        else:
            info["applied"] = False


        self._apply_mutation()


        if self._current_node == self._goal_node and "reach(" + self._goal_node + ")" not in self._completed_subgoals:
            self._completed_subgoals.append(f"reach({self._goal_node})")
            self._open_subgoals = [g for g in self._open_subgoals if g != f"reach({self._goal_node})"]

        done = (not self._open_subgoals) or (self._t >= self._t_max)
        reward = 1.0 if (not self._open_subgoals) else 0.0
        obs = self._make_observation(last_action=action, valid=validity.valid)
        obs.done = done

        return StepResult(observation=obs, reward=reward, done=done, info=info)


    def get_gold_state(self) -> dict:
        ws = empty_world_state()


        for n in self._nodes:
            ws["objects"][n] = {
                "type": "node",
                "props": {"decoy": n in self._decoys, "is_goal": n == self._goal_node},
            }
        for k, loc in self._keys.items():
            held = k in self._inventory
            ws["objects"][k] = {
                "type": "key",
                "props": {"location": loc if not held else "inventory", "held": held},
            }
        for s in self._switches:
            ws["objects"][s] = {
                "type": "switch",
                "props": {"location": self._switches[s], "on": self._switch_state[s]},
            }
        for d, meta in self._doors.items():
            ws["objects"][d] = {
                "type": "door",
                "props": {
                    "between_a": meta["between"][0],
                    "between_b": meta["between"][1],
                    "unlocked": meta["unlocked"],
                },
            }


        for n in self._nodes:
            contents: list[str] = []
            for k, loc in self._keys.items():
                if loc == n and k not in self._inventory:
                    contents.append(k)
            for s, loc in self._switches.items():
                if loc == n:
                    contents.append(s)
            ws["locations"][n] = {"type": "node", "contents": contents}


        for n, nbrs in self._edges.items():
            for m in nbrs:
                if n < m:
                    ws["relations"].append({"subj": n, "rel": "adjacent", "obj": m})
        ws["relations"].append({"subj": "agent", "rel": "at", "obj": self._current_node})


        ws["inventory"] = list(self._inventory)


        ws["open_subgoals"] = list(self._open_subgoals)
        ws["completed_subgoals"] = list(self._completed_subgoals)


        for d, meta in self._doors.items():
            if not meta["unlocked"]:
                missing = [p for p in meta["preconds"] if not self._precond_holds(p)]
                if missing:
                    ws["blocked_dependencies"].append({"action": f"unlock({d})", "missing": missing})


        return canonicalize_world_state(ws)


    def check_action_validity(self, action: str) -> ValidityCheck:
        verb, args = self._parse(action)
        if verb is None:
            return ValidityCheck(valid=False, missing=[], reason="parse_error")

        if verb == "noop":
            return ValidityCheck(valid=True)

        if verb == "move":
            if not args or args[0] not in self._nodes:
                return ValidityCheck(valid=False, missing=[args[0] if args else ""], reason="unknown_node")
            target = args[0]
            if target not in self._edges.get(self._current_node, []):

                for d, meta in self._doors.items():
                    if tuple(sorted([self._current_node, target])) == meta["between"]:
                        if not meta["unlocked"]:
                            return ValidityCheck(valid=False, missing=[d], reason="door_locked")
                        return ValidityCheck(valid=True)
                return ValidityCheck(valid=False, missing=[], reason="nonexistent_edge")
            return ValidityCheck(valid=True)

        if verb == "pick":
            if not args or args[0] not in self._keys:
                return ValidityCheck(valid=False, missing=[], reason="unknown_key")
            k = args[0]
            if k in self._inventory:
                return ValidityCheck(valid=False, missing=[], reason="already_held")
            if self._keys[k] != self._current_node:
                return ValidityCheck(valid=False, missing=[], reason="key_not_here")
            return ValidityCheck(valid=True)

        if verb == "drop":
            if not args or args[0] not in self._inventory:
                return ValidityCheck(valid=False, missing=[], reason="not_holding")
            return ValidityCheck(valid=True)

        if verb in ("turn_on", "turn_off"):
            if not args or args[0] not in self._switches:
                return ValidityCheck(valid=False, missing=[], reason="unknown_switch")
            sw = args[0]
            if self._switches[sw] != self._current_node:
                return ValidityCheck(valid=False, missing=[], reason="switch_not_here")
            want = verb == "turn_on"
            if self._switch_state[sw] == want:
                return ValidityCheck(valid=False, missing=[], reason="already_in_state")
            return ValidityCheck(valid=True)

        if verb == "unlock":
            if not args or args[0] not in self._doors:
                return ValidityCheck(valid=False, missing=[], reason="unknown_door")
            d = args[0]
            if self._doors[d]["unlocked"]:
                return ValidityCheck(valid=False, missing=[], reason="already_unlocked")
            missing = [p for p in self._doors[d]["preconds"] if not self._precond_holds(p)]
            if missing:
                return ValidityCheck(valid=False, missing=sorted(missing), reason="missing_precondition")

            if self._current_node not in self._doors[d]["between"]:
                return ValidityCheck(valid=False, missing=[], reason="not_at_door")
            return ValidityCheck(valid=True)

        if verb == "inspect":
            if not args or args[0] not in self._object_ids:
                return ValidityCheck(valid=False, missing=[], reason="unknown_object")
            return ValidityCheck(valid=True)

        return ValidityCheck(valid=False, missing=[], reason="parse_error")


    def compute_error_labels(self, agent_state: dict, action: str) -> list[str]:
        labels: list[str] = []
        verb, args = self._parse(action)
        if verb is None:
            return ["parse_error"]

        validity = self.check_action_validity(action)
        if not validity.valid:
            if validity.reason == "nonexistent_edge":
                labels.append("nonexistent_edge")
            if validity.reason in ("door_locked", "missing_precondition"):
                labels.append("missing_key")
            if validity.reason in ("key_not_here", "switch_not_here", "not_at_door"):
                labels.append("wrong_current_node")
            if validity.reason == "parse_error":
                labels.append("parse_error")


        if verb == "move" and args and args[0] in self._decoys:
            labels.append("decoy_pursuit")


        if verb == "move" and args and args[0] in self._visited:
            labels.append("revisited_loop")


        try:
            ag_inv = sorted(agent_state.get("inventory", []) or [])
            if ag_inv != sorted(self._inventory):
                labels.append("stale_inventory")
        except (AttributeError, TypeError):
            pass


        try:
            ag_done = set(agent_state.get("completed_subgoals", []) or [])
            real_done = set(self._completed_subgoals)
            if ag_done - real_done:
                labels.append("false_progress")
        except (AttributeError, TypeError):
            pass

        return sorted(set(labels))


    def get_meta(self) -> EnvMeta:
        return EnvMeta(
            name=self.name,
            stress_config=dict(self._stress),
            task_config=dict(self._task_config),
            action_templates=list(self._action_templates),
            object_ids=list(self._object_ids),
            extra={
                "nodes": list(self._nodes),
                "keys": sorted(self._keys.keys()),
                "switches": sorted(self._switches.keys()),
                "doors": sorted(self._doors.keys()),
                "goal_node": self._goal_node,
            },
        )


    def _parse(self, action: str) -> tuple[str | None, list[str]]:
        if not isinstance(action, str):
            return None, []
        a = action.strip()
        if a == "noop":
            return "noop", []
        if "(" not in a or not a.endswith(")"):
            return None, []
        verb, rest = a.split("(", 1)
        verb = verb.strip()
        inside = rest[:-1].strip()
        args = [x.strip() for x in inside.split(",")] if inside else []
        return verb, args

    def _precond_holds(self, p: str) -> bool:
        if p.startswith("hold(") and p.endswith(")"):
            return p[5:-1] in self._inventory
        if p.startswith("switch_on(") and p.endswith(")"):
            sw = p[10:-1]
            return self._switch_state.get(sw, False)
        if p.startswith("visited(") and p.endswith(")"):
            return p[8:-1] in self._visited
        return False

    def _apply_action(self, action: str) -> bool:
        verb, args = self._parse(action)
        if verb is None:
            return False
        if verb == "noop":
            return True
        if verb == "move":
            self._current_node = args[0]
            self._visited.add(self._current_node)
            return True
        if verb == "pick":
            self._inventory.append(args[0])
            self._inventory.sort()
            return True
        if verb == "drop":
            self._inventory.remove(args[0])
            return True
        if verb == "turn_on":
            self._switch_state[args[0]] = True
            return True
        if verb == "turn_off":
            self._switch_state[args[0]] = False
            return True
        if verb == "unlock":
            self._doors[args[0]]["unlocked"] = True
            return True
        if verb == "inspect":
            return True
        return False

    def _apply_mutation(self) -> None:
        mut = self._stress.get("mut_rate", "static")
        if mut == "static":
            return

        if mut == "low":
            if self._mut_rng.random() < 0.20 and self._switches:
                sw = self._mut_rng.choice(sorted(self._switches.keys()))
                self._switch_state[sw] = not self._switch_state[sw]
            return
        if mut == "medium":
            if self._mut_rng.random() < 0.20 and self._switches:
                sw = self._mut_rng.choice(sorted(self._switches.keys()))
                self._switch_state[sw] = not self._switch_state[sw]
            if self._mut_rng.random() < 0.05 and self._doors:
                d = self._mut_rng.choice(sorted(self._doors.keys()))
                self._doors[d]["unlocked"] = not self._doors[d]["unlocked"]
            return
        if mut == "high":

            choices = []
            if self._switches:
                choices.append("switch")
            if self._doors:
                choices.append("door")
            if choices:
                kind = self._mut_rng.choice(sorted(choices))
                if kind == "switch":
                    sw = self._mut_rng.choice(sorted(self._switches.keys()))
                    self._switch_state[sw] = not self._switch_state[sw]
                else:
                    d = self._mut_rng.choice(sorted(self._doors.keys()))
                    self._doors[d]["unlocked"] = not self._doors[d]["unlocked"]

    def _make_observation(self, last_action: str | None, valid: bool | None) -> Observation:
        obs_mode = self._stress.get("obs_noise", "clean")
        gold = self.get_gold_state()
        if obs_mode == "clean":
            partial = gold
        else:
            partial = self._build_partial_obs(gold, obs_mode)

        text_parts = [f"At {self._current_node}.",
                      f"Inventory: {sorted(self._inventory)}.",
                      f"Open subgoals: {sorted(self._open_subgoals)}."]
        if last_action is not None:
            text_parts.append(f"Last action: {last_action} (valid={valid}).")
        text = " ".join(text_parts)
        info = {"step": self._t, "valid_actions_hint": False}
        return Observation(text=text, partial_state=partial, done=False, info=info)

    def _build_partial_obs(self, gold: dict, mode: str) -> dict:

        partial = empty_world_state()
        cur = self._current_node
        partial["objects"][cur] = gold["objects"].get(cur, {"type": "node", "props": {}})
        partial["locations"][cur] = gold["locations"].get(cur, {"type": "node", "contents": []})
        partial["inventory"] = list(gold["inventory"])
        partial["open_subgoals"] = list(gold["open_subgoals"])
        partial["completed_subgoals"] = list(gold["completed_subgoals"])

        partial["relations"].append({"subj": "agent", "rel": "at", "obj": cur})

        if mode == "partial":
            return canonicalize_world_state(partial)


        if self._obs_rng.random() < 0.30 and self._nodes:
            other = self._obs_rng.choice([n for n in self._nodes if n != cur] or [cur])
            partial["locations"][other] = gold["locations"].get(other, {"type": "node", "contents": []})
            partial["objects"][other] = gold["objects"].get(other, {"type": "node", "props": {}})

        if mode == "conflict" and self._obs_rng.random() < 0.10:

            if self._keys:
                fake_key = self._obs_rng.choice(sorted(self._keys.keys()))
                if self._keys[fake_key] != cur and fake_key not in self._inventory:
                    contents = list(partial["locations"][cur]["contents"])
                    if fake_key not in contents:
                        contents.append(fake_key)
                        partial["locations"][cur] = {"type": "node", "contents": sorted(contents)}

        return canonicalize_world_state(partial)



TOOL_DAG_STATE_CARD_TABLE = {
    5: (5, 5, 3),
    10: (10, 10, 4),
    20: (20, 20, 6),
    40: (40, 40, 8),
}


TOOL_DAG_TYPE_POOL = ["TypeA", "TypeB", "TypeC", "TypeD", "TypeE", "TypeF", "TypeG", "TypeH"]


class ToolDAGEnv(Environment):
    name = "tool_dag"

    def __init__(self) -> None:
        self._task_config: dict = {}
        self._stress: dict = {}
        self._rng: random.Random = random.Random(0)
        self._mut_rng: random.Random = random.Random(0)
        self._obs_rng: random.Random = random.Random(0)

        self._tools: dict[str, dict] = {}
        self._initial_vars: dict[str, str] = {}
        self._vars: dict[str, str] = {}
        self._called_tools: list[str] = []
        self._var_seq: int = 0
        self._target_var: str = ""
        self._target_type: str = ""
        self._t: int = 0
        self._t_max: int = 0
        self._completed_subgoals: list[str] = []
        self._open_subgoals: list[str] = []
        self._action_templates: list[str] = []
        self._object_ids: list[str] = []


    def reset(self, task_config: dict, seed: int) -> Observation:
        self._task_config = dict(task_config)
        self._stress = dict(task_config.get("stress_config") or {})
        self._rng = random.Random(f"{seed}:topology")
        self._mut_rng = random.Random(f"{seed}:mutation")
        self._obs_rng = random.Random(f"{seed}:observation")

        state_card = int(self._stress.get("state_card", 5))
        if state_card not in TOOL_DAG_STATE_CARD_TABLE:
            raise ValueError(f"tool_dag: unknown state_card={state_card}")
        n_active, n_tools, dag_depth = TOOL_DAG_STATE_CARD_TABLE[state_card]

        dep_density = int(self._stress.get("dep_density", 1))
        self._t_max = int(self._stress.get("T", 40))
        self._t = 0
        self._called_tools = []
        self._var_seq = 0
        self._completed_subgoals = []


        self._initial_vars = {}
        for i in range(n_active):
            t = TOOL_DAG_TYPE_POOL[i % len(TOOL_DAG_TYPE_POOL)]
            self._initial_vars[f"v{i}"] = t
        self._vars = dict(self._initial_vars)


        self._tools = {}

        k = min(dep_density, len(TOOL_DAG_TYPE_POOL))

        n_layers = max(1, dag_depth)
        tools_per_layer = max(1, n_tools // n_layers)
        for tool_idx in range(n_tools):
            layer = min(tool_idx // tools_per_layer, n_layers - 1)

            type_indices = list(range(len(TOOL_DAG_TYPE_POOL)))
            self._rng.shuffle(type_indices)
            chosen_types = sorted({TOOL_DAG_TYPE_POOL[type_indices[i]] for i in range(k)})

            out_type = TOOL_DAG_TYPE_POOL[(layer + 1) % len(TOOL_DAG_TYPE_POOL)]
            tool_name = f"tool_{tool_idx}"
            self._tools[tool_name] = {
                "input_types": list(chosen_types),
                "output_type": out_type,
                "layer": layer,
            }


        last_layer = n_layers - 1
        last_layer_tools = [t for t, m in self._tools.items() if m["layer"] == last_layer]
        if last_layer_tools:
            self._target_type = self._tools[last_layer_tools[0]]["output_type"]
        else:
            self._target_type = TOOL_DAG_TYPE_POOL[0]
        self._target_var = "TARGET"
        self._open_subgoals = [f"finish({self._target_var})"]


        self._object_ids = sorted(list(self._initial_vars.keys()) + list(self._tools.keys()) + [self._target_var])
        templates: list[str] = []
        for tn in sorted(self._tools.keys()):
            templates.append(f"call({tn})")
        for v in sorted(self._initial_vars.keys()):
            templates.append(f"inspect_var({v})")
        templates.append(f"finish({self._target_var})")
        templates.append("declare(NEW_VAR,VAL)")
        templates.append("noop")
        self._action_templates = sorted(templates)

        return self._make_observation(last_action=None, valid=None)


    def step(self, action: str) -> StepResult:
        self._t += 1
        validity = self.check_action_validity(action)
        info: dict[str, Any] = {"validity": validity.to_dict()}
        if validity.valid:
            self._apply_action(action)
            info["applied"] = True
        else:
            info["applied"] = False

        self._apply_mutation()

        done = (not self._open_subgoals) or (self._t >= self._t_max)
        reward = 1.0 if (not self._open_subgoals) else 0.0
        obs = self._make_observation(last_action=action, valid=validity.valid)
        obs.done = done
        return StepResult(observation=obs, reward=reward, done=done, info=info)


    def get_gold_state(self) -> dict:
        ws = empty_world_state()
        for v, t in self._vars.items():
            ws["objects"][v] = {"type": "variable", "props": {"data_type": t}}
        for tn, meta in self._tools.items():
            ws["objects"][tn] = {
                "type": "tool",
                "props": {
                    "output_type": meta["output_type"],
                    "layer": meta["layer"],
                    "input_types_str": ",".join(meta["input_types"]),
                },
            }
        ws["inventory"] = sorted(self._vars.keys())
        ws["open_subgoals"] = list(self._open_subgoals)
        ws["completed_subgoals"] = list(self._completed_subgoals)


        avail_types = sorted(set(self._vars.values()))
        for tn, meta in self._tools.items():
            missing = [t for t in meta["input_types"] if t not in avail_types]
            if missing:
                ws["blocked_dependencies"].append({"action": f"call({tn})", "missing": [f"need_type({t})" for t in missing]})

        return canonicalize_world_state(ws)


    def check_action_validity(self, action: str) -> ValidityCheck:
        verb, args = self._parse(action)
        if verb is None:
            return ValidityCheck(valid=False, missing=[], reason="parse_error")

        if verb == "noop":
            return ValidityCheck(valid=True)

        if verb == "inspect_var":
            if not args or args[0] not in self._vars:
                return ValidityCheck(valid=False, missing=[], reason="nonexistent_variable")
            return ValidityCheck(valid=True)

        if verb == "declare":
            if len(args) < 2:
                return ValidityCheck(valid=False, missing=[], reason="parse_error")
            new_name = args[0]
            if new_name in self._vars:
                return ValidityCheck(valid=False, missing=[], reason="variable_exists")
            return ValidityCheck(valid=True)

        if verb == "call":
            if not args or args[0] not in self._tools:
                return ValidityCheck(valid=False, missing=[], reason="unknown_tool")
            tn = args[0]
            meta = self._tools[tn]
            avail_types = set(self._vars.values())
            missing = [f"need_type({t})" for t in meta["input_types"] if t not in avail_types]
            if missing:
                return ValidityCheck(valid=False, missing=sorted(missing), reason="missing_argument")
            return ValidityCheck(valid=True)

        if verb == "finish":
            if not args or args[0] != self._target_var:
                return ValidityCheck(valid=False, missing=[], reason="wrong_target")
            if self._target_var not in self._vars:
                return ValidityCheck(valid=False, missing=[self._target_var], reason="target_undefined")
            if self._vars[self._target_var] != self._target_type:
                return ValidityCheck(valid=False, missing=[], reason="type_mismatch")
            return ValidityCheck(valid=True)

        return ValidityCheck(valid=False, missing=[], reason="parse_error")


    def compute_error_labels(self, agent_state: dict, action: str) -> list[str]:
        labels: list[str] = []
        verb, args = self._parse(action)
        if verb is None:
            return ["parse_error"]

        validity = self.check_action_validity(action)
        if not validity.valid:
            if validity.reason == "missing_argument":
                labels.append("missing_argument")
            if validity.reason == "type_mismatch":
                labels.append("type_mismatch")
            if validity.reason == "nonexistent_variable":
                labels.append("nonexistent_variable")
            if validity.reason == "parse_error":
                labels.append("parse_error")


        if verb == "call" and args and args[0] in self._called_tools:
            labels.append("repeated_useless_call")


        try:
            ag_inv = set(agent_state.get("inventory", []) or [])
            real_inv = set(self._vars.keys())
            if ag_inv - real_inv:
                labels.append("fabricated_tool_result")
            if real_inv - ag_inv:
                labels.append("skipped_dependency")
        except (AttributeError, TypeError):
            pass


        try:
            ag_done = set(agent_state.get("completed_subgoals", []) or [])
            real_done = set(self._completed_subgoals)
            if ag_done - real_done:
                labels.append("wrong_argument_source")
        except (AttributeError, TypeError):
            pass

        return sorted(set(labels))


    def get_meta(self) -> EnvMeta:
        return EnvMeta(
            name=self.name,
            stress_config=dict(self._stress),
            task_config=dict(self._task_config),
            action_templates=list(self._action_templates),
            object_ids=list(self._object_ids),
            extra={
                "tools": sorted(self._tools.keys()),
                "initial_vars": sorted(self._initial_vars.keys()),
                "target_var": self._target_var,
                "target_type": self._target_type,
            },
        )


    def _parse(self, action: str) -> tuple[str | None, list[str]]:
        if not isinstance(action, str):
            return None, []
        a = action.strip()
        if a == "noop":
            return "noop", []
        if "(" not in a or not a.endswith(")"):
            return None, []
        verb, rest = a.split("(", 1)
        verb = verb.strip()
        inside = rest[:-1].strip()
        args = [x.strip() for x in inside.split(",")] if inside else []
        return verb, args

    def _apply_action(self, action: str) -> bool:
        verb, args = self._parse(action)
        if verb is None:
            return False
        if verb == "noop":
            return True
        if verb == "inspect_var":
            return True
        if verb == "declare":
            new_name = args[0]
            self._vars[new_name] = args[1] if len(args) > 1 else "TypeA"
            return True
        if verb == "call":
            tn = args[0]
            self._called_tools.append(tn)
            meta = self._tools[tn]
            self._var_seq += 1
            new_var = f"r{self._var_seq}"
            self._vars[new_var] = meta["output_type"]

            if self._target_var not in self._vars and meta["output_type"] == self._target_type:
                self._vars[self._target_var] = self._target_type
            return True
        if verb == "finish":
            if f"finish({self._target_var})" not in self._completed_subgoals:
                self._completed_subgoals.append(f"finish({self._target_var})")
                self._open_subgoals = [g for g in self._open_subgoals if g != f"finish({self._target_var})"]
            return True
        return False

    def _apply_mutation(self) -> None:
        mut = self._stress.get("mut_rate", "static")
        if mut == "static":
            return

        droppable = sorted([v for v in self._vars if v not in self._initial_vars and v != self._target_var])
        if mut == "low":
            if self._mut_rng.random() < 0.20 and droppable:
                v = self._mut_rng.choice(droppable)
                self._vars.pop(v, None)
            return
        if mut == "medium":
            if self._mut_rng.random() < 0.20 and droppable:
                v = self._mut_rng.choice(droppable)
                self._vars.pop(v, None)
            if self._mut_rng.random() < 0.05 and self._tools:

                tn = self._mut_rng.choice(sorted(self._tools.keys()))
                new_type = self._mut_rng.choice(sorted(TOOL_DAG_TYPE_POOL))
                self._tools[tn]["output_type"] = new_type
            return
        if mut == "high":

            options = []
            if droppable:
                options.append("drop")
            if self._tools:
                options.append("retype")
            if options:
                kind = self._mut_rng.choice(sorted(options))
                if kind == "drop":
                    v = self._mut_rng.choice(droppable)
                    self._vars.pop(v, None)
                else:
                    tn = self._mut_rng.choice(sorted(self._tools.keys()))
                    new_type = self._mut_rng.choice(sorted(TOOL_DAG_TYPE_POOL))
                    self._tools[tn]["output_type"] = new_type

    def _make_observation(self, last_action: str | None, valid: bool | None) -> Observation:
        obs_mode = self._stress.get("obs_noise", "clean")
        gold = self.get_gold_state()
        if obs_mode == "clean":
            partial = gold
        else:
            partial = self._build_partial_obs(gold, obs_mode)

        text_parts = [
            f"Active vars: {sorted(self._vars.keys())}.",
            f"Target var: {self._target_var} (type {self._target_type}).",
            f"Open subgoals: {sorted(self._open_subgoals)}.",
        ]
        if last_action is not None:
            text_parts.append(f"Last action: {last_action} (valid={valid}).")
        text = " ".join(text_parts)
        return Observation(text=text, partial_state=partial, done=False, info={"step": self._t})

    def _build_partial_obs(self, gold: dict, mode: str) -> dict:
        partial = empty_world_state()

        for v in sorted(self._vars.keys()):
            partial["objects"][v] = gold["objects"].get(v, {"type": "variable", "props": {}})
        partial["inventory"] = sorted(self._vars.keys())
        partial["open_subgoals"] = list(self._open_subgoals)
        partial["completed_subgoals"] = list(self._completed_subgoals)

        if mode == "partial":
            return canonicalize_world_state(partial)

        if mode in ("distractor", "conflict") and self._tools and self._obs_rng.random() < 0.30:
            tn = self._obs_rng.choice(sorted(self._tools.keys()))
            partial["objects"][tn] = gold["objects"].get(tn, {"type": "tool", "props": {}})

        if mode == "conflict" and self._obs_rng.random() < 0.10 and self._vars:
            fake_v = sorted(self._vars.keys())[0]

            partial["objects"][fake_v] = {
                "type": "variable",
                "props": {"data_type": "TypeFAKE"},
            }

        return canonicalize_world_state(partial)
