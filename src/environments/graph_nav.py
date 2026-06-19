"""Graph Navigation environment (EXP_PLAN §2.4).

State-card → node count, decoy %, plus keys/switches/doors.
dep_density → number of preconditions per unlockable door.

This implementation prioritizes determinism + API compliance for Stage 1
anchor_1 verification. Semantic richness sufficient for collapse-onset
diagnostics in later stages.
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


# state_card → (nodes, keys, switches, doors, decoy_pct)
STATE_CARD_TABLE = {
    5: (5, 1, 1, 1, 0.0),
    10: (10, 2, 2, 2, 0.20),
    20: (20, 4, 4, 4, 0.35),
    40: (40, 8, 6, 6, 0.50),
}

OBS_NOISE_MODES = {"clean", "partial", "distractor", "conflict"}
MUT_RATE_MODES = {"static", "low", "medium", "high"}


class GraphNavEnv(Environment):
    name = "graph_nav"

    # -------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------
    def __init__(self) -> None:
        self._task_config: dict = {}
        self._stress: dict = {}
        self._rng: random.Random = random.Random(0)
        self._mut_rng: random.Random = random.Random(0)
        self._obs_rng: random.Random = random.Random(0)
        self._nodes: list[str] = []
        self._edges: dict[str, list[str]] = {}  # adjacency, sorted
        self._decoys: set[str] = set()
        self._keys: dict[str, str] = {}  # key_id -> initial location node
        self._switches: dict[str, str] = {}  # sw_id -> initial node
        self._switch_state: dict[str, bool] = {}
        self._doors: dict[str, dict] = {}  # door_id -> {nodes:(a,b), preconds:[...], unlocked:bool}
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

    # -------------------------------------------------------------------
    # Environment API
    # -------------------------------------------------------------------
    def reset(self, task_config: dict, seed: int) -> Observation:
        self._task_config = dict(task_config)
        self._stress = dict(task_config.get("stress_config") or {})
        # Three independent RNG streams seeded deterministically from (seed, role).
        self._rng = random.Random(f"{seed}:topology")
        self._mut_rng = random.Random(f"{seed}:mutation")
        self._obs_rng = random.Random(f"{seed}:observation")

        state_card = int(self._stress.get("state_card", 5))
        if state_card not in STATE_CARD_TABLE:
            raise ValueError(f"graph_nav: unknown state_card={state_card}")
        n_nodes, n_keys, n_switches, n_doors, decoy_pct = STATE_CARD_TABLE[state_card]

        dep_density = int(self._stress.get("dep_density", 1))
        self._t_max = int(self._stress.get("T", 40))
        self._t = 0
        self._inventory = []
        self._visited = set()
        self._completed_subgoals = []
        self._switch_state = {}

        # Build nodes
        self._nodes = [f"n{i}" for i in range(n_nodes)]

        # Build a connected base graph (random spanning tree) → adjacency
        self._edges = {n: [] for n in self._nodes}
        order = list(self._nodes)
        self._rng.shuffle(order)
        for i in range(1, len(order)):
            parent = order[self._rng.randrange(i)]
            self._edges[parent].append(order[i])
            self._edges[order[i]].append(parent)
        # Add a small number of additional edges (deterministic) for branching
        extra_edges = max(0, n_nodes // 4)
        for _ in range(extra_edges):
            a, b = self._rng.sample(self._nodes, 2)
            if b not in self._edges[a]:
                self._edges[a].append(b)
                self._edges[b].append(a)
        # Canonicalize adjacency (sorted)
        self._edges = {n: sorted(self._edges[n]) for n in self._nodes}

        # Decoys
        n_decoy = int(round(n_nodes * decoy_pct))
        decoy_nodes = self._rng.sample(self._nodes, n_decoy) if n_decoy else []
        self._decoys = set(decoy_nodes)

        # Keys / switches at deterministic node locations
        non_decoy = [n for n in self._nodes if n not in self._decoys] or list(self._nodes)
        self._keys = {f"key_{i}": non_decoy[self._rng.randrange(len(non_decoy))] for i in range(n_keys)}
        self._switches = {f"sw_{i}": non_decoy[self._rng.randrange(len(non_decoy))] for i in range(n_switches)}
        for sw in self._switches:
            self._switch_state[sw] = False

        # Doors (each connects a pair of nodes, has dep_density preconditions)
        self._doors = {}
        for i in range(n_doors):
            # pick two distinct nodes
            a, b = self._rng.sample(self._nodes, 2)
            preconds: list[str] = []
            pool: list[str] = []
            pool.extend([f"hold({k})" for k in self._keys])
            pool.extend([f"switch_on({s})" for s in self._switches])
            pool.extend([f"visited({n})" for n in self._nodes])
            # Sample without replacement up to dep_density; pad with hold(key) repeats avoided
            k = min(dep_density, len(pool))
            preconds = self._rng.sample(pool, k)
            self._doors[f"door_{i}"] = {
                "between": tuple(sorted([a, b])),
                "preconds": sorted(preconds),
                "unlocked": False,
            }

        # Starting node (deterministic): first non-decoy
        self._current_node = non_decoy[0]
        # Goal: last non-decoy
        self._goal_node = non_decoy[-1]
        self._open_subgoals = [f"reach({self._goal_node})"]

        # Object ids and action templates
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

        # Mark start visited
        self._visited.add(self._current_node)

        return self._make_observation(last_action=None, valid=None)

    # -------------------------------------------------------------------
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

        # Mutation step (deterministic, seeded by mut_rng)
        self._apply_mutation()

        # Subgoal check
        if self._current_node == self._goal_node and "reach(" + self._goal_node + ")" not in self._completed_subgoals:
            self._completed_subgoals.append(f"reach({self._goal_node})")
            self._open_subgoals = [g for g in self._open_subgoals if g != f"reach({self._goal_node})"]

        done = (not self._open_subgoals) or (self._t >= self._t_max)
        reward = 1.0 if (not self._open_subgoals) else 0.0
        obs = self._make_observation(last_action=action, valid=validity.valid)
        obs.done = done

        return StepResult(observation=obs, reward=reward, done=done, info=info)

    # -------------------------------------------------------------------
    def get_gold_state(self) -> dict:
        ws = empty_world_state()

        # Objects
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

        # Locations (= nodes, contents = keys/switches present)
        for n in self._nodes:
            contents: list[str] = []
            for k, loc in self._keys.items():
                if loc == n and k not in self._inventory:
                    contents.append(k)
            for s, loc in self._switches.items():
                if loc == n:
                    contents.append(s)
            ws["locations"][n] = {"type": "node", "contents": contents}

        # Relations: adjacency
        for n, nbrs in self._edges.items():
            for m in nbrs:
                if n < m:  # avoid double-listing
                    ws["relations"].append({"subj": n, "rel": "adjacent", "obj": m})
        ws["relations"].append({"subj": "agent", "rel": "at", "obj": self._current_node})

        # Inventory
        ws["inventory"] = list(self._inventory)

        # Subgoals
        ws["open_subgoals"] = list(self._open_subgoals)
        ws["completed_subgoals"] = list(self._completed_subgoals)

        # Blocked dependencies (per locked door)
        for d, meta in self._doors.items():
            if not meta["unlocked"]:
                missing = [p for p in meta["preconds"] if not self._precond_holds(p)]
                if missing:
                    ws["blocked_dependencies"].append({"action": f"unlock({d})", "missing": missing})

        # Beliefs (not used by graph_nav gold)
        return canonicalize_world_state(ws)

    # -------------------------------------------------------------------
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
                # Check if blocked by a door
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
            # Must be standing at one of the door's endpoints
            if self._current_node not in self._doors[d]["between"]:
                return ValidityCheck(valid=False, missing=[], reason="not_at_door")
            return ValidityCheck(valid=True)

        if verb == "inspect":
            if not args or args[0] not in self._object_ids:
                return ValidityCheck(valid=False, missing=[], reason="unknown_object")
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
            if validity.reason == "nonexistent_edge":
                labels.append("nonexistent_edge")
            if validity.reason in ("door_locked", "missing_precondition"):
                labels.append("missing_key")
            if validity.reason in ("key_not_here", "switch_not_here", "not_at_door"):
                labels.append("wrong_current_node")
            if validity.reason == "parse_error":
                labels.append("parse_error")

        # decoy_pursuit: agent moved into a decoy node
        if verb == "move" and args and args[0] in self._decoys:
            labels.append("decoy_pursuit")

        # revisited_loop: agent re-entered an already-visited node
        if verb == "move" and args and args[0] in self._visited:
            labels.append("revisited_loop")

        # stale_inventory: agent_state declares holding a key that env says not held
        try:
            ag_inv = sorted(agent_state.get("inventory", []) or [])
            if ag_inv != sorted(self._inventory):
                labels.append("stale_inventory")
        except (AttributeError, TypeError):
            pass

        # false_progress: agent claimed completed subgoal that isn't done
        try:
            ag_done = set(agent_state.get("completed_subgoals", []) or [])
            real_done = set(self._completed_subgoals)
            if ag_done - real_done:
                labels.append("false_progress")
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
                "nodes": list(self._nodes),
                "keys": sorted(self._keys.keys()),
                "switches": sorted(self._switches.keys()),
                "doors": sorted(self._doors.keys()),
                "goal_node": self._goal_node,
            },
        )

    # -------------------------------------------------------------------
    # Internal helpers
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
        """Mut rate-driven state perturbation (deterministic via _mut_rng)."""
        mut = self._stress.get("mut_rate", "static")
        if mut == "static":
            return
        # low: 0.2 prob of one fact flip; medium: 0.2 + 5% switch toggle; high: always 1 fact flip
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
            # Always change one fact
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
        """Render observation. obs_noise determines visibility."""
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
        # partial: only current-location objects + inventory + subgoals
        partial = empty_world_state()
        cur = self._current_node
        partial["objects"][cur] = gold["objects"].get(cur, {"type": "node", "props": {}})
        partial["locations"][cur] = gold["locations"].get(cur, {"type": "node", "contents": []})
        partial["inventory"] = list(gold["inventory"])
        partial["open_subgoals"] = list(gold["open_subgoals"])
        partial["completed_subgoals"] = list(gold["completed_subgoals"])
        # agent-at relation
        partial["relations"].append({"subj": "agent", "rel": "at", "obj": cur})

        if mode == "partial":
            return canonicalize_world_state(partial)

        # distractor: 30% chance inject snapshot of another location
        if self._obs_rng.random() < 0.30 and self._nodes:
            other = self._obs_rng.choice([n for n in self._nodes if n != cur] or [cur])
            partial["locations"][other] = gold["locations"].get(other, {"type": "node", "contents": []})
            partial["objects"][other] = gold["objects"].get(other, {"type": "node", "props": {}})

        if mode == "conflict" and self._obs_rng.random() < 0.10:
            # Inject a fact contradicting reality: claim a key is in current location when it isn't
            if self._keys:
                fake_key = self._obs_rng.choice(sorted(self._keys.keys()))
                if self._keys[fake_key] != cur and fake_key not in self._inventory:
                    contents = list(partial["locations"][cur]["contents"])
                    if fake_key not in contents:
                        contents.append(fake_key)
                        partial["locations"][cur] = {"type": "node", "contents": sorted(contents)}

        return canonicalize_world_state(partial)
