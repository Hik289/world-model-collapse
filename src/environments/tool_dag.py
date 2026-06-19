"""Tool-DAG environment (EXP_PLAN §2.5).

state_card → active variables / total tools / DAG depth.
dep_density → number of typed input args per tool (K).
branching → plausible next-tool count (used implicitly via tool count).
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


# state_card → (active_vars, total_tools, dag_depth)
STATE_CARD_TABLE = {
    5: (5, 5, 3),
    10: (10, 10, 4),
    20: (20, 20, 6),
    40: (40, 40, 8),
}

# Type pool — enough distinct types to satisfy "K typed inputs all distinct" for K up to 6.
TYPE_POOL = ["TypeA", "TypeB", "TypeC", "TypeD", "TypeE", "TypeF", "TypeG", "TypeH"]


class ToolDAGEnv(Environment):
    name = "tool_dag"

    def __init__(self) -> None:
        self._task_config: dict = {}
        self._stress: dict = {}
        self._rng: random.Random = random.Random(0)
        self._mut_rng: random.Random = random.Random(0)
        self._obs_rng: random.Random = random.Random(0)

        self._tools: dict[str, dict] = {}  # tool -> {input_types: [...], output_type: ..., output_var_prefix: ...}
        self._initial_vars: dict[str, str] = {}  # var_name -> type
        self._vars: dict[str, str] = {}  # current vars -> type
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

    # -------------------------------------------------------------------
    def reset(self, task_config: dict, seed: int) -> Observation:
        self._task_config = dict(task_config)
        self._stress = dict(task_config.get("stress_config") or {})
        self._rng = random.Random(f"{seed}:topology")
        self._mut_rng = random.Random(f"{seed}:mutation")
        self._obs_rng = random.Random(f"{seed}:observation")

        state_card = int(self._stress.get("state_card", 5))
        if state_card not in STATE_CARD_TABLE:
            raise ValueError(f"tool_dag: unknown state_card={state_card}")
        n_active, n_tools, dag_depth = STATE_CARD_TABLE[state_card]

        dep_density = int(self._stress.get("dep_density", 1))
        self._t_max = int(self._stress.get("T", 40))
        self._t = 0
        self._called_tools = []
        self._var_seq = 0
        self._completed_subgoals = []

        # Build typed initial variable pool. Types cycle deterministically.
        self._initial_vars = {}
        for i in range(n_active):
            t = TYPE_POOL[i % len(TYPE_POOL)]
            self._initial_vars[f"v{i}"] = t
        self._vars = dict(self._initial_vars)

        # Build tools. Each tool consumes K distinct-type args from the available
        # type pool and produces an output of some output_type. Layout in layers
        # of depth `dag_depth` so the graph terminates and has resolvable inputs.
        self._tools = {}
        # Ensure K distinct types: if dep_density > available types, raise. dep_density max 6 ≤ 8.
        k = min(dep_density, len(TYPE_POOL))
        # Distribute tools across layers
        n_layers = max(1, dag_depth)
        tools_per_layer = max(1, n_tools // n_layers)
        for tool_idx in range(n_tools):
            layer = min(tool_idx // tools_per_layer, n_layers - 1)
            # Choose k distinct types deterministically
            type_indices = list(range(len(TYPE_POOL)))
            self._rng.shuffle(type_indices)
            chosen_types = sorted({TYPE_POOL[type_indices[i]] for i in range(k)})
            # Output type cycles via deterministic choice
            out_type = TYPE_POOL[(layer + 1) % len(TYPE_POOL)]
            tool_name = f"tool_{tool_idx}"
            self._tools[tool_name] = {
                "input_types": list(chosen_types),
                "output_type": out_type,
                "layer": layer,
            }

        # Pick target type as final layer's output type
        last_layer = n_layers - 1
        last_layer_tools = [t for t, m in self._tools.items() if m["layer"] == last_layer]
        if last_layer_tools:
            self._target_type = self._tools[last_layer_tools[0]]["output_type"]
        else:
            self._target_type = TYPE_POOL[0]
        self._target_var = "TARGET"
        self._open_subgoals = [f"finish({self._target_var})"]

        # Build action templates + object ids
        self._object_ids = sorted(list(self._initial_vars.keys()) + list(self._tools.keys()) + [self._target_var])
        templates: list[str] = []
        for tn in sorted(self._tools.keys()):
            templates.append(f"call({tn})")  # short form, args resolved via available vars
        for v in sorted(self._initial_vars.keys()):
            templates.append(f"inspect_var({v})")
        templates.append(f"finish({self._target_var})")
        templates.append("declare(NEW_VAR,VAL)")
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

        # Blocked dependencies: each tool whose input types aren't all available
        avail_types = sorted(set(self._vars.values()))
        for tn, meta in self._tools.items():
            missing = [t for t in meta["input_types"] if t not in avail_types]
            if missing:
                ws["blocked_dependencies"].append({"action": f"call({tn})", "missing": [f"need_type({t})" for t in missing]})

        return canonicalize_world_state(ws)

    # -------------------------------------------------------------------
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

    # -------------------------------------------------------------------
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

        # repeated_useless_call: same tool called twice with same available types
        if verb == "call" and args and args[0] in self._called_tools:
            labels.append("repeated_useless_call")

        # skipped_dependency: agent declared/used a var without going through producer
        # (proxy: agent_state inventory contains a var not in env vars)
        try:
            ag_inv = set(agent_state.get("inventory", []) or [])
            real_inv = set(self._vars.keys())
            if ag_inv - real_inv:
                labels.append("fabricated_tool_result")
            if real_inv - ag_inv:
                labels.append("skipped_dependency")
        except (AttributeError, TypeError):
            pass

        # wrong_argument_source: declared subgoal completion without proper chain
        try:
            ag_done = set(agent_state.get("completed_subgoals", []) or [])
            real_done = set(self._completed_subgoals)
            if ag_done - real_done:
                labels.append("wrong_argument_source")
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
                "tools": sorted(self._tools.keys()),
                "initial_vars": sorted(self._initial_vars.keys()),
                "target_var": self._target_var,
                "target_type": self._target_type,
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
            # If output type matches target type and target not yet set → bind target
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
        # Drop a random non-initial var with some probability
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
                # Re-type a tool's output
                tn = self._mut_rng.choice(sorted(self._tools.keys()))
                new_type = self._mut_rng.choice(sorted(TYPE_POOL))
                self._tools[tn]["output_type"] = new_type
            return
        if mut == "high":
            # Always drop one var or retype one tool
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
                    new_type = self._mut_rng.choice(sorted(TYPE_POOL))
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
        # Only current variables and tools visible at base
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
            # Lie about its type
            partial["objects"][fake_v] = {
                "type": "variable",
                "props": {"data_type": "TypeFAKE"},
            }

        return canonicalize_world_state(partial)
