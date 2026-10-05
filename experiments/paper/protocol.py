from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

PROTOCOL = "paper-main42-v1"
ENVIRONMENTS = {"stateful_puzzle", "graph_nav", "tool_dag"}
AXES = ("state_size", "state_dependency", "horizon", "branching", "observation", "mutation")
LEGACY_KEYS = dict(zip(AXES, ("state_card", "dep_density", "T", "branching", "obs_noise", "mut_rate")))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_config(path):
    config = json.loads(Path(path).read_text())
    required = {"name", "seed_namespace", "environments", "models", "archetypes", "variants", "decoding_seed", "sweeps"}
    if set(config) != required:
        raise ValueError(f"Config keys must be {sorted(required)}")
    for key in ("name", "seed_namespace"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    for key in ("environments", "models"):
        values = config[key]
        if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v for v in values):
            raise ValueError(f"Invalid {key}")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {key}")
    if not set(config["environments"]) <= ENVIRONMENTS:
        raise ValueError("Unknown environment")
    for model in config["models"]:
        if not model.startswith(("gpt-", "claude-", "azure:", "meta.", "us.meta.")):
            raise ValueError(f"Unsupported provider prefix: {model}")
    positive_int(config["archetypes"], "archetypes")
    positive_int(config["variants"], "variants")
    if type(config["decoding_seed"]) is not int or config["decoding_seed"] < 0:
        raise ValueError("decoding_seed must be a nonnegative integer")
    if not isinstance(config["sweeps"], list) or not config["sweeps"]:
        raise ValueError("sweeps must be a nonempty list")
    for sweep in config["sweeps"]:
        if set(sweep) != set(AXES):
            raise ValueError(f"Every sweep must specify {AXES}")
        for key in AXES:
            values = sweep[key]
            if not isinstance(values, list) or not values or any(isinstance(v, (list, dict)) for v in values):
                raise ValueError(f"Invalid sweep axis {key}")
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate levels in {key}")
        for key in AXES[:4]:
            for value in sweep[key]:
                positive_int(value, key)
        if not set(sweep["state_dependency"]) <= {1, 2, 4, 6}:
            raise ValueError("Supported SD levels: 1, 2, 4, 6")
        if not set(sweep["observation"]) <= {"clean", "partial", "distractor", "conflict"}:
            raise ValueError("Unknown observation mode")
        if not set(sweep["mutation"]) <= {"static", "low", "medium", "high"}:
            raise ValueError("Unknown mutation mode")
        levels = {5, 10, 20, 40}
        if config["environments"] == ["stateful_puzzle"]:
            levels.update(range(11, 20))
        if not set(sweep["state_size"]) <= levels:
            raise ValueError(f"State sizes supported by all selected environments: {sorted(levels)}")
    return config


def build_plan(config):
    tasks, seen_seeds = {}, {}
    for sweep in config["sweeps"]:
        for values in itertools.product(*(sweep[key] for key in AXES)):
            stress = dict(zip(AXES, values))
            for env, archetype, variant in itertools.product(
                config["environments"], range(config["archetypes"]), range(config["variants"])
            ):
                identity = {"namespace": config["seed_namespace"], "env_name": env,
                            "stress": stress, "archetype": archetype, "variant": variant}
                task_id = digest(identity)
                seed = int(task_id[:16], 16)
                if seed in seen_seeds and seen_seeds[seed] != task_id:
                    raise ValueError("SHA-256 seed prefix collision")
                seen_seeds[seed] = task_id
                legacy_stress = {LEGACY_KEYS[k]: v for k, v in stress.items()}
                tasks[task_id] = {"task_id": task_id, "task_seed": seed, "env_name": env,
                                  "stress": stress, "stress_config": legacy_stress,
                                  "task_config": {"archetype": f"archetype_{archetype:02d}",
                                                  "variant": variant, "stress_config": legacy_stress}}
    jobs = []
    for task in tasks.values():
        for model in config["models"]:
            job = {**task, "model": model, "decoding_seed": config["decoding_seed"]}
            job["job_id"] = digest({"task_id": task["task_id"], "model": model,
                                    "decoding_seed": config["decoding_seed"], "protocol": PROTOCOL})
            jobs.append(job)
    return {"protocol": PROTOCOL, "config": config, "config_hash": digest(config),
            "unique_tasks": len(tasks), "episode_count": len(jobs), "jobs": jobs}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{lineno}") from exc
    return rows
