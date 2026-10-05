#!/usr/bin/env python3
"""Stage 4 G1-trigger dispatch — stateful_puzzle × 4 sc × 4 dd × 100 task.

Pre-registered in STAGE_4_PREREQUISITE_CHECKLIST.md Item #7.
Director sign-off: 2026-06-01 00:09 UTC.

Reads:
  - experiments/stage4_prep/stage4_task_seeds.json (1600 cells, sha256 seeds)

Writes:
  - data/raw_logs/stage4_step.jsonl
  - data/raw_logs/stage4_episode.jsonl
  - data/raw_logs/cost_tracker.jsonl (incremental)
  - experiments/stage4_g1_trigger/stage4_results.json (on completion)
  - experiments/stage4_g1_trigger/completed_cells.json (every 50 ep)

Resumability:
  - On startup: load completed_cells.json, skip those task_ids
  - Atomic write: write to .tmp + os.replace

Cost-tracker 0 verify:
  - After 100 episodes complete, sum slice cost in cost_tracker.jsonl
  - If > $0.001 → log [STAGE_4_ABORT_COST_NONZERO] and os._exit(13)
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.llm_client import LLMClient  # noqa: E402
from src.runner import (  # noqa: E402
    CellSpec, CostTracker, EpisodeOutcome, run_pilot_slice,
)


SLICE_NAME = "stage4_g1_trigger_sp_haiku"
SEEDS_PATH = ROOT / "experiments" / "stage4_prep" / "stage4_task_seeds.json"
OUT_DIR = ROOT / "experiments" / "stage4_g1_trigger"
LOG_DIR = ROOT / "data" / "raw_logs"
COMPLETED_PATH = OUT_DIR / "completed_cells.json"
RESULTS_PATH = OUT_DIR / "stage4_results.json"


def jst_now() -> str:
    return datetime.now(tz=timezone(timedelta(hours=9))).isoformat()


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def load_completed() -> set[str]:
    if not COMPLETED_PATH.exists():
        return set()
    try:
        data = json.loads(COMPLETED_PATH.read_text())
        return set(data.get("task_ids", []))
    except Exception:
        return set()


def save_completed(task_ids: set[str]) -> None:
    payload = {
        "schema_version": "stage4_completed_v1",
        "updated_jst": jst_now(),
        "n_completed": len(task_ids),
        "task_ids": sorted(task_ids),
    }
    atomic_write_json(COMPLETED_PATH, payload)


def cell_from_dict(d: dict) -> CellSpec:
    return CellSpec(
        env_name=d["env"],
        model=d["model"],
        stress_config=d["stress_config"],
        task_config=d["task_config"],
        task_seed=int(d["task_seed"]),
        decoding_seed=int(d["decoding_seed"]),
        world_regime=d["world_regime"],
        task_id=d["task_id"],
        memory_mode=d.get("memory_mode", "C_struct"),
    )


def verify_cost_zero_at_100ep(cost_tracker_path: Path, slice_name: str,
                              tolerance: float = 1e-3) -> tuple[bool, float, int]:
    """Sum episode.cost_usd for our slice across all rows; return (ok, total, n)."""
    total = 0.0
    n = 0
    if not cost_tracker_path.exists():
        return True, 0.0, 0  # no records yet → ok
    with open(cost_tracker_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("slice") != slice_name:
                continue
            ep_records = rec.get("episodes", [])
            if isinstance(ep_records, list):
                for er in ep_records:
                    if isinstance(er, dict):
                        c = float(er.get("cost_usd", 0.0))
                        total += c
                        n += 1
            # Also support legacy aggregate
            elif "total_cost_usd" in rec:
                total = max(total, float(rec.get("total_cost_usd", 0.0)))
                n = max(n, int(rec.get("n_episodes", 0)))
    return (total <= tolerance), total, n


def analyze_stage4(outcomes: list[EpisodeOutcome]) -> dict:
    """Per-cell aggregate over sc × dd × env (single env for Stage 4)."""
    from collections import defaultdict
    by_sc_dd = defaultdict(lambda: {"n": 0, "n_success": 0, "n_error": 0,
                                    "total_steps": 0, "total_in": 0, "total_out": 0,
                                    "total_cost": 0.0, "task_ids": []})
    for o in outcomes:
        sc = int(o.cell.stress_config["state_card"])
        dd = int(o.cell.stress_config["dep_density"])
        key = f"sc={sc},dd={dd}"
        d = by_sc_dd[key]
        d["n"] += 1
        d["task_ids"].append(o.cell.task_id)
        if o.success: d["n_success"] += 1
        if o.error: d["n_error"] += 1
        d["total_steps"] += o.steps
        d["total_in"] += o.input_tokens
        d["total_out"] += o.output_tokens
        d["total_cost"] += o.cost_usd

    out = {"per_cell": {}}
    for key, d in by_sc_dd.items():
        n = d["n"]
        out["per_cell"][key] = {
            "n": n,
            "n_success": d["n_success"],
            "success_rate": d["n_success"] / n if n else 0.0,
            "n_error": d["n_error"],
            "mean_steps": d["total_steps"] / n if n else 0.0,
            "total_input_tokens": d["total_in"],
            "total_output_tokens": d["total_out"],
            "total_cost_usd": round(d["total_cost"], 6),
        }
    return out


def legacy_main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--cost-verify-after", type=int, default=100)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[stage4] === Stage 4 G1-trigger dispatch ===")
    print(f"[stage4] start: {jst_now()}")

    # Load grid
    grid = json.loads(SEEDS_PATH.read_text())
    all_cells = [cell_from_dict(c) for c in grid["cells"]]

    # Resumability: skip completed
    completed = load_completed()
    if completed:
        cells = [c for c in all_cells if c.task_id not in completed]
        print(f"[stage4] resuming: {len(completed)} done previously, {len(cells)} remaining")
    else:
        cells = all_cells
        print(f"[stage4] fresh start: {len(cells)} cells")

    if not cells:
        print("[stage4] all cells already complete; computing final aggregate")
        return _finalize(all_cells, [])

    client = LLMClient()
    ct = CostTracker(
        out_path=LOG_DIR / "cost_tracker.jsonl",
        phase="stage4",
        slice_name=SLICE_NAME,
        emit_every=args.checkpoint_every,
    )

    completed_lock = threading.Lock()
    cost_verified = {"done": False, "abort": False}
    outcomes: list[EpisodeOutcome] = []
    t_start = time.perf_counter()

    def progress(i, n, o: EpisodeOutcome):
        success = "✓" if o.success else "✗"
        tag = "OK" if o.error is None else f"ERR({o.error[:60]})"
        elapsed = (time.perf_counter() - t_start) / 60.0
        rate = i / max(elapsed, 0.01)
        print(f"[stage4 {i}/{n}] {o.cell.task_id} {success}{tag} "
              f"steps={o.steps} | elapsed={elapsed:.1f}min rate={rate:.2f}ep/min")
        # Checkpoint
        with completed_lock:
            completed.add(o.cell.task_id)
            outcomes.append(o)
            if i % args.checkpoint_every == 0 or i == n:
                save_completed(completed)
                print(f"[stage4]   checkpoint: {len(completed)}/{len(all_cells)} cells done")
        # Cost verify at 100 ep
        if not cost_verified["done"] and i >= args.cost_verify_after:
            cost_verified["done"] = True
            ok, total, ncost = verify_cost_zero_at_100ep(
                LOG_DIR / "cost_tracker.jsonl", SLICE_NAME, tolerance=1e-3,
            )
            print(f"[stage4]   cost-verify @ {i} ep: total=${total:.6f} over {ncost} records, ok={ok}")
            if not ok:
                cost_verified["abort"] = True
                print(f"[STAGE_4_ABORT_COST_NONZERO] total=${total:.4f} > tolerance; aborting")
                save_completed(completed)
                os._exit(13)

    print(f"[stage4] launching run_pilot_slice (n_workers={args.n_workers}) ...")
    outs = run_pilot_slice(
        cells=cells,
        client=client,
        step_jsonl_path=LOG_DIR / "stage4_step.jsonl",
        episode_jsonl_path=LOG_DIR / "stage4_episode.jsonl",
        cost_tracker=ct,
        n_workers=args.n_workers,
        progress_fn=progress,
    )
    outcomes.extend(outs)  # progress() already appends; redundancy safe via dedup later
    return _finalize(all_cells, outcomes)


def _finalize(all_cells: list, outcomes: list) -> int:
    # Dedup outcomes by task_id (progress may have appended each twice)
    seen: dict[str, EpisodeOutcome] = {}
    for o in outcomes:
        seen[o.cell.task_id] = o
    uniq = list(seen.values())

    summary = analyze_stage4(uniq)
    summary["meta"] = {
        "timestamp_jst": jst_now(),
        "slice": SLICE_NAME,
        "model": "claude-haiku-4-5",
        "env": "stateful_puzzle",
        "grid_axes": {
            "state_cards": [5, 10, 20, 40],
            "dep_densities": [1, 2, 4, 6],
            "n_task_per_cell": 100,
            "total_cells_planned": len(all_cells),
        },
        "n_completed": len(uniq),
    }
    atomic_write_json(RESULTS_PATH, summary)
    save_completed({o.cell.task_id for o in uniq})

    print(f"\n[stage4] === DONE === wrote {RESULTS_PATH}")
    print(f"[stage4] completed {len(uniq)}/{len(all_cells)} cells")
    # Compact 4x4 grid display
    print("\n  success_rate grid (rows=state_card, cols=dep_density):")
    print("  {:>6} | {:>6} {:>6} {:>6} {:>6}".format("sc\\dd", 1, 2, 4, 6))
    for sc in [5, 10, 20, 40]:
        row = [f"{sc:>6}"]
        for dd in [1, 2, 4, 6]:
            key = f"sc={sc},dd={dd}"
            cell = summary["per_cell"].get(key, {})
            sr = cell.get("success_rate")
            row.append(f"{sr:>6.0%}" if sr is not None else "  N/A")
        print("  " + " ".join(row).replace(" | ", " | ", 1) + " |")
    return 0


PAPER_CONFIGS = {'ablations': {'name': 'paper_ablations',
               'seed_namespace': 'world-model-collapse-main42-v1-ablations',
               'environments': ['stateful_puzzle'],
               'models': ['claude-haiku-4-5'],
               'archetypes': 10,
               'variants': 10,
               'decoding_seed': 42,
               'sweeps': [{'state_size': [10],
                           'state_dependency': [6],
                           'horizon': [10, 20, 40, 80],
                           'branching': [4],
                           'observation': ['clean'],
                           'mutation': ['static']},
                          {'state_size': [10],
                           'state_dependency': [6],
                           'horizon': [40],
                           'branching': [2, 4, 8, 16],
                           'observation': ['clean'],
                           'mutation': ['static']},
                          {'state_size': [10],
                           'state_dependency': [6],
                           'horizon': [40],
                           'branching': [4],
                           'observation': ['clean', 'partial', 'distractor', 'conflict'],
                           'mutation': ['static']},
                          {'state_size': [10],
                           'state_dependency': [6],
                           'horizon': [40],
                           'branching': [4],
                           'observation': ['clean'],
                           'mutation': ['static', 'low', 'medium', 'high']}]},
 'main_grid': {'name': 'paper_main_grid',
               'seed_namespace': 'world-model-collapse-main42-v1-main_grid',
               'environments': ['stateful_puzzle', 'graph_nav', 'tool_dag'],
               'models': ['claude-haiku-4-5', 'gpt-4o-mini'],
               'archetypes': 10,
               'variants': 10,
               'decoding_seed': 42,
               'sweeps': [{'state_size': [5, 10, 20, 40],
                           'state_dependency': [1, 2, 4, 6],
                           'horizon': [40],
                           'branching': [4],
                           'observation': ['clean'],
                           'mutation': ['static']}]},
 'pilot_sd': {'name': 'paper_pilot_sd',
              'seed_namespace': 'world-model-collapse-main42-v1-pilot_sd',
              'environments': ['stateful_puzzle', 'graph_nav', 'tool_dag'],
              'models': ['claude-haiku-4-5', 'gpt-4o-mini'],
              'archetypes': 10,
              'variants': 1,
              'decoding_seed': 42,
              'sweeps': [{'state_size': [10],
                          'state_dependency': [1, 2, 4, 6],
                          'horizon': [40],
                          'branching': [4],
                          'observation': ['clean'],
                          'mutation': ['static']}]},
 'ss_fine': {'name': 'paper_ss_fine',
             'seed_namespace': 'world-model-collapse-main42-v1-ss_fine',
             'environments': ['stateful_puzzle'],
             'models': ['claude-haiku-4-5', 'gpt-4o-mini', 'gpt-4o', 'meta.llama3-70b-instruct-v1:0'],
             'archetypes': 10,
             'variants': 5,
             'decoding_seed': 42,
             'sweeps': [{'state_size': [5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 40],
                         'state_dependency': [1],
                         'horizon': [40],
                         'branching': [4],
                         'observation': ['clean'],
                         'mutation': ['static']}]},
 't_fine': {'name': 'paper_t_fine',
            'seed_namespace': 'world-model-collapse-main42-v1-t_fine',
            'environments': ['stateful_puzzle'],
            'models': ['claude-haiku-4-5'],
            'archetypes': 10,
            'variants': 5,
            'decoding_seed': 42,
            'sweeps': [{'state_size': [10],
                        'state_dependency': [6],
                        'horizon': [22, 25, 28, 30, 32, 35, 38, 42, 48, 55, 65],
                        'branching': [4],
                        'observation': ['clean'],
                        'mutation': ['static']}]}}


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
    config = json.loads(canonical(path)) if isinstance(path, dict) else json.loads(Path(path).read_text())
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


def source_digest():
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root / "src").rglob("*.py")) + [Path(__file__).resolve(), root / "analysis/stage4_g1_acceptance.py"]
    return digest({str(p.relative_to(root)): p.read_text() for p in sources})


@contextlib.contextmanager
def output_lock(directory):
    import fcntl
    path = directory / ".run.lock"
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another run is using {directory}") from exc
        yield


class BufferWriter:
    def __init__(self):
        self.rows = []

    def write_record(self, record):
        from dataclasses import asdict
        self.rows.append(asdict(record) if not isinstance(record, dict) else record)


def execute(plan, directory, max_episodes=None):

    from src.agents.llm_client import LLMClient
    from src.agents.llm_agent import build_llm_agent
    from src.environments import ENV_REGISTRY
    from src.evaluation.runner import EpisodeContext, run_episode

    client = LLMClient(fixed_temperature=0.0, strict_api_errors=True)
    completed_dir = directory / "episodes"
    completed_dir.mkdir(exist_ok=True)
    done = {p.stem for p in completed_dir.glob("*.json")}
    planned_ids = {job["job_id"] for job in plan["jobs"]}
    if done - planned_ids:
        raise ValueError("Output directory contains episodes outside this manifest")
    remaining = [job for job in plan["jobs"] if job["job_id"] not in done]
    selected = remaining if max_episodes is None else remaining[:max_episodes]
    for index, job in enumerate(selected, 1):
        try:
            env = ENV_REGISTRY[job["env_name"]]()
            env.reset(job["task_config"], job["task_seed"])
            agent = build_llm_agent(client, job["model"], env.get_meta().action_templates)
            ctx = EpisodeContext(run_id=job["job_id"], task_id=job["task_id"],
                                 task_seed=job["task_seed"], decoding_seed=job["decoding_seed"],
                                 world_regime=plan["config"]["name"], stress_config=job["stress_config"])
            steps, episodes = BufferWriter(), BufferWriter()
            run_episode(env, agent, job["task_config"], ctx, steps, episodes)
            atomic_json(completed_dir / f"{job['job_id']}.json",
                        {"protocol": plan["protocol"], "job": job,
                         "episode": episodes.rows[0], "steps": steps.rows})
        except Exception as exc:


            with (directory / "errors.jsonl").open("a") as handle:
                handle.write(canonical({"job_id": job["job_id"], "error_type": type(exc).__name__}) + "\n")
            raise RuntimeError(f"Episode {job['job_id']} incomplete ({type(exc).__name__}); resume after resolving the error") from None
        print(f"Completed {index}/{len(selected)}: {job['env_name']} {job['model']} "
              f"SS={job['stress']['state_size']} SD={job['stress']['state_dependency']}", flush=True)
    print(f"Stored {len(done) + len(selected)}/{plan['episode_count']} completed episodes")


def paper_main():
    parser = argparse.ArgumentParser(description="Three-environment paper experiment suites")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--suite", choices=sorted(PAPER_CONFIGS), default="main_grid")
    source.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true", help="Write manifest only; no environments or models are run")
    parser.add_argument("--max-episodes", type=int, help="Maximum new episodes in this invocation")
    args = parser.parse_args()
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")
    plan = build_plan(load_config(args.config if args.config else PAPER_CONFIGS[args.suite]))
    plan["source_digest"] = source_digest()
    args.output.mkdir(parents=True, exist_ok=True)
    with output_lock(args.output):
        manifest = args.output / "manifest.json"
        if manifest.exists():
            existing = json.loads(manifest.read_text())
            if existing != plan:
                raise ValueError("Configuration or implementation changed; use a new output directory")
        else:
            if any((args.output / "episodes").glob("*.json")):
                raise ValueError("Cannot adopt existing episodes without their original manifest")
            atomic_json(manifest, plan)
        print(f"{plan['episode_count']} episodes, {plan['unique_tasks']} unique tasks; manifest: {manifest}")
        if not args.plan_only:
            execute(plan, args.output, args.max_episodes)


def main():
    if "--legacy" in sys.argv:
        sys.argv.remove("--legacy")
        return legacy_main()
    if any(arg.split("=", 1)[0] in {"--n-workers", "--checkpoint-every", "--cost-verify-after"} for arg in sys.argv[1:]):
        return legacy_main()
    return paper_main()


if __name__ == "__main__":
    sys.exit(main())
