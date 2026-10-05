from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

from .protocol import atomic_json, build_plan, canonical, digest, load_config


def source_digest():
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root / "src").rglob("*.py")) + sorted((root / "experiments/paper").rglob("*.py"))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true", help="Write manifest only; no environments or models are run")
    parser.add_argument("--max-episodes", type=int, help="Maximum new episodes in this invocation")
    args = parser.parse_args()
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")
    plan = build_plan(load_config(args.config))
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


if __name__ == "__main__":
    main()
