from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

from .protocol import AXES, atomic_json, digest
from .statistics import adjust_pvalues, bootstrap_crossing, mn_score, wilson

GROUP = ("env_name", "model", *AXES)


def group_record(job):
    return {"env_name": job["env_name"], "model": job["model"], **job["stress"]}


def key(record, fields=GROUP):
    return tuple(record[k] for k in fields)


def failure_times(steps, window, threshold=0.5):
    if type(window) is not int or window <= 0 or not 0 < threshold < 1:
        raise ValueError("Invalid onset window or threshold")
    accuracy = [row["world_state_accuracy"] for row in steps]
    tau_w = next((i + 1 for i in range(window - 1, len(steps))
                  if sum(accuracy[i - window + 1:i + 1]) / window < threshold), None)
    tau_a = next((i + 1 for i, row in enumerate(steps) if not row["action_valid"]), None)
    return tau_w, tau_a


def load_completed(directory, manifest):
    from src.evaluation.runner import jaccard, world_state_facts
    expected = {job["job_id"]: job for job in manifest["jobs"]}
    if len(expected) != len(manifest["jobs"]):
        raise ValueError("Duplicate job IDs in manifest")
    completed = []
    for path in sorted((directory / "episodes").glob("*.json")):
        record = json.loads(path.read_text())
        job = expected.get(path.stem)
        if job is None or record["job"] != job or record["protocol"] != manifest["protocol"]:
            raise ValueError(f"Manifest mismatch in {path.name}")
        episode, steps = record["episode"], record["steps"]
        if not steps or len(steps) != episode["steps_taken"] or len(steps) > job["stress"]["horizon"]:
            raise ValueError(f"Incomplete episode {path.name}")
        if type(episode["final_success"]) is not bool:
            raise ValueError(f"Non-boolean success in {path.name}")
        for row in [episode, *steps]:
            for field in ("task_id", "task_seed", "decoding_seed", "model", "env_name", "stress_config"):
                if row[field] != job[field]:
                    raise ValueError(f"Episode identity mismatch: {field} in {path.name}")
            if row["run_id"] != job["job_id"]:
                raise ValueError(f"Run identity mismatch in {path.name}")
        for i, row in enumerate(steps):
            if row["step"] != i or type(row["action_valid"]) is not bool:
                raise ValueError(f"Invalid or discontinuous steps in {path.name}")
            value = row["world_state_accuracy"]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid fidelity in {path.name}")
            gold = row.get("gold_world_state_before")
            if not isinstance(gold, dict) or not gold:
                raise ValueError(f"Missing pre-action gold state in {path.name}")
            recomputed = jaccard(world_state_facts(row["agent_world_state"]), world_state_facts(gold))
            if not math.isclose(value, recomputed, abs_tol=1e-10):
                raise ValueError(f"Fidelity does not match the pre-action state in {path.name}")
        completed.append(record)
    return completed


def load_boundary(path, manifest):
    if path is None:
        return None
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Boundary selection must be a nonempty JSON list")
    fields = ("env_name", "model", "state_size", "state_dependency")
    allowed = {key(group_record(job), fields) for job in manifest["jobs"]}
    selected = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(fields):
            raise ValueError(f"Boundary cells must have exactly {fields}")
        entry = key(row, fields)
        if entry not in allowed or entry in selected:
            raise ValueError(f"Unknown or duplicated boundary cell: {entry}")
        selected.add(entry)
    return selected


def summarize(manifest, completed, window, boundary=None, repetitions=1000):
    planned, observed = defaultdict(int), defaultdict(list)
    for job in manifest["jobs"]:
        planned[key(group_record(job))] += 1
    for record in completed:
        observed[key(group_record(record["job"]))].append(record)
    cells = []
    for cell_key, planned_n in sorted(planned.items()):
        records = observed[cell_key]
        n = len(records)
        successes = sum(r["episode"]["final_success"] for r in records)
        ci = wilson(successes, n) if n else (None, None)
        cells.append({**dict(zip(GROUP, cell_key)), "planned_n": planned_n, "n": n,
                      "successes": successes, "success_rate": successes / n if n else None,
                      "ci_low": ci[0], "ci_high": ci[1], "complete": n == planned_n})


    tests, incomplete_pairs = [], 0
    for axis in ("state_size", "state_dependency"):
        group_fields = tuple(f for f in GROUP if f != axis)
        lines = defaultdict(list)
        for cell in cells:
            lines[key(cell, group_fields)].append(cell)
        for line in lines.values():
            line.sort(key=lambda c: c[axis])
            for left, right in zip(line, line[1:]):
                if not left["complete"] or not right["complete"]:
                    incomplete_pairs += 1
                    continue
                tests.append({**{f: left[f] for f in group_fields}, "axis": axis,
                              "lower": left[axis], "upper": right[axis],
                              **mn_score(left["successes"], left["n"], right["successes"], right["n"])})

    if incomplete_pairs == 0:
        adjust_pvalues(tests)

    onsets, groups = [], defaultdict(list)
    boundary_fields = ("env_name", "model", "state_size", "state_dependency")
    summary_fields = ("env_name", "model", "horizon", "branching", "observation", "mutation")
    for record in completed:
        job, episode, steps = record["job"], record["episode"], record["steps"]
        info = group_record(job)
        tau_w, tau_a = failure_times(steps, window)
        selected = boundary is None or key(info, boundary_fields) in boundary
        collapsed = not episode["final_success"]
        paired = collapsed and tau_w is not None and tau_a is not None
        onset = {**info, "job_id": job["job_id"], "tau_w": tau_w, "tau_a": tau_a,
                 "collapsed": collapsed, "selected": selected, "paired": paired,
                 "lead": tau_a - tau_w if paired else None}
        onsets.append(onset)
        if selected:
            groups[key(info, summary_fields)].append((onset, steps))

    failure_order, fidelity = [], []
    for group_key, records in sorted(groups.items()):
        info = dict(zip(summary_fields, group_key))
        paired = [(o, steps) for o, steps in records if o["paired"]]
        n = len(paired)
        leads = [o["lead"] for o, _ in paired]
        failure_order.append({**info, "completed_n": len(records),
                              "collapsed_n": sum(o["collapsed"] for o, _ in records),
                              "paired_n": n,
                              "missing_world_onset_n": sum(o["collapsed"] and o["tau_w"] is None for o, _ in records),
                              "missing_action_onset_n": sum(o["collapsed"] and o["tau_a"] is None for o, _ in records),
                              "world_first_pct": 100 * sum(v > 0 for v in leads) / n if n else None,
                              "same_step_pct": 100 * sum(v == 0 for v in leads) / n if n else None,
                              "action_first_pct": 100 * sum(v < 0 for v in leads) / n if n else None,
                              "median_lead": statistics.median(leads) if n else None})


        action_observed = [(o, steps) for o, steps in records if o["collapsed"] and o["tau_a"] is not None]
        for offset in (-3, -2, -1, 0):
            values = [steps[o["tau_a"] - 1 + offset]["world_state_accuracy"]
                      for o, steps in action_observed if o["tau_a"] - 1 + offset >= 0]
            fidelity.append({**info, "relative_step": offset, "n": len(values),
                             "mean_fidelity": statistics.mean(values) if values else None})

    critical = []
    lines = defaultdict(list)
    fields = tuple(f for f in GROUP if f != "state_size")
    for cell in cells:
        if cell["state_dependency"] == 1:
            lines[key(cell, fields)].append(cell)
    for line_key, line in sorted(lines.items()):
        if all(c["complete"] for c in line) and len(line) >= 2:
            critical.append({**dict(zip(fields, line_key)), **bootstrap_crossing(line, repetitions=repetitions)})


    pilot_lines = defaultdict(list)
    pilot_fields = tuple(f for f in GROUP if f not in ("state_dependency", "model"))
    if manifest["config"]["name"] == "paper_pilot_sd":
        for cell in cells:
            pilot_lines[key(cell, pilot_fields)].append(cell)
    pilots = []
    for line_key, line in sorted(pilot_lines.items()):
        by_model = defaultdict(list)
        for cell in line:
            by_model[cell["model"]].append(cell)
        metrics = []
        for model, series in sorted(by_model.items()):
            series.sort(key=lambda c: c["state_dependency"])
            complete = all(c["complete"] for c in series) and len(series) >= 2
            drops = [a["success_rate"] - b["success_rate"] for a, b in zip(series, series[1:])] if complete else []
            metrics.append({"model": model, "complete": complete,
                            "nonincreasing": all(d >= -1e-12 for d in drops) if complete else None,
                            "max_drop": max(drops) if complete else None})
        passed = len(metrics) >= 2 and all(m["complete"] and m["nonincreasing"] and m["max_drop"] >= .20 for m in metrics)
        pilots.append({**dict(zip(pilot_fields, line_key)), "models": metrics, "passes_pilot_criteria": passed})
    return {"protocol": manifest["protocol"], "config_hash": manifest["config_hash"],
            "source_digest": manifest["source_digest"], "window": window, "fidelity_threshold": .5,
            "time_index": "one-based, pre-action state compared with validity of the same action",
            "collapse_definition": "final_success is false",
            "selection": "all tested cells" if boundary is None else sorted(boundary),
            "completed_episodes": len(completed), "planned_episodes": manifest["episode_count"],
            "incomplete_adjacent_pairs": incomplete_pairs,
            "multiple_testing_family": "all planned adjacent SS and SD pairs; corrections withheld until complete",
            "cells": cells, "cliff_tests": tests, "onsets": onsets, "failure_order": failure_order,
            "fidelity_before_action": fidelity, "critical_points": critical, "pilot_checks": pilots}


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        if rows:
            fields = list(dict.fromkeys(k for row in rows for k in row))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, required=True, help="Rolling window h from the analysis protocol; explicitly required")
    parser.add_argument("--boundary-cells", type=Path, help="Optional explicit near-boundary cell selection")
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()
    if args.window <= 0 or args.bootstrap_repetitions <= 0:
        parser.error("Window and bootstrap repetitions must be positive")
    manifest = json.loads((args.run / "manifest.json").read_text())
    if digest(manifest["config"]) != manifest["config_hash"]:
        raise ValueError("Manifest config hash mismatch")
    completed = load_completed(args.run, manifest)
    if not completed:
        parser.error("No completed episodes; a plan is not experimental data")
    boundary = load_boundary(args.boundary_cells, manifest)
    report = summarize(manifest, completed, args.window, boundary, args.bootstrap_repetitions)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "report.json", report)
    for name in ("cells", "cliff_tests", "onsets", "failure_order", "fidelity_before_action", "critical_points"):
        write_csv(args.output / f"{name}.csv", report[name])
    print(f"Analyzed {len(completed)}/{manifest['episode_count']} completed episodes; {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
