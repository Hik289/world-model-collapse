from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.paper.protocol import atomic_json, build_plan, load_config
from experiments.paper.analyze import failure_times, load_completed, summarize
from experiments.paper.statistics import adjust_pvalues, bootstrap_crossing, crossing, mn_score, wilson
from experiments.paper.run import BufferWriter, execute
from src.agents.llm_client import LLMClient, RawCallResult
from src.agents.base import BaseAgent, CallOutcome
from src.agents.prompts import planner_user
from src.environments import ENV_REGISTRY
from src.evaluation.runner import EpisodeContext, run_episode

ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "experiments/paper/configs/main_grid.json")

    def test_main_grid_count_and_paired_seeds(self):
        plan = build_plan(self.config)
        self.assertEqual(plan["episode_count"], 9600)
        self.assertEqual(plan["unique_tasks"], 4800)
        tasks = {}
        for job in plan["jobs"]:
            tasks.setdefault(job["task_id"], []).append(job)
        self.assertEqual(len({jobs[0]["task_seed"] for jobs in tasks.values()}), 4800)
        for jobs in tasks.values():
            self.assertEqual(len(jobs), 2)
            self.assertEqual(jobs[0]["task_seed"], jobs[1]["task_seed"])
            self.assertNotEqual(jobs[0]["job_id"], jobs[1]["job_id"])
        self.assertEqual(plan, build_plan(self.config))

    def test_configs_supported_and_ablations_deduplicated(self):
        for path in (ROOT / "experiments/paper/configs").glob("*.json"):
            config = load_config(path)
            plan = build_plan(config)
            self.assertEqual(len({j["job_id"] for j in plan["jobs"]}), plan["episode_count"])
        config = load_config(ROOT / "experiments/paper/configs/ablations.json")
        self.assertEqual(build_plan(config)["episode_count"], 1300)

    def test_invalid_config_rejected(self):
        for field, value in (("archetypes", 0), ("variants", True), ("environments", ["unknown"]), ("models", ["gpt-x", "gpt-x"])):
            config = copy.deepcopy(self.config)
            config[field] = value
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.json"
                atomic_json(path, config)
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_restored_environment_seed_determinism(self):
        task = {"stress_config": {"state_card": 5, "dep_density": 2, "T": 2,
                                  "branching": 4, "obs_noise": "clean", "mut_rate": "static"}}
        for name in ("graph_nav", "tool_dag"):
            a, b = ENV_REGISTRY[name](), ENV_REGISTRY[name]()
            self.assertEqual(a.reset(task, 12).to_dict(), b.reset(task, 12).to_dict())
            self.assertEqual(a.get_gold_state(), b.get_gold_state())

    def test_all_action_templates_reach_planner(self):
        actions = [f"inspect(v{i})" for i in range(100)] + ["move(goal)"]
        self.assertIn("move(goal)", planner_user("observation", {}, actions))


class StatisticsTests(unittest.TestCase):
    def test_window_and_action_use_same_clock(self):
        rows = [{"world_state_accuracy": v, "action_valid": i != 4}
                for i, v in enumerate([1, .2, .2, .2, .2])]
        self.assertEqual(failure_times(rows, 2), (3, 5))
        rows[1]["world_state_accuracy"] = .8
        self.assertEqual(failure_times(rows, 2), (4, 5))

    def test_strict_threshold_and_censoring(self):
        rows = [{"world_state_accuracy": .5, "action_valid": True}] * 3
        self.assertEqual(failure_times(rows, 2), (None, None))
        rows[0] = {"world_state_accuracy": .1, "action_valid": False}
        self.assertEqual(failure_times(rows, 4), (None, 1))

    def test_mn_against_independent_implementation(self):
        import math
        from scipy.optimize import brentq
        for counts in ((99, 100, 1, 100), (78, 100, 45, 100), (45, 51, 3, 31)):
            ours = mn_score(*counts)
            k1, n1, k2, n2 = counts
            score = lambda q: k1 / (q + .3) - (n1 - k1) / (.7 - q) + k2 / q - (n2 - k2) / (1 - q)
            q = brentq(score, 1e-12, .7 - 1e-12)
            variance = ((q + .3) * (.7 - q) / n1 + q * (1 - q) / n2) * (n1 + n2) / (n1 + n2 - 1)
            expected = .5 * math.erfc((k1 / n1 - k2 / n2 - .3) / math.sqrt(2 * variance))
            self.assertAlmostEqual(ours["p2_constrained"], q, places=6)
            self.assertAlmostEqual(ours["p_value"], expected, places=6)

    def test_mn_boundary_likelihoods(self):
        result = mn_score(100, 100, 0, 100)
        self.assertAlmostEqual(result["p1_constrained"], .65, places=6)
        self.assertAlmostEqual(result["p2_constrained"], .35, places=6)
        self.assertLess(result["p_value"], .01)
        zero = mn_score(0, 100, 0, 100)
        self.assertEqual(zero["p1_constrained"], .3)
        self.assertEqual(zero["p2_constrained"], 0)
        one = mn_score(100, 100, 100, 100)
        self.assertEqual(one["p1_constrained"], 1)
        self.assertAlmostEqual(one["p2_constrained"], .7)
        self.assertGreater(zero["p_value"], .99)
        self.assertGreater(one["p_value"], .99)

    def test_wilson_and_multiple_testing(self):
        lo, hi = wilson(0, 100)
        self.assertAlmostEqual(lo, 0)
        self.assertGreater(hi, 0)
        rows = [{"p_value": p} for p in [.001, .02, .9]]
        adjust_pvalues(rows)
        self.assertTrue(rows[0]["significant_bonferroni"])
        self.assertFalse(rows[1]["significant_bonferroni"])
        self.assertAlmostEqual(rows[1]["p_bh"], .03)

    def test_localization_does_not_extrapolate_or_use_upward_crossings(self):
        self.assertEqual(crossing([(10, .8), (20, .2)]), 15)
        self.assertIsNone(crossing([(10, .2), (20, .8)]))
        self.assertIsNone(crossing([(10, .8), (20, .6)]))
        self.assertIsNone(crossing([(10, .5)]))
        cells = [{"state_size": x, "n": 10, "success_rate": p} for x, p in [(10, 1), (20, 0)]]
        result = bootstrap_crossing(cells, repetitions=20)
        self.assertEqual(result["state_size_star"], 15)
        self.assertEqual(result["bracketed_fraction"], 1)

    def test_analysis_denominators_and_missing_offsets(self):
        config = load_config(ROOT / "experiments/paper/configs/main_grid.json")
        config.update(environments=["stateful_puzzle"], models=["gpt-4o-mini"], archetypes=1, variants=4)
        config["sweeps"][0].update(state_size=[5], state_dependency=[1])
        plan = build_plan(config)
        plan["source_digest"] = "fixture"
        traces = [([.1, .1, .1], [True, True, False], False),
                  ([1, 1, 1], [True, True, True], False),
                  ([.1, .1, .1], [False, True, True], True),
                  ([.8, .7, .6], [False, True, True], False)]
        completed = []
        for job, (values, valid, success) in zip(plan["jobs"], traces):
            completed.append({"job": job, "episode": {"final_success": success},
                              "steps": [{"world_state_accuracy": v, "action_valid": a} for v, a in zip(values, valid)]})
        report = summarize(plan, completed, 2, repetitions=20)
        row = report["failure_order"][0]
        self.assertEqual(row["collapsed_n"], 3)
        self.assertEqual(row["paired_n"], 1)
        self.assertEqual(row["world_first_pct"], 100)
        self.assertEqual(row["median_lead"], 1)
        fidelity = {r["relative_step"]: r for r in report["fidelity_before_action"]}
        self.assertIsNone(fidelity[-3]["mean_fidelity"])
        self.assertEqual(fidelity[-1]["n"], 1)
        self.assertEqual(fidelity[0]["n"], 2)
        self.assertAlmostEqual(fidelity[0]["mean_fidelity"], .45)

    def test_incomplete_grid_does_not_bridge_gaps(self):
        config = load_config(ROOT / "experiments/paper/configs/main_grid.json")
        config.update(environments=["stateful_puzzle"], models=["gpt-4o-mini"], archetypes=1, variants=1)
        config["sweeps"][0].update(state_size=[5, 10, 20], state_dependency=[1])
        plan = build_plan(config)
        plan["source_digest"] = "fixture"
        completed = [{"job": j, "episode": {"final_success": True},
                      "steps": [{"world_state_accuracy": 1, "action_valid": True}]}
                     for j in plan["jobs"] if j["stress"]["state_size"] != 10]
        report = summarize(plan, completed, 2, repetitions=20)
        self.assertEqual(report["incomplete_adjacent_pairs"], 2)
        self.assertEqual(report["cliff_tests"], [])
        self.assertEqual(report["critical_points"], [])


class RunnerTests(unittest.TestCase):
    def test_api_error_never_becomes_a_behavioral_failure(self):
        with patch.dict("os.environ", {}, clear=True):
            client = LLMClient(fixed_temperature=0.0, strict_api_errors=True)
        with patch.object(client, "call_raw", return_value=RawCallResult(text="", api_error="unavailable")) as call:
            with self.assertRaises(RuntimeError):
                client.call_typed("gpt-4o-mini", "planner", "system", "user", 42)
            self.assertEqual(call.call_count, 4)
            self.assertTrue(all(c.kwargs["temperature"] == 0 for c in call.call_args_list))

    def test_same_time_logging_and_nonblocking_self_diag(self):
        env = ENV_REGISTRY["stateful_puzzle"]()
        task = {"stress_config": {"state_card": 5, "dep_density": 1, "T": 1,
                                  "branching": 4, "obs_noise": "clean", "mut_rate": "static"}}
        agent = BaseAgent("fixture", "fixture", "C_struct",
                          planner=lambda **kw: CallOutcome({"next_action": "noop"}),
                          updater=lambda **kw: CallOutcome({"full_world_state": kw["observation_partial_state"]}),
                          self_diag=lambda **kw: CallOutcome({"self_check_valid": False, "should_replan": True}))
        ctx = EpisodeContext("r", "t", 42, 42, "fixture", task["stress_config"])
        sw, ew = BufferWriter(), BufferWriter()
        run_episode(env, agent, task, ctx, sw, ew)
        self.assertEqual(len(sw.rows), 1)
        row = sw.rows[0]
        self.assertEqual(row["env_name"], "stateful_puzzle")
        self.assertTrue(row["gold_world_state_before"])
        self.assertEqual(row["world_state_accuracy"], 1)
        self.assertTrue(row["action_valid"])
        self.assertFalse(row["self_check_valid"])

    def test_atomic_execution_resume_and_log_validation(self):
        config = load_config(ROOT / "experiments/paper/configs/main_grid.json")
        config.update(environments=["stateful_puzzle"], models=["gpt-4o-mini"], archetypes=1, variants=1)
        config["sweeps"][0].update(state_size=[5], state_dependency=[1], horizon=[1])
        plan = build_plan(config)
        agent = BaseAgent("fixture", "gpt-4o-mini", "C_struct",
                          planner=lambda **kw: CallOutcome({"next_action": "noop"}),
                          updater=lambda **kw: CallOutcome({"full_world_state": kw["observation_partial_state"]}),
                          self_diag=lambda **kw: CallOutcome({"self_check_valid": True}))
        with tempfile.TemporaryDirectory() as tmp, patch("src.agents.llm_client.LLMClient"), patch("src.agents.llm_agent.build_llm_agent", return_value=agent) as builder:
            execute(plan, Path(tmp))
            execute(plan, Path(tmp))
            self.assertEqual(builder.call_count, 1)
            self.assertEqual(len(load_completed(Path(tmp), plan)), 1)
            path = next((Path(tmp) / "episodes").glob("*.json"))
            record = json.loads(path.read_text())
            record["steps"][0]["world_state_accuracy"] = .123
            atomic_json(path, record)
            with self.assertRaises(ValueError):
                load_completed(Path(tmp), plan)

    def test_incomplete_episode_is_not_saved(self):
        config = load_config(ROOT / "experiments/paper/configs/main_grid.json")
        config.update(environments=["stateful_puzzle"], models=["gpt-4o-mini"], archetypes=1, variants=1)
        config["sweeps"][0].update(state_size=[5], state_dependency=[1], horizon=[1])
        plan = build_plan(config)
        with tempfile.TemporaryDirectory() as tmp, patch("src.agents.llm_client.LLMClient"), patch("src.agents.llm_agent.build_llm_agent", side_effect=RuntimeError("fixture")):
            with self.assertRaises(RuntimeError):
                execute(plan, Path(tmp))
            self.assertEqual(list((Path(tmp) / "episodes").glob("*.json")), [])
            self.assertEqual(len((Path(tmp) / "errors.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
