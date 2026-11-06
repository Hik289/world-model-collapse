from __future__ import annotations
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import requests
sys.path.insert(0, '.')
import src.agents.prompts as prompts
_orig_json_short = prompts._json_short
prompts._json_short = lambda obj, limit=4000: _orig_json_short(obj, limit=10000000)
from src.agents.json_parser import parse_call_output
from src.agents.prompts import PLANNER_SYS, UPDATER_SYS, planner_user, updater_user
from src.environments.base import canonicalize_world_state, empty_world_state
from src.environments.stateful_puzzle import StatefulPuzzleEnv
from src.evaluation.runner import jaccard, world_state_facts

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write('\n')
    os.replace(tmp, path)

def jsonl_append(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + '\n')
        f.flush()

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def sha256_file(path: str) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except Exception:
        return None

def git_head() -> str | None:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None
_orig_build_partial_obs = StatefulPuzzleEnv._build_partial_obs

def _build_partial_obs_fixed(self, gold, mode):
    partial = _orig_build_partial_obs(self, gold, mode)
    current_room = self._current_room
    for neighbor in self._room_adj.get(current_room, []):
        partial['relations'].append({'subj': current_room, 'rel': 'adjacent', 'obj': neighbor})
    for container_id, container_meta in self._containers.items():
        if container_meta['location'] == current_room and container_meta['open']:
            partial['locations'][container_id] = gold['locations'].get(container_id, {'type': 'container', 'contents': []})
            for item_id in container_meta['contents']:
                partial['objects'][item_id] = gold['objects'].get(item_id, {'type': 'item', 'props': {}})
    partial['blocked_dependencies'] = []
    return canonicalize_world_state(partial)
StatefulPuzzleEnv._build_partial_obs = _build_partial_obs_fixed
MODEL_BY_KEY = {'gpt55': 'gpt-5.5', 'gpt54': 'gpt-5.4', 'mini': 'gpt-5.4-mini'}

class ModelTransportError(RuntimeError):
    pass

def call_model(model, system_prompt, user_prompt, label):
    from src.agents.llm_client import call_trace_model
    return call_trace_model(model, system_prompt, user_prompt, label)

def main() -> None:
    if len(sys.argv) not in (6, 7):
        raise SystemExit('usage: trace_v1_runner.py <model_key> <SC> <DD> <T> <run_dir> [task_index]')
    model_key = sys.argv[1]
    sc = int(sys.argv[2])
    dd = int(sys.argv[3])
    horizon = int(sys.argv[4])
    run_dir = Path(sys.argv[5]).resolve()
    task_index = int(sys.argv[6]) if len(sys.argv) == 7 else 0
    model = MODEL_BY_KEY[model_key]
    seed_suffix = '' if task_index == 0 else f'_s{task_index}'
    episode_id = f'{model_key}_sc{sc}_dd{dd}_T{horizon}{seed_suffix}'
    episode_dir = run_dir / 'raw' / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    steps_path = episode_dir / 'steps.jsonl'
    if steps_path.exists() and steps_path.stat().st_size > 0:
        raise RuntimeError(f'Refusing to overwrite non-empty raw trace: {steps_path}')
    with open('experiments/stage4_prep/stage4_task_seeds.json', encoding='utf-8') as f:
        grid = json.load(f)
    cell = [candidate for candidate in grid['cells'] if candidate['state_card'] == sc and candidate['dep_density'] == dd and (candidate['task_index'] == task_index)][0]
    cell = copy.deepcopy(cell)
    cell['stress_config']['obs_noise'] = 'partial'
    cell['task_config']['stress_config']['obs_noise'] = 'partial'
    cell['stress_config']['T'] = horizon
    cell['task_config']['stress_config']['T'] = horizon
    env = StatefulPuzzleEnv()
    observation = env.reset(cell['task_config'], seed=cell['task_seed'])
    if observation.partial_state.get('blocked_dependencies'):
        raise RuntimeError('Dynamic blocked_dependencies leaked into partial observation')
    task_spec = {subgoal_id: {'all_of': list(meta['preconds'])} for subgoal_id, meta in sorted(env._subgoals.items())}
    task_spec_text = json.dumps(task_spec, sort_keys=True, separators=(',', ':'))
    action_templates = list(env._action_templates)
    num_subgoals = len(env._open_subgoals)
    manifest = {'schema': 'worldmodelphase.trace_v1.episode_manifest', 'created_at': utc_now(), 'episode_id': episode_id, 'model_key': model_key, 'model': model, 'task_seed': cell['task_seed'], 'task_index': task_index, 'state_card': sc, 'dep_density': dd, 'horizon': horizon, 'observation_mode': 'partial', 'public_task_spec': task_spec, 'model_feedback_policy': {'dynamic_blocked_dependencies_visible': False, 'failure_reason_visible': False, 'missing_preconditions_visible': False, 'last_outcome_fields': ['valid']}, 'probe_fixes': ['disable_input_json_truncation', 'current_room_doors_visible', 'opened_local_container_contents_visible', 'public_static_subgoal_definitions', 'dynamic_gold_blocked_dependencies_hidden', 'transport_fail_closed'], 'task_config': cell['task_config'], 'stress_config': cell['stress_config'], 'action_templates': action_templates, 'prompt_hashes': {'updater_system_sha256': sha256_text(UPDATER_SYS), 'planner_system_sha256': sha256_text(PLANNER_SYS)}, 'source_hashes': {'runner_sha256': sha256_file(__file__), 'stateful_puzzle_sha256': sha256_file('src/environments/stateful_puzzle.py'), 'prompts_sha256': sha256_file('src/agents/prompts.py'), 'json_parser_sha256': sha256_file('src/agents/json_parser.py')}, 'git_head': git_head()}
    json_dump(episode_dir / 'manifest.json', manifest)
    agent_world_state = empty_world_state()
    history: list[dict] = []
    last_action = None
    last_outcome = None
    wsa_values: list[float] = []
    valid_actions = 0
    updater_parse_failures = 0
    planner_parse_failures = 0
    transport_retries = 0
    started_at = time.time()
    print(f"TRACE_V1_START episode={episode_id} model={model} seed={cell['task_seed']}", flush=True)
    try:
        for step in range(env._t_max):
            step_started = time.time()
            gold_before = copy.deepcopy(env.get_gold_state())
            obs_text = observation.text
            obs_partial = copy.deepcopy(observation.partial_state)
            if obs_partial.get('blocked_dependencies'):
                raise RuntimeError(f"blocked_dependencies leakage at step {step}: {obs_partial['blocked_dependencies']}")
            prev_state = copy.deepcopy(agent_world_state)
            history_visible = copy.deepcopy(history[-5:])
            updater_prompt = updater_user(observation_text=obs_text, observation_partial_state=obs_partial, prev_agent_world_state=prev_state, last_action=last_action, last_action_outcome=last_outcome)
            updater_raw, updater_transport = call_model(model, UPDATER_SYS, updater_prompt, f'{episode_id}:step={step}:updater')
            transport_retries += updater_transport['retry_count']
            updater_parsed, updater_valid, updater_error = parse_call_output(updater_raw, 'updater')
            if updater_valid and updater_parsed:
                agent_world_state = canonicalize_world_state(updater_parsed.get('full_world_state') or agent_world_state)
            else:
                updater_parse_failures += 1
            state_after_updater = copy.deepcopy(agent_world_state)
            planner_base_prompt = planner_user(obs_text, state_after_updater, action_templates, history_visible)
            planner_prompt = planner_base_prompt + '\n\nPUBLIC TASK SPECIFICATION ' + '(static; every all_of list is a conjunction):\n' + task_spec_text
            planner_raw, planner_transport = call_model(model, PLANNER_SYS, planner_prompt, f'{episode_id}:step={step}:planner')
            transport_retries += planner_transport['retry_count']
            planner_parsed, planner_valid, planner_error = parse_call_output(planner_raw, 'planner')
            if planner_valid and planner_parsed:
                action = planner_parsed.get('next_action', 'noop')
            else:
                planner_parse_failures += 1
                action = 'noop'
            agent_facts = world_state_facts(state_after_updater)
            gold_facts = world_state_facts(gold_before)
            global_wsa = jaccard(agent_facts, gold_facts)
            wsa_values.append(global_wsa)
            validity = env.check_action_validity(action)
            step_result = env.step(action)
            gold_after = copy.deepcopy(env.get_gold_state())
            is_valid = bool(validity.valid)
            valid_actions += int(is_valid)
            record = {'schema': 'worldmodelphase.trace_v1.step', 'episode_id': episode_id, 'step': step, 'timestamp': utc_now(), 'model_visible': {'observation_text': obs_text, 'observation_partial_state': obs_partial, 'public_task_spec': task_spec, 'prev_agent_world_state': prev_state, 'last_action': last_action, 'last_action_outcome': copy.deepcopy(last_outcome), 'recent_history': history_visible, 'action_templates': action_templates}, 'updater': {'system_prompt': UPDATER_SYS, 'user_prompt': updater_prompt, 'raw_output': updater_raw, 'parsed_output': updater_parsed, 'valid_json': updater_valid, 'parse_error': updater_error, 'fallback_used': not updater_valid, 'canonical_state': state_after_updater, 'transport': updater_transport}, 'planner': {'system_prompt': PLANNER_SYS, 'user_prompt': planner_prompt, 'raw_output': planner_raw, 'parsed_output': planner_parsed, 'valid_json': planner_valid, 'parse_error': planner_error, 'fallback_used': not planner_valid, 'proposed_action': action, 'transport': planner_transport}, 'evaluator_only': {'gold_before': gold_before, 'gold_after': gold_after, 'action_valid': is_valid, 'validity_reason': validity.reason, 'missing_preconditions': list(validity.missing), 'step_info': copy.deepcopy(step_result.info), 'global_wsa_legacy': global_wsa, 'agent_fact_count': len(agent_facts), 'gold_fact_count': len(gold_facts)}, 'transition': {'action': action, 'done': bool(step_result.done), 'reward': step_result.reward, 'next_observation_text': step_result.observation.text, 'next_observation_partial_state': copy.deepcopy(step_result.observation.partial_state)}, 'timing': {'step_elapsed_s': time.time() - step_started}}
            jsonl_append(steps_path, record)
            print(f't={step} action={action} valid={is_valid} legacy_wsa={global_wsa:.3f} completed={len(env._completed_subgoals)}/{num_subgoals}', flush=True)
            last_action = action
            last_outcome = {'valid': is_valid}
            history.append({'step': step, 'action': action, 'valid': is_valid})
            observation = step_result.observation
            if step_result.done:
                break
        success = len(env._open_subgoals) == 0
        summary = {'schema': 'worldmodelphase.trace_v1.episode_summary', 'episode_id': episode_id, 'status': 'completed', 'completed_at': utc_now(), 'success': success, 'steps': step + 1, 'mean_legacy_wsa': sum(wsa_values) / len(wsa_values), 'final_legacy_wsa': wsa_values[-1], 'valid_actions': valid_actions, 'action_validity_rate': valid_actions / (step + 1), 'completed_subgoals': len(env._completed_subgoals), 'total_subgoals': num_subgoals, 'updater_parse_failures': updater_parse_failures, 'planner_parse_failures': planner_parse_failures, 'transport_retries': transport_retries, 'elapsed_s': time.time() - started_at}
        json_dump(episode_dir / 'summary.json', summary)
        print(f'TRACE_V1_RESULT {json.dumps(summary, sort_keys=True)}', flush=True)
    except Exception as exc:
        failure = {'schema': 'worldmodelphase.trace_v1.failure', 'episode_id': episode_id, 'status': 'failed', 'failed_at': utc_now(), 'error_type': type(exc).__name__, 'error': repr(exc), 'elapsed_s': time.time() - started_at}
        json_dump(episode_dir / 'failure.json', failure)
        print(f'TRACE_V1_FAILURE {json.dumps(failure, sort_keys=True)}', flush=True)
        raise
if __name__ == '__main__':
    main()
