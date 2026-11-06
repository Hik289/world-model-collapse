from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import alfworld_common as AC
import alfworld_gold_state as G
import src.agents.prompts as prompts
_orig_json_short = prompts._json_short
prompts._json_short = lambda obj, limit=4000: _orig_json_short(obj, limit=10000000)
from src.agents.json_parser import parse_call_output
from src.agents.prompts import PLANNER_SYS, SELF_DIAG_SYS, UPDATER_SYS, planner_user, self_diag_user, updater_user
from src.environments.base import canonicalize_world_state, empty_world_state
from src.evaluation.runner import jaccard, world_state_facts
MODEL_BY_KEY = {'gpt55': 'gpt-5.5', 'gpt54': 'gpt-5.4', 'mini': 'gpt-5.4-mini', 'haiku': 'claude-haiku-4-5', 'sonnet': 'claude-sonnet-4-5', 'opus': 'claude-opus-4-8'}

class ModelTransportError(RuntimeError):
    pass

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

def call_model(model, system_prompt, user_prompt, label):
    from src.agents.llm_client import call_trace_model
    return call_trace_model(model, system_prompt, user_prompt, label)

def _facts_of(env_infos, idx=0):
    return env_infos['facts'][idx]

def _observed_from_gold(facts, observed: set) -> set:
    p = G.parse_facts(facts)
    observed |= set(p['checked'])
    if p['agent_loc'] is not None and p['agent_loc'] in p['receptacles']:
        observed.add(p['agent_loc'])
    return observed

def _survey_action(admissible: list[str], visited: set, opened: set) -> str | None:
    for c in admissible:
        if c.startswith('open '):
            r = c[5:].strip()
            if r not in opened:
                return c
    gos = sorted((c for c in admissible if c.startswith('go to ')))
    for c in gos:
        r = c[6:].strip()
        if r not in visited:
            return c
    return None

def _match_action(proposed: str, admissible: list[str]) -> tuple[str, bool]:
    if proposed in admissible:
        return (proposed, True)
    low = proposed.strip().lower()
    for c in admissible:
        if c.lower() == low:
            return (c, True)
    hits = [c for c in admissible if low and (low in c.lower() or c.lower() in low)]
    if len(hits) == 1:
        return (hits[0], False)
    if 'look' in admissible:
        return ('look', False)
    return (admissible[0], False)

def run_episode(game_file: str, run_dir: str | Path, model_key: str='mini', max_steps: int=40, episode_id: str | None=None, task_index: int=0, survey_steps: int=0) -> dict:
    model = MODEL_BY_KEY[model_key]
    run_dir = Path(run_dir).resolve()
    task_type = G.task_type_from_gamefile(game_file)
    env = AC.make_single_env(game_file, max_steps=max_steps + 5)
    obs_list, infos = env.reset()
    obs_text = obs_list[0]
    facts0 = _facts_of(infos)
    sc = G.compute_sc(facts0)
    state_card = sc['sc_facts']
    if episode_id is None:
        gid = hashlib.sha1(game_file.encode()).hexdigest()[:8]
        episode_id = f'{model_key}_{task_type}_sc{state_card}_{gid}'
    episode_dir = run_dir / 'raw' / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    steps_path = episode_dir / 'steps.jsonl'
    if steps_path.exists() and steps_path.stat().st_size > 0:
        summ = episode_dir / 'summary.json'
        if summ.exists():
            return json.loads(summ.read_text())
        steps_path.unlink()
    manifest = {'schema': 'worldmodelphase.trace_v1.episode_manifest', 'created_at': utc_now(), 'episode_id': episode_id, 'model_key': model_key, 'model': model, 'benchmark': 'alfworld', 'game_file': game_file, 'task_type': task_type, 'task_seed': task_index, 'task_index': task_index, 'state_card': state_card, 'dep_density': 0, 'horizon': max_steps, 'survey_steps': survey_steps, 'observation_mode': 'partial', 'sc_components': sc, 'public_task_spec': {}, 'prompt_hashes': {'updater_system_sha256': sha256_text(UPDATER_SYS), 'planner_system_sha256': sha256_text(PLANNER_SYS), 'self_diag_system_sha256': sha256_text(SELF_DIAG_SYS)}}
    json_dump(episode_dir / 'manifest.json', manifest)
    agent_world_state = empty_world_state()
    history: list[dict] = []
    last_action = None
    last_outcome = None
    survey_visited: set = set()
    survey_opened: set = set()
    observed_receptacles: set = set()
    object_loc_memory: dict = {}
    wsa_values: list[float] = []
    valid_actions = 0
    updater_parse_failures = 0
    planner_parse_failures = 0
    transport_retries = 0
    started_at = time.time()
    won = False
    step = -1
    print(f'ALFWORLD_START episode={episode_id} task={task_type} sc={state_card} model={model}', flush=True)
    try:
        for step in range(max_steps):
            step_started = time.time()
            facts = _facts_of(infos)
            gold_before = G.build_full_gold(facts)
            observed_receptacles = _observed_from_gold(facts, observed_receptacles)
            obs_partial = G.build_partial_obs(facts, observed_receptacles, object_loc_memory=object_loc_memory)
            admissible = list(infos['admissible_commands'][0])
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
            planner_prompt = planner_user(obs_text, state_after_updater, admissible, history_visible)
            planner_raw, planner_transport = call_model(model, PLANNER_SYS, planner_prompt, f'{episode_id}:step={step}:planner')
            transport_retries += planner_transport['retry_count']
            planner_parsed, planner_valid, planner_error = parse_call_output(planner_raw, 'planner')
            if planner_valid and planner_parsed:
                proposed_action = planner_parsed.get('next_action', 'look')
                required_preconditions = planner_parsed.get('required_preconditions', [])
            else:
                planner_parse_failures += 1
                proposed_action = 'look'
                required_preconditions = []
            self_diag_prompt = self_diag_user(state_after_updater, proposed_action, required_preconditions)
            self_diag_raw, self_diag_transport = call_model(model, SELF_DIAG_SYS, self_diag_prompt, f'{episode_id}:step={step}:self_diag')
            transport_retries += self_diag_transport['retry_count']
            self_diag_parsed, self_diag_valid, self_diag_error = parse_call_output(self_diag_raw, 'self_diag')
            action_cmd, was_admissible = _match_action(proposed_action, admissible)
            survey_override = None
            if step < survey_steps:
                survey_override = _survey_action(admissible, survey_visited, survey_opened)
                if survey_override is not None:
                    action_cmd = survey_override
                    was_admissible = True
                    if survey_override.startswith('open '):
                        survey_opened.add(survey_override[5:].strip())
                    elif survey_override.startswith('go to '):
                        survey_visited.add(survey_override[6:].strip())
            agent_facts = world_state_facts(state_after_updater)
            gold_facts = world_state_facts(gold_before)
            global_wsa = jaccard(agent_facts, gold_facts)
            wsa_values.append(global_wsa)
            valid_actions += int(was_admissible)
            obs_list, reward, done, infos = env.step([action_cmd])
            next_obs_text = obs_list[0]
            done_flag = done[0] if isinstance(done, (list, tuple)) else bool(done)
            won = bool(infos['won'][0]) if 'won' in infos else won
            gold_after = G.build_full_gold(_facts_of(infos))
            record = {'schema': 'worldmodelphase.trace_v1.step', 'episode_id': episode_id, 'step': step, 'timestamp': utc_now(), 'model_visible': {'observation_text': obs_text, 'observation_partial_state': obs_partial, 'public_task_spec': {}, 'prev_agent_world_state': prev_state, 'last_action': last_action, 'last_action_outcome': copy.deepcopy(last_outcome), 'recent_history': history_visible, 'action_templates': admissible}, 'updater': {'raw_output': updater_raw, 'parsed_output': updater_parsed, 'valid_json': updater_valid, 'parse_error': updater_error, 'fallback_used': not updater_valid, 'canonical_state': state_after_updater, 'transport': updater_transport}, 'planner': {'raw_output': planner_raw, 'parsed_output': planner_parsed, 'valid_json': planner_valid, 'parse_error': planner_error, 'fallback_used': not planner_valid, 'proposed_action': proposed_action, 'executed_action': action_cmd, 'was_admissible': was_admissible, 'survey_override': survey_override, 'transport': planner_transport}, 'self_diag': {'raw_output': self_diag_raw, 'parsed_output': self_diag_parsed, 'valid_json': self_diag_valid, 'parse_error': self_diag_error, 'transport': self_diag_transport}, 'evaluator_only': {'gold_before': gold_before, 'gold_after': gold_after, 'action_valid': was_admissible, 'global_wsa_legacy': global_wsa, 'agent_fact_count': len(agent_facts), 'gold_fact_count': len(gold_facts), 'reward': reward[0] if isinstance(reward, (list, tuple)) else reward, 'won': won}, 'transition': {'action': action_cmd, 'done': done_flag, 'reward': reward[0] if isinstance(reward, (list, tuple)) else reward, 'next_observation_text': next_obs_text}, 'timing': {'step_elapsed_s': time.time() - step_started}}
            jsonl_append(steps_path, record)
            print(f't={step} act={action_cmd!r} adm={was_admissible} legacy_wsa={global_wsa:.3f} won={won}', flush=True)
            last_action = action_cmd
            last_outcome = {'valid': was_admissible}
            history.append({'step': step, 'action': action_cmd, 'valid': was_admissible})
            obs_text = next_obs_text
            if done_flag or won:
                break
        steps_done = step + 1
        summary = {'schema': 'worldmodelphase.trace_v1.episode_summary', 'episode_id': episode_id, 'status': 'completed', 'completed_at': utc_now(), 'benchmark': 'alfworld', 'task_type': task_type, 'state_card': state_card, 'success': won, 'steps': steps_done, 'mean_legacy_wsa': sum(wsa_values) / len(wsa_values) if wsa_values else 0.0, 'final_legacy_wsa': wsa_values[-1] if wsa_values else 0.0, 'valid_actions': valid_actions, 'action_validity_rate': valid_actions / steps_done if steps_done else 0.0, 'completed_subgoals': int(won), 'total_subgoals': 1, 'updater_parse_failures': updater_parse_failures, 'planner_parse_failures': planner_parse_failures, 'transport_retries': transport_retries, 'elapsed_s': time.time() - started_at}
        json_dump(episode_dir / 'summary.json', summary)
        print(f'ALFWORLD_RESULT {json.dumps(summary, sort_keys=True)}', flush=True)
        return summary
    except Exception as exc:
        failure = {'schema': 'worldmodelphase.trace_v1.failure', 'episode_id': episode_id, 'status': 'failed', 'failed_at': utc_now(), 'error_type': type(exc).__name__, 'error': repr(exc), 'elapsed_s': time.time() - started_at}
        json_dump(episode_dir / 'failure.json', failure)
        print(f'ALFWORLD_FAILURE {json.dumps(failure, sort_keys=True)}', flush=True)
        raise
    finally:
        try:
            env.close()
        except Exception:
            pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('game_file')
    ap.add_argument('run_dir')
    ap.add_argument('--model-key', default='mini')
    ap.add_argument('--max-steps', type=int, default=40)
    ap.add_argument('--survey-steps', type=int, default=0)
    ap.add_argument('--episode-id', default=None)
    args = ap.parse_args()
    run_episode(args.game_file, args.run_dir, model_key=args.model_key, max_steps=args.max_steps, episode_id=args.episode_id, survey_steps=args.survey_steps)
if __name__ == '__main__':
    main()
