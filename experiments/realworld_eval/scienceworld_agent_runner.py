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
import scienceworld_gold_state as G
import src.agents.prompts as prompts
_orig_json_short = prompts._json_short
prompts._json_short = lambda obj, limit=4000: _orig_json_short(obj, limit=10000000)
from src.agents.json_parser import parse_call_output
from src.agents.prompts import PLANNER_SYS, SELF_DIAG_SYS, UPDATER_SYS, planner_user, self_diag_user, updater_user
from src.environments.base import canonicalize_world_state, empty_world_state
from src.evaluation.runner import jaccard, world_state_facts
from scienceworld import ScienceWorldEnv
MODEL_BY_KEY = {'haiku': 'claude-haiku-4-5', 'sonnet': 'claude-sonnet-4-5', 'opus': 'claude-opus-4-8', 'opus46': 'claude-opus-4-6', 'sonnet45': 'claude-sonnet-4-5-20250929', 'opus41': 'claude-opus-4-1-20250805', 'gpt55': 'gpt-5.5', 'gpt56luna': 'gpt-5.6-luna', 'gpt56sol': 'gpt-5.6-sol', 'gpt56terra': 'gpt-5.6-terra'}

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

def _filter_actions(valid: list[str], cap: int=120, block_focus: bool=False) -> list[str]:
    keep = []
    for a in valid:
        al = a.lower()
        if al.startswith(('connect ', 'disconnect ')):
            continue
        if block_focus and al.startswith('focus on '):
            continue
        keep.append(a)
    priority = ('open ', 'close ', 'go to ', 'teleport to ', 'pick up ', 'move ', 'focus on ', 'activate ', 'deactivate ', 'look', 'read ', 'wait')

    def rank(a):
        al = a.lower()
        for i, p in enumerate(priority):
            if al.startswith(p):
                return i
        return len(priority)
    keep.sort(key=rank)
    return keep[:cap]

def _match_action(proposed: str, valid: list[str]) -> tuple[str, bool]:
    if proposed in valid:
        return (proposed, True)
    low = proposed.strip().lower()
    for c in valid:
        if c.lower() == low:
            return (c, True)
    hits = [c for c in valid if low and (low in c.lower() or c.lower() in low)]
    if len(hits) == 1:
        return (hits[0], False)
    if 'look around' in valid:
        return ('look around', False)
    return (valid[0] if valid else 'look around', False)

def run_episode(task_name: str, run_dir: str | Path, variation: int=0, model_key: str='haiku', max_steps: int=35, episode_id: str | None=None, action_cap: int=120, block_focus: bool=False) -> dict:
    model = MODEL_BY_KEY[model_key]
    run_dir = Path(run_dir).resolve()
    env = ScienceWorldEnv('', envStepLimit=max_steps + 10)
    env.load(task_name, variation, '')
    obs_text, info = env.reset()
    task_desc = env.getTaskDescription()
    tree0 = env.getObjectTree()
    sc = G.compute_sc(tree0)
    state_card = sc['sc_facts']
    if episode_id is None:
        episode_id = f'{model_key}_{task_name}_v{variation}_sc{state_card}'
    episode_dir = run_dir / 'raw' / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    steps_path = episode_dir / 'steps.jsonl'
    if steps_path.exists() and steps_path.stat().st_size > 0:
        summ = episode_dir / 'summary.json'
        if summ.exists():
            env.close()
            return json.loads(summ.read_text())
        steps_path.unlink()
    manifest = {'schema': 'worldmodelphase.trace_v1.episode_manifest', 'created_at': utc_now(), 'episode_id': episode_id, 'model_key': model_key, 'model': model, 'benchmark': 'scienceworld', 'task_name': task_name, 'task_type': task_name, 'variation': variation, 'task_seed': variation, 'task_index': variation, 'state_card': state_card, 'dep_density': 0, 'horizon': max_steps, 'survey_steps': 0, 'observation_mode': 'partial', 'sc_components': sc, 'task_description': task_desc, 'public_task_spec': {}, 'prompt_hashes': {'updater_system_sha256': sha256_text(UPDATER_SYS), 'planner_system_sha256': sha256_text(PLANNER_SYS), 'self_diag_system_sha256': sha256_text(SELF_DIAG_SYS)}}
    json_dump(episode_dir / 'manifest.json', manifest)
    agent_world_state = empty_world_state()
    history: list[dict] = []
    last_action = None
    last_outcome = None
    object_loc_memory: dict = {}
    wsa_values: list[float] = []
    valid_action_count = 0
    updater_parse_failures = 0
    planner_parse_failures = 0
    transport_retries = 0
    started_at = time.time()
    won = False
    final_score = 0.0
    step = -1
    print(f"SCIENCEWORLD_START episode={episode_id} task={task_name} sc={state_card} rooms={sc['n_rooms']} model={model}", flush=True)
    obs_text = f'{task_desc}\n\n{obs_text}'
    stuck_in_loop = False
    try:
        for step in range(max_steps):
            step_started = time.time()
            tree = env.getObjectTree()
            gold_before = G.build_full_gold(tree)
            agent_room = G.agent_room_ref(tree)
            obs_partial = G.build_partial_obs(tree, agent_room=agent_room, object_loc_memory=object_loc_memory)
            valid_raw = env.getValidActionObjectCombinations()
            valid_actions = _filter_actions(valid_raw, cap=action_cap, block_focus=block_focus)
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
            planner_prompt = planner_user(obs_text, state_after_updater, valid_actions, history_visible)
            planner_raw, planner_transport = call_model(model, PLANNER_SYS, planner_prompt, f'{episode_id}:step={step}:planner')
            transport_retries += planner_transport['retry_count']
            planner_parsed, planner_valid, planner_error = parse_call_output(planner_raw, 'planner')
            if planner_valid and planner_parsed:
                proposed_action = planner_parsed.get('next_action', 'look around')
                required_preconditions = planner_parsed.get('required_preconditions', [])
            else:
                planner_parse_failures += 1
                proposed_action = 'look around'
                required_preconditions = []
            self_diag_prompt = self_diag_user(state_after_updater, proposed_action, required_preconditions)
            self_diag_raw, self_diag_transport = call_model(model, SELF_DIAG_SYS, self_diag_prompt, f'{episode_id}:step={step}:self_diag')
            transport_retries += self_diag_transport['retry_count']
            self_diag_parsed, self_diag_valid, self_diag_error = parse_call_output(self_diag_raw, 'self_diag')
            action_cmd, was_admissible = _match_action(proposed_action, valid_actions)
            agent_facts = world_state_facts(state_after_updater)
            gold_facts = world_state_facts(gold_before)
            global_wsa = jaccard(agent_facts, gold_facts)
            wsa_values.append(global_wsa)
            valid_action_count += int(was_admissible)
            next_obs_text, reward, done, info = env.step(action_cmd)
            score = info.get('score', 0) if isinstance(info, dict) else 0
            final_score = score
            done_flag = bool(done)
            won = bool(isinstance(score, (int, float)) and score >= 100)
            gold_after = G.build_full_gold(env.getObjectTree())
            record = {'schema': 'worldmodelphase.trace_v1.step', 'episode_id': episode_id, 'step': step, 'timestamp': utc_now(), 'model_visible': {'observation_text': obs_text, 'observation_partial_state': obs_partial, 'public_task_spec': {}, 'prev_agent_world_state': prev_state, 'last_action': last_action, 'last_action_outcome': copy.deepcopy(last_outcome), 'recent_history': history_visible, 'action_templates': valid_actions}, 'updater': {'raw_output': updater_raw, 'parsed_output': updater_parsed, 'valid_json': updater_valid, 'parse_error': updater_error, 'fallback_used': not updater_valid, 'canonical_state': state_after_updater, 'transport': updater_transport}, 'planner': {'raw_output': planner_raw, 'parsed_output': planner_parsed, 'valid_json': planner_valid, 'parse_error': planner_error, 'fallback_used': not planner_valid, 'proposed_action': proposed_action, 'executed_action': action_cmd, 'was_admissible': was_admissible, 'survey_override': None, 'transport': planner_transport}, 'self_diag': {'raw_output': self_diag_raw, 'parsed_output': self_diag_parsed, 'valid_json': self_diag_valid, 'parse_error': self_diag_error, 'transport': self_diag_transport}, 'evaluator_only': {'gold_before': gold_before, 'gold_after': gold_after, 'action_valid': was_admissible, 'global_wsa_legacy': global_wsa, 'agent_fact_count': len(agent_facts), 'gold_fact_count': len(gold_facts), 'reward': reward, 'score': score, 'won': won, 'agent_room': agent_room}, 'transition': {'action': action_cmd, 'done': done_flag, 'reward': reward, 'next_observation_text': next_obs_text}, 'timing': {'step_elapsed_s': time.time() - step_started}}
            jsonl_append(steps_path, record)
            print(f't={step} room={agent_room} act={action_cmd!r} adm={was_admissible} score={score} legacy_wsa={global_wsa:.3f} gold={len(gold_facts)} agent={len(agent_facts)}', flush=True)
            last_action = action_cmd
            last_outcome = {'valid': was_admissible, 'obs': next_obs_text[:200]}
            history.append({'step': step, 'action': action_cmd, 'valid': was_admissible})
            obs_text = next_obs_text
            if done_flag or won:
                break
            if len(history) >= 6:
                recent = [h['action'] for h in history[-6:]]
                if len(set(recent)) <= 2 and recent[-1] == recent[-3] == recent[-5]:
                    stuck_in_loop = True
                    break
        steps_done = step + 1
        summary = {'schema': 'worldmodelphase.trace_v1.episode_summary', 'episode_id': episode_id, 'status': 'completed', 'completed_at': utc_now(), 'benchmark': 'scienceworld', 'task_type': task_name, 'task_name': task_name, 'variation': variation, 'state_card': state_card, 'success': won, 'final_score': final_score, 'steps': steps_done, 'mean_legacy_wsa': sum(wsa_values) / len(wsa_values) if wsa_values else 0.0, 'final_legacy_wsa': wsa_values[-1] if wsa_values else 0.0, 'valid_actions': valid_action_count, 'action_validity_rate': valid_action_count / steps_done if steps_done else 0.0, 'completed_subgoals': int(won), 'total_subgoals': 1, 'updater_parse_failures': updater_parse_failures, 'planner_parse_failures': planner_parse_failures, 'transport_retries': transport_retries, 'elapsed_s': time.time() - started_at, 'stuck_in_loop': stuck_in_loop}
        json_dump(episode_dir / 'summary.json', summary)
        print(f'SCIENCEWORLD_RESULT {json.dumps(summary, sort_keys=True)}', flush=True)
        return summary
    except Exception as exc:
        failure = {'schema': 'worldmodelphase.trace_v1.failure', 'episode_id': episode_id, 'status': 'failed', 'failed_at': utc_now(), 'error_type': type(exc).__name__, 'error': repr(exc), 'elapsed_s': time.time() - started_at}
        json_dump(episode_dir / 'failure.json', failure)
        print(f'SCIENCEWORLD_FAILURE {json.dumps(failure, sort_keys=True)}', flush=True)
        raise
    finally:
        try:
            env.close()
        except Exception:
            pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task_name')
    ap.add_argument('run_dir')
    ap.add_argument('--variation', type=int, default=0)
    ap.add_argument('--model-key', default='haiku')
    ap.add_argument('--max-steps', type=int, default=35)
    ap.add_argument('--action-cap', type=int, default=120)
    ap.add_argument('--episode-id', default=None)
    args = ap.parse_args()
    run_episode(args.task_name, args.run_dir, variation=args.variation, model_key=args.model_key, max_steps=args.max_steps, episode_id=args.episode_id, action_cap=args.action_cap)
if __name__ == '__main__':
    main()
