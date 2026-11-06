from __future__ import annotations
import os
import sys
import textworld.envs.pddl.textgen as _tg

def _patched_eval_derive(self, context=None):
    context = context or self.context
    ns = dict(context.get('variables', {}))
    try:
        value = eval(self.expression, {'__builtins__': __builtins__}, ns)
    except NameError:
        value = ''
    except Exception:
        value = ''
    return [_tg.TerminalSymbol(value)]
_tg.EvalSymbol.derive = _patched_eval_derive
import textworld
import textworld.gym
import alfworld.agents.environment as _environment
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv, AlfredDemangler, AlfredInfos
DATA = os.environ['ALFWORLD_DATA']
_SP = os.path.dirname(_environment.__file__).rsplit('/agents', 1)[0]
SPLIT_TO_TRAINEVAL = {'valid_seen': 'eval_in_distribution', 'valid_unseen': 'eval_out_of_distribution'}

def _base_config():
    return {'dataset': {'data_path': f'{DATA}/json_2.1.1/train', 'eval_id_data_path': f'{DATA}/json_2.1.1/valid_seen', 'eval_ood_data_path': f'{DATA}/json_2.1.1/valid_unseen', 'num_train_games': -1, 'num_eval_games': -1}, 'logic': {'domain': f'{_SP}/data/alfred.pddl', 'grammar': f'{_SP}/data/alfred.twl2'}, 'env': {'domain_randomization': False, 'expert_type': 'handcoded', 'goal_desc_human_anns_prob': 0.0, 'task_types': [1, 2, 3, 4, 5, 6]}, 'general': {'training_method': 'dqn'}, 'rl': {'training': {'max_nb_steps_per_episode': 100}}}
_GAME_CACHE: dict[str, list[str]] = {}

def list_games(split: str) -> list[str]:
    if split in _GAME_CACHE:
        return _GAME_CACHE[split]
    train_eval = SPLIT_TO_TRAINEVAL[split]
    env_obj = AlfredTWEnv(_base_config(), train_eval=train_eval)
    games = sorted(env_obj.game_files)
    _GAME_CACHE[split] = games
    return games

def make_single_env(game_file: str, max_steps: int=100):
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True, extras=['gamefile'])
    wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
    env_id = textworld.gym.register_games([game_file], request_infos, batch_size=1, asynchronous=True, max_episode_steps=max_steps, wrappers=wrappers)
    return textworld.gym.make(env_id)
