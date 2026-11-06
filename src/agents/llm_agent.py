from __future__ import annotations
import re
from .base import BaseAgent, CallOutcome, PlannerCall, SelfDiagCall, UpdaterCall
from .llm_client import LLMClient
from .prompts import planner_user, self_diag_user, updater_user
from .prompts import PLANNER_SYS, SELF_DIAG_SYS, UPDATER_SYS
from .prompts import MODE_A_PLANNER_SYS, MODE_A_SELF_DIAG_SYS, MODE_A_UPDATER_SYS, planner_user_mode_a, self_diag_user_mode_a, updater_user_mode_a
MODE_A_MAX_TOKENS = {'planner': 400, 'updater': 500, 'self_diag': 250}
NL_MEMORY_KEY = '_natural_understanding'

class LLMPlanner(PlannerCall):

    def __init__(self, client: LLMClient, model: str, action_templates: list[str]):
        self.client = client
        self.model = model
        self.action_templates = action_templates

    def __call__(self, observation_text, agent_world_state, history, decoding_seed):
        usr = planner_user(observation_text, agent_world_state, self.action_templates, history)
        return self.client.call_typed(model=self.model, call_type='planner', system_prompt=PLANNER_SYS, user_prompt=usr, seed=decoding_seed)

class LLMUpdater(UpdaterCall):

    def __init__(self, client: LLMClient, model: str):
        self.client = client
        self.model = model

    def __call__(self, observation_text, observation_partial_state, prev_agent_world_state, last_action, last_action_outcome, decoding_seed):
        usr = updater_user(observation_text=observation_text, observation_partial_state=observation_partial_state, prev_agent_world_state=prev_agent_world_state, last_action=last_action, last_action_outcome=last_action_outcome)
        return self.client.call_typed(model=self.model, call_type='updater', system_prompt=UPDATER_SYS, user_prompt=usr, seed=decoding_seed)

class LLMSelfDiag(SelfDiagCall):

    def __init__(self, client: LLMClient, model: str):
        self.client = client
        self.model = model

    def __call__(self, agent_world_state, proposed_action, required_preconditions, decoding_seed):
        usr = self_diag_user(agent_world_state=agent_world_state, proposed_action=proposed_action, required_preconditions=required_preconditions)
        return self.client.call_typed(model=self.model, call_type='self_diag', system_prompt=SELF_DIAG_SYS, user_prompt=usr, seed=decoding_seed)
_EMPTY_WS_STUB = {'objects': {}, 'locations': {}, 'relations': [], 'inventory': [], 'open_subgoals': [], 'completed_subgoals': [], 'blocked_dependencies': [], 'beliefs': []}

def _extract_nl_memory(agent_world_state: dict) -> str:
    if not isinstance(agent_world_state, dict):
        return ''
    return str(agent_world_state.get(NL_MEMORY_KEY) or '')

def _wrap_nl_memory(text: str) -> dict:
    out = dict(_EMPTY_WS_STUB)
    out['objects'] = {}
    out['locations'] = {}
    out['relations'] = []
    out['inventory'] = []
    out['open_subgoals'] = []
    out['completed_subgoals'] = []
    out['blocked_dependencies'] = []
    out['beliefs'] = []
    out['beliefs'] = [{'prop': f'NL::{text}', 'confidence': 1.0}]
    return out

def _extract_nl_from_canonical_ws(ws: dict) -> str:
    if not isinstance(ws, dict):
        return ''
    for b in ws.get('beliefs') or []:
        if not isinstance(b, dict):
            continue
        prop = str(b.get('prop', ''))
        if prop.startswith('NL::'):
            return prop[len('NL::'):]
    return ''
_ACTION_LINE_RE = re.compile('(?im)^\\s*ACTION\\s*[:=]\\s*(.+?)\\s*$')
_VERDICT_LINE_RE = re.compile('(?im)^\\s*VERDICT\\s*[:=]\\s*([A-Za-z_]+)\\s*$')

def _parse_planner_mode_a(text: str, action_templates: list[str]) -> tuple[str, str]:
    reasoning = ''
    m_reason = re.search('(?im)^\\s*REASONING\\s*[:=]\\s*(.+?)$', text)
    if m_reason:
        reasoning = m_reason.group(1).strip()
    m = _ACTION_LINE_RE.search(text or '')
    candidate = m.group(1).strip().strip('\'"`') if m else ''
    while candidate and candidate[-1] in '.,;':
        candidate = candidate[:-1]
    if candidate.endswith(')') and candidate.count('(') < candidate.count(')'):
        candidate = candidate[:-1]
    if candidate.startswith('`') and candidate.endswith('`'):
        candidate = candidate.strip('`')
    if candidate:
        return (candidate, reasoning)
    if action_templates:
        for tmpl in action_templates:
            if tmpl and tmpl in (text or ''):
                return (tmpl, reasoning)
    return ('noop', reasoning)

def _parse_self_diag_mode_a(text: str) -> tuple[bool, str]:
    assessment = ''
    m_a = re.search('(?im)^\\s*ASSESSMENT\\s*[:=]\\s*(.+?)$', text or '')
    if m_a:
        assessment = m_a.group(1).strip()
    m = _VERDICT_LINE_RE.search(text or '')
    if m:
        v = m.group(1).strip().upper()
        if v in ('REPLAN', 'ABORT', 'NO', 'FALSE', 'INVALID'):
            return (False, assessment)
        if v in ('EXECUTE', 'RUN', 'GO', 'YES', 'TRUE', 'VALID'):
            return (True, assessment)
    return (True, assessment)

class ModeAUpdater(UpdaterCall):

    def __init__(self, client: LLMClient, model: str):
        self.client = client
        self.model = model

    def __call__(self, observation_text, observation_partial_state, prev_agent_world_state, last_action, last_action_outcome, decoding_seed):
        prev_nl = _extract_nl_from_canonical_ws(prev_agent_world_state)
        usr = updater_user_mode_a(observation_text=observation_text, prev_current_understanding=prev_nl, last_action=last_action, last_action_outcome=last_action_outcome)
        raw = self.client.call_raw(model=self.model, system_prompt=MODE_A_UPDATER_SYS, user_prompt=usr, seed=decoding_seed if self.model.startswith('gpt-') or self.model.startswith('azure:') else None, temperature=0.0, max_tokens=MODE_A_MAX_TOKENS['updater'])
        new_understanding = (raw.text or '').strip()
        if not new_understanding and prev_nl:
            new_understanding = prev_nl
        parsed = {'changed_facts': [], 'removed_facts': [], 'full_world_state': _wrap_nl_memory(new_understanding)}
        return CallOutcome(parsed=parsed, raw_text=raw.text, valid_json=True if raw.text else False, retries=0, input_tokens=raw.input_tokens, output_tokens=raw.output_tokens, wallclock_ms=raw.wallclock_ms, fallback_used=not bool(raw.text), extra={'system_fingerprint': raw.system_fingerprint, 'mode': 'A_free'})

class ModeAPlanner(PlannerCall):

    def __init__(self, client: LLMClient, model: str, action_templates: list[str]):
        self.client = client
        self.model = model
        self.action_templates = action_templates

    def __call__(self, observation_text, agent_world_state, history, decoding_seed):
        nl_mem = _extract_nl_from_canonical_ws(agent_world_state)
        usr = planner_user_mode_a(observation_text=observation_text, current_understanding=nl_mem, action_templates=self.action_templates, history=history)
        raw = self.client.call_raw(model=self.model, system_prompt=MODE_A_PLANNER_SYS, user_prompt=usr, seed=decoding_seed if self.model.startswith('gpt-') or self.model.startswith('azure:') else None, temperature=0.0, max_tokens=MODE_A_MAX_TOKENS['planner'])
        action, reasoning = _parse_planner_mode_a(raw.text or '', self.action_templates)
        parsed = {'next_action': action, 'required_preconditions': [], 'expected_effects': [], 'confidence': 0.5}
        return CallOutcome(parsed=parsed, raw_text=raw.text, valid_json=True if raw.text else False, retries=0, input_tokens=raw.input_tokens, output_tokens=raw.output_tokens, wallclock_ms=raw.wallclock_ms, fallback_used=not bool(raw.text), extra={'system_fingerprint': raw.system_fingerprint, 'mode': 'A_free', 'reasoning': reasoning})

class ModeASelfDiag(SelfDiagCall):

    def __init__(self, client: LLMClient, model: str):
        self.client = client
        self.model = model

    def __call__(self, agent_world_state, proposed_action, required_preconditions, decoding_seed):
        nl_mem = _extract_nl_from_canonical_ws(agent_world_state)
        usr = self_diag_user_mode_a(current_understanding=nl_mem, proposed_action=proposed_action)
        raw = self.client.call_raw(model=self.model, system_prompt=MODE_A_SELF_DIAG_SYS, user_prompt=usr, seed=decoding_seed if self.model.startswith('gpt-') or self.model.startswith('azure:') else None, temperature=0.0, max_tokens=MODE_A_MAX_TOKENS['self_diag'])
        valid, assessment = _parse_self_diag_mode_a(raw.text or '')
        parsed = {'self_check_valid': valid, 'missing_preconditions': [], 'should_replan': not valid}
        return CallOutcome(parsed=parsed, raw_text=raw.text, valid_json=True if raw.text else False, retries=0, input_tokens=raw.input_tokens, output_tokens=raw.output_tokens, wallclock_ms=raw.wallclock_ms, fallback_used=not bool(raw.text), extra={'system_fingerprint': raw.system_fingerprint, 'mode': 'A_free', 'assessment': assessment})

def build_llm_agent(client: LLMClient, model: str, action_templates: list[str], memory_mode: str='C_struct') -> BaseAgent:
    if memory_mode == 'A_free':
        return BaseAgent(name=f'llm:{model}:A_free', model_id=model, memory_mode=memory_mode, planner=ModeAPlanner(client, model, action_templates), updater=ModeAUpdater(client, model), self_diag=ModeASelfDiag(client, model))
    return BaseAgent(name=f'llm:{model}:{memory_mode}', model_id=model, memory_mode=memory_mode, planner=LLMPlanner(client, model, action_templates), updater=LLMUpdater(client, model), self_diag=LLMSelfDiag(client, model))
