from __future__ import annotations
import os
import time
import json
from dataclasses import dataclass, field
from typing import Any
import requests
from openai import OpenAI
from .base import CallOutcome
from .json_parser import parse_call_output
try:
    import boto3
    _BOTO3_AVAILABLE = True
except Exception:
    _BOTO3_AVAILABLE = False
ANTHROPIC_API_URL = os.environ.get('ANTHROPIC_API_URL', 'https://api.anthropic.com/v1/messages')
ANTHROPIC_VERSION = '2023-06-01'
DEFAULT_MAX_TOKENS = {'planner': 300, 'updater': 1200, 'self_diag': 200}

@dataclass
class RawCallResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    wallclock_ms: int = 0
    system_fingerprint: str = ''
    raw_meta: dict = field(default_factory=dict)
    api_error: str = ''

class LLMClient:

    def __init__(self, openai_api_key: str | None=None, anthropic_api_url: str=ANTHROPIC_API_URL, request_timeout: float=60.0, fixed_temperature: float | None=None, strict_api_errors: bool=False):
        api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        self.oai = OpenAI(api_key=api_key, timeout=request_timeout) if api_key else None
        self.anthropic_url = anthropic_api_url
        self.timeout = request_timeout
        self.fixed_temperature = fixed_temperature
        self.strict_api_errors = strict_api_errors
        self.model_ids = json.loads(os.environ.get('MODEL_ID_MAP', '{}'))
        self._bedrock = None
        self._azure: OpenAI | None = None
        azure_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
        azure_key = os.environ.get('AZURE_OPENAI_KEY')
        if azure_endpoint and azure_key:
            try:
                self._azure = OpenAI(api_key=azure_key, base_url=azure_endpoint, timeout=request_timeout, default_query={'api-version': 'preview'}, default_headers={'api-key': azure_key})
            except Exception:
                self._azure = None

    def call_raw(self, model: str, system_prompt: str, user_prompt: str, seed: int | None, temperature: float, max_tokens: int) -> RawCallResult:
        model = self.model_ids.get(model, model)
        if model.startswith('bedrock:'):
            model = model.split(':', 1)[1]
            if 'anthropic.' in model:
                return self._bedrock_anthropic_raw(model, system_prompt, user_prompt, temperature, max_tokens)
            return self._bedrock_raw(model, system_prompt, user_prompt, temperature, max_tokens)
        if model.startswith(('anthropic.', 'us.anthropic.', 'eu.anthropic.', 'global.anthropic.')):
            return self._bedrock_anthropic_raw(model, system_prompt, user_prompt, temperature, max_tokens)
        if model.startswith('azure:'):
            return self._azure_raw(model.split(':', 1)[1], system_prompt, user_prompt, seed, temperature, max_tokens)
        if model.startswith('gpt-'):
            return self._openai_raw(model, system_prompt, user_prompt, seed, temperature, max_tokens)
        if model.startswith('claude-'):
            return self._anthropic_raw(model, system_prompt, user_prompt, temperature, max_tokens)
        if model.startswith('meta.') or model.startswith('us.meta.'):
            return self._bedrock_raw(model, system_prompt, user_prompt, temperature, max_tokens)
        raise ValueError(f'unsupported model {model}')

    def _azure_raw(self, deployment: str, system_prompt: str, user_prompt: str, seed: int | None, temperature: float, max_tokens: int) -> RawCallResult:
        if self._azure is None:
            return RawCallResult(text='', api_error='azure_not_configured')
        t0 = time.perf_counter()
        try:
            kwargs: dict[str, Any] = {'model': deployment, 'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}], 'max_tokens': max_tokens, 'temperature': temperature}
            if seed is not None:
                kwargs['seed'] = seed
            resp = self._azure.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ''
            usage = resp.usage
            return RawCallResult(text=text, input_tokens=usage.prompt_tokens if usage else 0, output_tokens=usage.completion_tokens if usage else 0, wallclock_ms=int((time.perf_counter() - t0) * 1000), system_fingerprint=getattr(resp, 'system_fingerprint', '') or '', raw_meta={'id': getattr(resp, 'id', ''), 'model': getattr(resp, 'model', deployment)})
        except Exception as e:
            return RawCallResult(text='', wallclock_ms=int((time.perf_counter() - t0) * 1000), api_error=type(e).__name__)

    def _openai_raw(self, model: str, system_prompt: str, user_prompt: str, seed: int | None, temperature: float, max_tokens: int) -> RawCallResult:
        if self.oai is None:
            return RawCallResult(text='', api_error='OPENAI_API_KEY is not configured')
        t0 = time.perf_counter()
        try:
            kwargs: dict[str, Any] = {'model': model, 'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}], 'max_tokens': max_tokens, 'temperature': temperature}
            if model.startswith(('gpt-5', 'o1', 'o3', 'o4')):
                kwargs.pop('temperature', None)
                kwargs['max_completion_tokens'] = kwargs.pop('max_tokens')
            elif seed is not None:
                kwargs['seed'] = seed
            resp = self.oai.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ''
            usage = resp.usage
            return RawCallResult(text=text, input_tokens=usage.prompt_tokens if usage else 0, output_tokens=usage.completion_tokens if usage else 0, wallclock_ms=int((time.perf_counter() - t0) * 1000), system_fingerprint=getattr(resp, 'system_fingerprint', '') or '', raw_meta={'id': resp.id, 'model': resp.model})
        except Exception as e:
            return RawCallResult(text='', wallclock_ms=int((time.perf_counter() - t0) * 1000), api_error=type(e).__name__)

    def _anthropic_raw(self, model: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> RawCallResult:
        t0 = time.perf_counter()
        payload = {'model': model, 'max_tokens': max_tokens, 'temperature': temperature, 'system': system_prompt, 'messages': [{'role': 'user', 'content': user_prompt}]}
        headers = {'Content-Type': 'application/json', 'anthropic-version': ANTHROPIC_VERSION}
        if os.environ.get('ANTHROPIC_API_KEY'):
            headers['x-api-key'] = os.environ['ANTHROPIC_API_KEY']
        try:
            r = requests.post(self.anthropic_url, json=payload, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            text_chunks = [c.get('text', '') for c in data.get('content', []) if c.get('type') == 'text']
            text = ''.join(text_chunks)
            usage = data.get('usage', {})
            return RawCallResult(text=text, input_tokens=int(usage.get('input_tokens', 0)), output_tokens=int(usage.get('output_tokens', 0)), wallclock_ms=int((time.perf_counter() - t0) * 1000), system_fingerprint='', raw_meta={'id': data.get('id', ''), 'model': data.get('model', model)})
        except Exception as e:
            return RawCallResult(text='', wallclock_ms=int((time.perf_counter() - t0) * 1000), api_error=type(e).__name__)

    def _ensure_bedrock(self):
        if self._bedrock is not None:
            return self._bedrock
        if not _BOTO3_AVAILABLE:
            raise RuntimeError('boto3 not installed; cannot use Bedrock models')
        region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
        self._bedrock = boto3.client('bedrock-runtime', region_name=region)
        return self._bedrock

    def _bedrock_raw(self, model: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> RawCallResult:
        t0 = time.perf_counter()
        try:
            client = self._ensure_bedrock()
        except Exception as e:
            return RawCallResult(text='', api_error=type(e).__name__)
        if 'llama3' in model or 'llama-3' in model:
            prompt = f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            body = {'prompt': prompt, 'max_gen_len': max_tokens, 'temperature': max(temperature, 0.01), 'top_p': 0.9}
        else:
            return RawCallResult(text='', api_error=f'unsupported bedrock model format: {model}')
        try:
            resp = client.invoke_model(modelId=model, body=json.dumps(body), contentType='application/json', accept='application/json')
            payload = json.loads(resp['body'].read())
            text = payload.get('generation', '') or ''
            in_tok = int(payload.get('prompt_token_count', 0))
            out_tok = int(payload.get('generation_token_count', 0))
            return RawCallResult(text=text, input_tokens=in_tok, output_tokens=out_tok, wallclock_ms=int((time.perf_counter() - t0) * 1000), system_fingerprint='', raw_meta={'model': model, 'stop_reason': payload.get('stop_reason', '')})
        except Exception as e:
            return RawCallResult(text='', wallclock_ms=int((time.perf_counter() - t0) * 1000), api_error=type(e).__name__)

    def _bedrock_anthropic_raw(self, model, system_prompt, user_prompt, temperature, max_tokens):
        started = time.perf_counter()
        try:
            client = self._ensure_bedrock()
            body = {'anthropic_version': 'bedrock-2023-05-31', 'max_tokens': max_tokens, 'temperature': temperature, 'system': system_prompt, 'messages': [{'role': 'user', 'content': user_prompt}]}
            response = client.invoke_model(modelId=model, body=json.dumps(body), contentType='application/json', accept='application/json')
            payload = json.loads(response['body'].read())
            text = ''.join((part.get('text', '') for part in payload.get('content', []) if part.get('type') == 'text'))
            usage = payload.get('usage', {})
            return RawCallResult(text=text, input_tokens=usage.get('input_tokens', 0), output_tokens=usage.get('output_tokens', 0), wallclock_ms=int(1000 * (time.perf_counter() - started)), raw_meta={'model': model, 'stop_reason': payload.get('stop_reason', '')}, api_error='' if text.strip() else 'empty_response')
        except Exception as exc:
            return RawCallResult(text='', api_error=type(exc).__name__, wallclock_ms=int(1000 * (time.perf_counter() - started)))

    def call_typed(self, model: str, call_type: str, system_prompt: str, user_prompt: str, seed: int | None, max_retries: int=3, base_temperature: float=0.7, max_tokens: int | None=None) -> CallOutcome:
        if call_type not in ('planner', 'updater', 'self_diag'):
            raise ValueError(f'unknown call_type {call_type}')
        mt = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS[call_type]
        is_anthropic = model.startswith('claude-')
        is_bedrock = model.startswith('meta.') or model.startswith('us.meta.')
        no_seed = is_anthropic or is_bedrock
        attempt_temps = [0.0] * (max_retries + 1) if no_seed else [base_temperature, base_temperature + 0.1, base_temperature + 0.2, base_temperature + 0.3]
        attempt_seeds: list[int | None] = [None] * (max_retries + 1) if no_seed else [seed, (seed or 0) + 1, (seed or 0) + 2, (seed or 0) + 3]
        if self.fixed_temperature is not None:
            attempt_temps = [self.fixed_temperature] * (max_retries + 1)
        total_in = 0
        total_out = 0
        total_ms = 0
        last_text = ''
        last_err = ''
        fingerprint = ''
        api_errors = 0
        for attempt in range(max_retries + 1):
            temp = attempt_temps[min(attempt, len(attempt_temps) - 1)]
            s = attempt_seeds[min(attempt, len(attempt_seeds) - 1)]
            raw = self.call_raw(model=model, system_prompt=system_prompt, user_prompt=user_prompt, seed=s, temperature=temp, max_tokens=mt)
            total_in += raw.input_tokens
            total_out += raw.output_tokens
            total_ms += raw.wallclock_ms
            if not fingerprint and raw.system_fingerprint:
                fingerprint = raw.system_fingerprint
            last_text = raw.text
            if raw.api_error or not raw.text.strip():
                api_errors += 1
                last_err = f"api_error_attempt_{attempt}:{raw.api_error or 'empty_response'}"
                continue
            parsed, valid, err = parse_call_output(raw.text, call_type)
            if valid and parsed is not None:
                return CallOutcome(parsed=parsed, raw_text=raw.text, valid_json=True, retries=attempt, input_tokens=total_in, output_tokens=total_out, wallclock_ms=total_ms, fallback_used=False, extra={'system_fingerprint': fingerprint, 'model': raw.raw_meta.get('model', model)})
            last_err = err
        if self.strict_api_errors and api_errors:
            raise RuntimeError(f'{model}: {call_type} failed after API errors; episode incomplete')
        return CallOutcome(parsed=_fallback_parsed(call_type), raw_text=last_text, valid_json=False, retries=max_retries, input_tokens=total_in, output_tokens=total_out, wallclock_ms=total_ms, fallback_used=True, extra={'system_fingerprint': fingerprint, 'last_err': last_err, 'error_label': 'json_parse_failure'})

def _fallback_parsed(call_type: str) -> dict:
    if call_type == 'planner':
        return {'next_action': 'noop', 'required_preconditions': [], 'expected_effects': [], 'confidence': 0.0}
    if call_type == 'updater':
        return {'changed_facts': [], 'removed_facts': [], 'full_world_state': {'objects': {}, 'locations': {}, 'relations': [], 'inventory': [], 'open_subgoals': [], 'completed_subgoals': [], 'blocked_dependencies': [], 'beliefs': []}}
    if call_type == 'self_diag':
        return {'self_check_valid': False, 'missing_preconditions': [], 'should_replan': True}
    return {}

class ConfiguredLLMClient(LLMClient):

    def __init__(self, timeout=1200.0, **kwargs):
        kwargs.setdefault('request_timeout', timeout)
        kwargs.setdefault('fixed_temperature', 0.0)
        kwargs.setdefault('strict_api_errors', True)
        super().__init__(**kwargs)
        self.model_ids.setdefault('claude-sonnet-4-5', 'bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0')

    def call_typed(self, *args, **kwargs):
        call_type = kwargs.get('call_type', args[1] if len(args) > 1 else None)
        if kwargs.get('max_tokens') is None:
            kwargs['max_tokens'] = {'planner': 600, 'updater': 8000, 'self_diag': 400}[call_type]
        return super().call_typed(*args, **kwargs)

def call_trace_model(model, system_prompt, user_prompt, label, timeout=240.0, max_attempts=6):
    client = ConfiguredLLMClient(timeout=timeout)
    started = time.perf_counter()
    events = []
    for attempt in range(max_attempts):
        raw = client.call_raw(model, system_prompt, user_prompt, seed=None, temperature=0.0, max_tokens=8000)
        if raw.text.strip() and (not raw.api_error):
            return (raw.text, {'label': label, 'model': raw.raw_meta.get('model', model), 'transport': 'configured_api', 'total_latency_s': time.perf_counter() - started, 'successful_attempt_latency_s': raw.wallclock_ms / 1000, 'retry_events': events, 'retry_count': len(events), 'rate_limit_hits': 0, 'usage': {'prompt_tokens': raw.input_tokens, 'completion_tokens': raw.output_tokens, 'total_tokens': raw.input_tokens + raw.output_tokens}, 'finish_reason': raw.raw_meta.get('stop_reason'), 'system_fingerprint': raw.system_fingerprint})
        events.append({'attempt': attempt + 1, 'error': raw.api_error or 'empty_response'})
        if attempt + 1 < max_attempts:
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError('Model transport failed; episode is incomplete')
