"""LLM API client + 3-call wrapper with JSON validation + retry.

Model tiers (EXP_PLAN §1):
  - gpt-4o-mini (OpenAI, primary)
  - claude-haiku-4-5 (Anthropic via local proxy http://127.0.0.1:18801)
  - gpt-4o (OpenAI, Pilot anchor_3 only)

Constraints:
  - OpenAI: pass `seed` for deterministic sampling. Report `system_fingerprint`.
  - Anthropic: pass `temperature=0.0` (proxy supports messages API; no `seed`).
  - Production calls: up to 3 retries with mild prompt re-temperature. 4th
    attempt fallback marks `valid_json=False` and labels error_label.
  - anchor_3 warm-up: 0 retries (raw stability test). Use `call_raw()`.

Per-call telemetry returned: input_tokens, output_tokens, wallclock_ms,
system_fingerprint (OpenAI only), retries.
"""

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

# Bedrock (lazy import — only when a bedrock model is requested)
try:
    import boto3  # type: ignore
    _BOTO3_AVAILABLE = True
except Exception:  # pragma: no cover - we'll error at call time
    _BOTO3_AVAILABLE = False


# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------

ANTHROPIC_PROXY_URL = os.environ.get(
    "ANTHROPIC_PROXY_URL", "http://127.0.0.1:18801/v1/messages"
)
ANTHROPIC_VERSION = "2023-06-01"

# Default token limits per call type (kept modest to bound cost during anchor_3)
DEFAULT_MAX_TOKENS = {
    "planner": 300,
    "updater": 1200,  # updater outputs full world state
    "self_diag": 200,
}


@dataclass
class RawCallResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    wallclock_ms: int = 0
    system_fingerprint: str = ""
    raw_meta: dict = field(default_factory=dict)
    api_error: str = ""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMClient:
    """Cross-provider client used by anchor_3 verifier and production agents."""

    def __init__(
        self,
        openai_api_key: str | None = None,
        anthropic_proxy_url: str = ANTHROPIC_PROXY_URL,
        request_timeout: float = 60.0,
    ):
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in env")
        self.oai = OpenAI(api_key=api_key, timeout=request_timeout)
        self.anthropic_url = anthropic_proxy_url
        self.timeout = request_timeout
        self._bedrock = None  # lazy initialised on first bedrock call
        # Azure OpenAI fallback (used for `azure:<deployment>` model strings).
        # Used by cross-harness Exp C.2 when primary OpenAI key is rate-limited.
        self._azure: OpenAI | None = None
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_key = os.environ.get("AZURE_OPENAI_KEY")
        if azure_endpoint and azure_key:
            try:
                self._azure = OpenAI(
                    api_key=azure_key,
                    base_url=azure_endpoint,
                    timeout=request_timeout,
                    default_query={"api-version": "preview"},
                    default_headers={"api-key": azure_key},
                )
            except Exception:
                self._azure = None

    # -------------------------------------------------------------------
    # Raw call (no retry, no JSON validation) — for anchor_3 stability test
    # and seed parity test.
    # -------------------------------------------------------------------
    def call_raw(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        seed: int | None,
        temperature: float,
        max_tokens: int,
    ) -> RawCallResult:
        if model.startswith("azure:"):
            return self._azure_raw(model.split(":", 1)[1], system_prompt, user_prompt, seed, temperature, max_tokens)
        if model.startswith("gpt-"):
            return self._openai_raw(model, system_prompt, user_prompt, seed, temperature, max_tokens)
        if model.startswith("claude-"):
            return self._anthropic_raw(model, system_prompt, user_prompt, temperature, max_tokens)
        if model.startswith("meta.") or model.startswith("us.meta."):
            return self._bedrock_raw(model, system_prompt, user_prompt, temperature, max_tokens)
        raise ValueError(f"unsupported model {model}")

    def _azure_raw(
        self, deployment: str, system_prompt: str, user_prompt: str,
        seed: int | None, temperature: float, max_tokens: int,
    ) -> RawCallResult:
        if self._azure is None:
            return RawCallResult(text="", api_error="azure_not_configured")
        t0 = time.perf_counter()
        try:
            kwargs: dict[str, Any] = {
                "model": deployment,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if seed is not None:
                kwargs["seed"] = seed
            resp = self._azure.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            return RawCallResult(
                text=text,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                wallclock_ms=int((time.perf_counter() - t0) * 1000),
                system_fingerprint=getattr(resp, "system_fingerprint", "") or "",
                raw_meta={"id": getattr(resp, "id", ""), "model": getattr(resp, "model", deployment)},
            )
        except Exception as e:
            return RawCallResult(
                text="",
                wallclock_ms=int((time.perf_counter() - t0) * 1000),
                api_error=f"{type(e).__name__}: {e}",
            )

    def _openai_raw(
        self, model: str, system_prompt: str, user_prompt: str,
        seed: int | None, temperature: float, max_tokens: int,
    ) -> RawCallResult:
        t0 = time.perf_counter()
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if seed is not None:
                kwargs["seed"] = seed
            resp = self.oai.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            return RawCallResult(
                text=text,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                wallclock_ms=int((time.perf_counter() - t0) * 1000),
                system_fingerprint=getattr(resp, "system_fingerprint", "") or "",
                raw_meta={"id": resp.id, "model": resp.model},
            )
        except Exception as e:
            return RawCallResult(
                text="",
                wallclock_ms=int((time.perf_counter() - t0) * 1000),
                api_error=f"{type(e).__name__}: {e}",
            )

    def _anthropic_raw(
        self, model: str, system_prompt: str, user_prompt: str,
        temperature: float, max_tokens: int,
    ) -> RawCallResult:
        t0 = time.perf_counter()
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
        }
        try:
            r = requests.post(self.anthropic_url, json=payload, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            text_chunks = [c.get("text", "") for c in data.get("content", []) if c.get("type") == "text"]
            text = "".join(text_chunks)
            usage = data.get("usage", {})
            return RawCallResult(
                text=text,
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                wallclock_ms=int((time.perf_counter() - t0) * 1000),
                system_fingerprint="",
                raw_meta={"id": data.get("id", ""), "model": data.get("model", model)},
            )
        except Exception as e:
            return RawCallResult(
                text="",
                wallclock_ms=int((time.perf_counter() - t0) * 1000),
                api_error=f"{type(e).__name__}: {e}",
            )

    # -------------------------------------------------------------------
    # Bedrock raw call (Llama-3 70B Instruct + future models).
    # -------------------------------------------------------------------
    def _ensure_bedrock(self):
        if self._bedrock is not None:
            return self._bedrock
        if not _BOTO3_AVAILABLE:
            raise RuntimeError("boto3 not installed; cannot use Bedrock models")
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        # boto3 auto-picks up AWS_BEARER_TOKEN_BEDROCK if exported, OR
        # AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY (preferred for sigv4).
        self._bedrock = boto3.client(
            "bedrock-runtime",
            region_name=region,
        )
        return self._bedrock

    def _bedrock_raw(
        self, model: str, system_prompt: str, user_prompt: str,
        temperature: float, max_tokens: int,
    ) -> RawCallResult:
        t0 = time.perf_counter()
        try:
            client = self._ensure_bedrock()
        except Exception as e:
            return RawCallResult(text="", api_error=f"bedrock_init_error: {e}")

        # Llama-3 prompt format (Instruct, with chat template)
        if "llama3" in model or "llama-3" in model:
            prompt = (
                "<|begin_of_text|>"
                "<|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_prompt}<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_prompt}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            body = {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": max(temperature, 0.01),  # bedrock llama requires >0
                "top_p": 0.9,
            }
        else:
            return RawCallResult(
                text="",
                api_error=f"unsupported bedrock model format: {model}",
            )

        try:
            resp = client.invoke_model(
                modelId=model,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(resp["body"].read())
            text = payload.get("generation", "") or ""
            in_tok = int(payload.get("prompt_token_count", 0))
            out_tok = int(payload.get("generation_token_count", 0))
            return RawCallResult(
                text=text,
                input_tokens=in_tok,
                output_tokens=out_tok,
                wallclock_ms=int((time.perf_counter() - t0) * 1000),
                system_fingerprint="",
                raw_meta={
                    "model": model,
                    "stop_reason": payload.get("stop_reason", ""),
                },
            )
        except Exception as e:
            return RawCallResult(
                text="",
                wallclock_ms=int((time.perf_counter() - t0) * 1000),
                api_error=f"{type(e).__name__}: {e}",
            )

    # -------------------------------------------------------------------
    # Production call: validate JSON + up to 3 retries.
    # -------------------------------------------------------------------
    def call_typed(
        self,
        model: str,
        call_type: str,
        system_prompt: str,
        user_prompt: str,
        seed: int | None,
        max_retries: int = 3,
        base_temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> CallOutcome:
        if call_type not in ("planner", "updater", "self_diag"):
            raise ValueError(f"unknown call_type {call_type}")
        mt = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS[call_type]

        # For Anthropic / Bedrock, ignore seed and use temperature=0.0 in production.
        # Azure routes to OpenAI under the hood so it accepts seed.
        is_anthropic = model.startswith("claude-")
        is_bedrock = model.startswith("meta.") or model.startswith("us.meta.")
        no_seed = is_anthropic or is_bedrock
        attempt_temps = (
            [0.0] * (max_retries + 1) if no_seed
            else [base_temperature, base_temperature + 0.1, base_temperature + 0.2, base_temperature + 0.3]
        )
        attempt_seeds: list[int | None] = (
            [None] * (max_retries + 1) if no_seed
            else [seed, (seed or 0) + 1, (seed or 0) + 2, (seed or 0) + 3]
        )

        total_in = 0
        total_out = 0
        total_ms = 0
        last_text = ""
        last_err = ""
        fingerprint = ""

        for attempt in range(max_retries + 1):
            temp = attempt_temps[min(attempt, len(attempt_temps) - 1)]
            s = attempt_seeds[min(attempt, len(attempt_seeds) - 1)]
            raw = self.call_raw(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                seed=s,
                temperature=temp,
                max_tokens=mt,
            )
            total_in += raw.input_tokens
            total_out += raw.output_tokens
            total_ms += raw.wallclock_ms
            if not fingerprint and raw.system_fingerprint:
                fingerprint = raw.system_fingerprint
            last_text = raw.text
            if raw.api_error:
                last_err = f"api_error_attempt_{attempt}:{raw.api_error}"
                continue
            parsed, valid, err = parse_call_output(raw.text, call_type)
            if valid and parsed is not None:
                return CallOutcome(
                    parsed=parsed,
                    raw_text=raw.text,
                    valid_json=True,
                    retries=attempt,
                    input_tokens=total_in,
                    output_tokens=total_out,
                    wallclock_ms=total_ms,
                    fallback_used=False,
                    extra={"system_fingerprint": fingerprint},
                )
            last_err = err

        # All attempts failed → return fallback.
        return CallOutcome(
            parsed=_fallback_parsed(call_type),
            raw_text=last_text,
            valid_json=False,
            retries=max_retries,
            input_tokens=total_in,
            output_tokens=total_out,
            wallclock_ms=total_ms,
            fallback_used=True,
            extra={"system_fingerprint": fingerprint, "last_err": last_err, "error_label": "json_parse_failure"},
        )


def _fallback_parsed(call_type: str) -> dict:
    """Default safe payload when JSON parsing fails all retries (§3.2)."""
    if call_type == "planner":
        return {
            "next_action": "noop",
            "required_preconditions": [],
            "expected_effects": [],
            "confidence": 0.0,
        }
    if call_type == "updater":
        return {
            "changed_facts": [],
            "removed_facts": [],
            "full_world_state": {
                "objects": {}, "locations": {}, "relations": [], "inventory": [],
                "open_subgoals": [], "completed_subgoals": [],
                "blocked_dependencies": [], "beliefs": [],
            },
        }
    if call_type == "self_diag":
        return {
            "self_check_valid": False,
            "missing_preconditions": [],
            "should_replan": True,
        }
    return {}
