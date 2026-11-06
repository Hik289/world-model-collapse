from __future__ import annotations
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
PRICING = {'gpt-4o-mini': {'in_per_m': 0.15, 'out_per_m': 0.6, 'ep_cap': 0.5, 'v2_est': 0.3}, 'claude-haiku-4-5': {'in_per_m': 0.0, 'out_per_m': 0.0, 'ep_cap': None, 'v2_est': 0.0}, 'gpt-4o': {'in_per_m': 2.5, 'out_per_m': 10.0, 'ep_cap': 2.0, 'v2_est': 1.6}, 'meta.llama3-70b-instruct-v1:0': {'in_per_m': 2.65, 'out_per_m': 3.5, 'ep_cap': 1.5, 'v2_est': 1.0}, 'azure:gpt-4o': {'in_per_m': 2.5, 'out_per_m': 10.0, 'ep_cap': 2.0, 'v2_est': 1.6}, 'azure:gpt-4o-mini': {'in_per_m': 0.15, 'out_per_m': 0.6, 'ep_cap': 0.14, 'v2_est': 0.11}}
OUTPUT_TOK_PER_EP_THRESHOLD_BY_REGIME = {'I_stable': 50000, 'II_large': 50000, 'III_coupled': 80000, 'III_coupled_backdrop': 80000, 'IV_noisy': 80000, 'V_volatile': 80000, 'G1_trigger_2D': 80000, 'G1_trigger_2D_mini': 200000, 'G2_full_grid': 80000, 'G3_2d_phase': 80000, 'S25_ablation_T': 200000, 'S25_ablation_branching': 200000, 'S25_ablation_obs_noise': 200000, 'S25_ablation_mut_rate': 200000, 'ExpSC_fine': 200000, 'ExpT_fine': 200000, 'Exp_B_mode_a': 200000, 'Exp_C1_llama3': 200000, 'Exp_C2_gpt4o': 200000}
DEFAULT_OUTPUT_TOK_THRESHOLD = 50000

def _threshold_for_regime(regime: str) -> int:
    if regime in OUTPUT_TOK_PER_EP_THRESHOLD_BY_REGIME:
        return OUTPUT_TOK_PER_EP_THRESHOLD_BY_REGIME[regime]
    lower = regime.lower()
    for suffix in ('_2d', '_grid', '_phase', '_full_grid', '_coupled', '_noisy', '_volatile', '_ablation', '_ablation_t', '_ablation_branching', '_ablation_obs_noise', '_ablation_mut_rate'):
        if lower.endswith(suffix):
            import sys
            print(f"[cost_tracker WARN] unknown high-stress regime '{regime}' -> using 80k threshold (matched suffix '{suffix}'); add to OUTPUT_TOK_PER_EP_THRESHOLD_BY_REGIME to silence.", file=sys.stderr)
            return 80000
    return DEFAULT_OUTPUT_TOK_THRESHOLD

def jst_now() -> str:
    return datetime.now(tz=timezone(timedelta(hours=9))).isoformat()

def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICING:
        return 0.0
    p = PRICING[model]
    return input_tokens / 1000000.0 * p['in_per_m'] + output_tokens / 1000000.0 * p['out_per_m']

@dataclass
class EpisodeRecord:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    json_retries_total: int
    json_calls_total: int
    n_steps: int
    success: bool
    rerun: bool = False
    world_regime: str = ''

@dataclass
class CostTracker:
    out_path: Path
    phase: str = 'pilot'
    slice_name: str = 'pilot_p0'
    emit_every: int = 10
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _records: dict[str, list[EpisodeRecord]] = field(default_factory=dict)
    _stopped: bool = False
    _stop_reason: str = ''
    _emitted_so_far: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def record_episode(self, rec: EpisodeRecord) -> None:
        with self._lock:
            self._records.setdefault(rec.model, []).append(rec)
            n_so_far = len(self._records[rec.model])
            if n_so_far - self._emitted_so_far.get(rec.model, 0) >= self.emit_every:
                self._emit_locked(rec.model)
                self._emitted_so_far[rec.model] = n_so_far
            self._check_triggers_locked(rec.model)

    def finalize(self) -> None:
        with self._lock:
            for model in self._records:
                self._emit_locked(model, final=True)

    def _summary_locked(self, model: str) -> dict[str, Any]:
        recs = self._records.get(model, [])
        n = len(recs)
        if n == 0:
            return {}
        in_tok = sum((r.input_tokens for r in recs))
        out_tok = sum((r.output_tokens for r in recs))
        cost = sum((r.cost_usd for r in recs))
        json_retries = sum((r.json_retries_total for r in recs))
        json_calls = sum((r.json_calls_total for r in recs))
        reruns = sum((1 for r in recs if r.rerun))
        in_per_ep = in_tok / n
        out_per_ep = out_tok / n
        cost_per_ep = cost / n
        v2_est = PRICING.get(model, {}).get('v2_est', 0.0)
        deviation_pct = (cost_per_ep - v2_est) / v2_est * 100.0 if v2_est else 0.0
        json_retry_rate = json_retries / max(1, json_calls)
        rerun_rate = reruns / n
        return {'n': n, 'input_tokens_total': in_tok, 'output_tokens_total': out_tok, 'cost_usd_total': cost, 'input_tokens_per_ep': in_per_ep, 'output_tokens_per_ep': out_per_ep, 'cost_usd_per_ep': cost_per_ep, 'v2_estimate_per_ep': v2_est, 'deviation_pct': deviation_pct, 'json_retry_rate': json_retry_rate, 'rerun_rate': rerun_rate}

    def _emit_locked(self, model: str, final: bool=False) -> None:
        s = self._summary_locked(model)
        if not s:
            return
        record = {'ts': jst_now(), 'phase': self.phase, 'slice': self.slice_name, 'model': model, 'ep_completed_this_phase': s['n'], 'actual_input_tok_total': s['input_tokens_total'], 'actual_output_tok_total': s['output_tokens_total'], 'actual_usd_so_far': round(s['cost_usd_total'], 6), 'usd_per_ep_running_mean': round(s['cost_usd_per_ep'], 6), 'deviation_vs_v2_estimate_pct': round(s['deviation_pct'], 2), 'json_retry_rate_so_far': round(s['json_retry_rate'], 4), 'ep_rerun_rate_so_far': round(s['rerun_rate'], 4), 'triggered_stop': self._stopped, 'stop_reason': self._stop_reason if self._stopped else None, 'final': bool(final)}
        with self.out_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n')

    def _output_tok_threshold_locked(self, model: str) -> int:
        from collections import Counter
        recs = self._records.get(model, [])
        regimes = [r.world_regime for r in recs if r.world_regime]
        if not regimes:
            return DEFAULT_OUTPUT_TOK_THRESHOLD
        majority = Counter(regimes).most_common(1)[0][0]
        return _threshold_for_regime(majority)

    def _check_triggers_locked(self, model: str) -> None:
        s = self._summary_locked(model)
        if not s:
            return
        if s['n'] < 10:
            return
        triggers: list[str] = []
        ep_cap = PRICING.get(model, {}).get('ep_cap')
        if ep_cap is not None and s['cost_usd_per_ep'] > ep_cap:
            triggers.append(f"{model}_usd_per_ep_exceeded:{s['cost_usd_per_ep']:.4f}>{ep_cap}")
        if PRICING.get(model, {}).get('v2_est') and s['deviation_pct'] > 20.0:
            triggers.append(f"{model}_deviation_pct_exceeded:{s['deviation_pct']:.1f}%>20%")
        if s['json_retry_rate'] > 0.08:
            triggers.append(f"json_retry_rate_high:{s['json_retry_rate']:.4f}>0.08")
        if s['rerun_rate'] > 0.03:
            triggers.append(f"ep_rerun_rate_high:{s['rerun_rate']:.4f}>0.03")
        if s['input_tokens_per_ep'] > 500000:
            triggers.append(f"input_tok_per_ep_high:{s['input_tokens_per_ep']:.0f}>500000")
        out_thresh = self._output_tok_threshold_locked(model)
        if s['output_tokens_per_ep'] > out_thresh:
            triggers.append(f"output_tok_per_ep_high:{s['output_tokens_per_ep']:.0f}>{out_thresh}")
        if triggers:
            self._stopped = True
            self._stop_reason = ';'.join(triggers)

    def is_stopped(self) -> bool:
        with self._lock:
            return self._stopped

    def force_stop(self, reason: str) -> None:
        with self._lock:
            self._stopped = True
            self._stop_reason = (self._stop_reason + ';' if self._stop_reason else '') + reason

    def stop_reason(self) -> str:
        with self._lock:
            return self._stop_reason

    def summary_all(self) -> dict[str, Any]:
        with self._lock:
            return {m: self._summary_locked(m) for m in self._records}
