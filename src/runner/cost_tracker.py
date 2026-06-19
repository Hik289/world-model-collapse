"""CostTracker — implements BUDGET_PLAN §10 hard-stop triggers + Appendix B
cost_tracker.jsonl emission (every 10 episodes).

Pricing (USD per 1M tokens), as of 2026-05-30:
  gpt-4o-mini       in=$0.150  out=$0.600
  claude-haiku-4-5  in=$0.800  out=$4.000   (Anthropic published; lab proxy passes through)
  gpt-4o            in=$2.500  out=$10.000

Thresholds (BUDGET_PLAN §10.1-10.3):
  - mini   $/ep > $0.14
  - gpt-4o $/ep > $2.00
  - any model $/ep > +20% vs v2 estimate ($0.11 mini / $1.60 gpt-4o)
  - JSON parse retry rate > 8%
  - episode rerun rate > 3%
  - per-model anchor_3 JSON valid rate < 95%  (passed in Stage 2; not re-tested here)
  - input tokens/ep (Mode C) > 500k
  - output tokens/ep > 50k
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any


PRICING = {
    "gpt-4o-mini":      {"in_per_m": 0.150, "out_per_m": 0.600, "ep_cap": 0.50, "v2_est": 0.30},
    # Lab Anthropic proxy at 127.0.0.1:18801 uses Zekun's OAuth quota — actual $ = 0.
    # Published list prices were misused for estimation (P4 phantom $50.51, smoke
    # phantom $2.40). Set rates to 0 so cost_tracker reflects *billed* spend.
    # If we ever switch haiku to a paid API channel, restore list rates here.
    "claude-haiku-4-5": {"in_per_m": 0.0, "out_per_m": 0.0, "ep_cap": None, "v2_est": 0.0},
    "gpt-4o":           {"in_per_m": 2.500, "out_per_m": 10.000, "ep_cap": 2.00, "v2_est": 1.60},
    # Cross-model supplementary (Exp C, 2026-06-07).
    # Llama-3 70B Instruct via Bedrock list pricing (US East). ep_cap is a
    # defensive bound — primary budget guard is the runner-level --cost-cap.
    "meta.llama3-70b-instruct-v1:0":
        {"in_per_m": 2.650, "out_per_m": 3.500, "ep_cap": 1.50, "v2_est": 1.00},
    # Azure OpenAI routes (used by Exp C.2 when primary OpenAI quota is
    # exhausted). The deployment is gpt-4o; pricing parity with the gpt-4o
    # entry above.
    "azure:gpt-4o":
        {"in_per_m": 2.500, "out_per_m": 10.000, "ep_cap": 2.00, "v2_est": 1.60},
    "azure:gpt-4o-mini":
        {"in_per_m": 0.150, "out_per_m": 0.600, "ep_cap": 0.14, "v2_est": 0.11},
}

# §10.3 output-tokens-per-ep threshold, Regime-aware (Director condition #4, 2026-05-30):
#   - Regime I / II: 50k (original; Mode C JSON typical baseline)
#   - Regime III / IV / V: 80k (these regimes at T=40 with struct mgmt + failed ep
#     accumulation naturally exceed 50k; not prompt bloat)
# Regime is identified via the world_regime field on EpisodeContext; for safety
# we infer the threshold per-episode from the regime tag of episodes in the
# tracker (mode = majority regime tag among recorded eps), defaulting to 50k.
OUTPUT_TOK_PER_EP_THRESHOLD_BY_REGIME = {
    "I_stable":               50_000,
    "II_large":               50_000,
    "III_coupled":            80_000,
    "III_coupled_backdrop":   80_000,
    "IV_noisy":               80_000,
    "V_volatile":             80_000,
    # Stage 4 G1-trigger grid uses sc {5,10,20,40} × dd {1,2,4,6} × T=40 with the
    # stateful_puzzle env. The high-stress quadrant (sc≥20, dd≥4) naturally
    # exceeds 50k output tokens per episode (Pilot P4 stateful_puzzle dep=6 mean
    # 19.2 steps × ~3.5k tok/step ≈ 67k/ep) — analogous to Regime III. Use the
    # same 80k threshold to avoid spurious §10.3 triggers; cost in this slice is
    # $0 (haiku proxy) regardless of token volume.
    "G1_trigger_2D":          80_000,
    # Stage 5 B cross-model G1 replication on gpt-4o-mini.
    # NOTE 2026-06-07: post-reboot resume showed mini sc=20 boundary cells
    # routinely 80-95k output tokens/ep (struct mgmt + failed-ep history
    # accumulation under T=40). Raised from 80k to 200k to match ablation
    # threshold and avoid spurious §10.3 halt — paid model so cost still
    # gated by --cost-cap $300 + per-ep $/ep mini cap $0.14 (§10.1).
    "G1_trigger_2D_mini":     200_000,
    # Future Stage 6 / Full Grid placeholders.
    "G2_full_grid":           80_000,
    "G3_2d_phase":            80_000,
    # S2.5 ablation slices (Stage 5 B parallel) — haiku on stateful_puzzle
    # routinely produces 100-180k output tokens/ep under sc=10 dd=6 backdrop
    # with T=40 max; threshold lifted to 200k to avoid spurious §10.3
    # triggers. Cost is $0 (lab proxy haiku), so high token volumes are
    # economically harmless.
    "S25_ablation_T":         200_000,
    "S25_ablation_branching": 200_000,
    "S25_ablation_obs_noise": 200_000,
    "S25_ablation_mut_rate":  200_000,
    # Critical-scan fine-grained sweeps (Stage 5 B follow-up, 2026-06-06).
    # Zoom-in around sc★ (state_card 11-19) and T★ (T 22-65) phase transitions.
    # haiku via lab proxy → $0; same 200k threshold as parent ablations.
    "ExpSC_fine":             200_000,
    "ExpT_fine":              200_000,
    # Cross-harness / cross-model supplementary (Exp B + C, 2026-06-07).
    # Mode A free-form has shorter outputs than Mode C (no full_world_state
    # JSON), but Llama-3 / GPT-4o tend to be more verbose; set 200k to match
    # ablation policy.
    "Exp_B_mode_a":           200_000,
    "Exp_C1_llama3":          200_000,
    "Exp_C2_gpt4o":           200_000,
}
DEFAULT_OUTPUT_TOK_THRESHOLD = 50_000


def _threshold_for_regime(regime: str) -> int:
    """Per-regime §10.3 output-tok-per-ep threshold with safe wildcard.

    Explicit dict hits take precedence. For unknown regime tags ending in
    one of (_2d, _grid, _phase, _full_grid, _coupled, _noisy, _volatile)
    we WARN and use 80k (Regime III/IV/V default) instead of the 50k
    baseline — these naturally exceed 50k under T=40 + struct mgmt and
    would otherwise raise spurious §10 triggers (STAGE-3-017 root cause).
    """
    if regime in OUTPUT_TOK_PER_EP_THRESHOLD_BY_REGIME:
        return OUTPUT_TOK_PER_EP_THRESHOLD_BY_REGIME[regime]
    lower = regime.lower()
    for suffix in ("_2d", "_grid", "_phase", "_full_grid", "_coupled",
                   "_noisy", "_volatile", "_ablation",
                   "_ablation_t", "_ablation_branching",
                   "_ablation_obs_noise", "_ablation_mut_rate"):
        if lower.endswith(suffix):
            import sys
            print(f"[cost_tracker WARN] unknown high-stress regime "
                  f"'{regime}' -> using 80k threshold (matched suffix "
                  f"'{suffix}'); add to OUTPUT_TOK_PER_EP_THRESHOLD_BY_REGIME "
                  f"to silence.", file=sys.stderr)
            return 80_000
    return DEFAULT_OUTPUT_TOK_THRESHOLD


def jst_now() -> str:
    return datetime.now(tz=timezone(timedelta(hours=9))).isoformat()


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICING:
        return 0.0
    p = PRICING[model]
    return (input_tokens / 1_000_000.0) * p["in_per_m"] + (output_tokens / 1_000_000.0) * p["out_per_m"]


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
    world_regime: str = ""  # used for Regime-aware §10.3 threshold (cond #4)


@dataclass
class CostTracker:
    """Thread-safe per-model tracker emitting cost_tracker.jsonl every N episodes."""

    out_path: Path
    phase: str = "pilot"          # "pilot" | "full_grid"
    slice_name: str = "pilot_p0"  # human-readable slice id
    emit_every: int = 10

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _records: dict[str, list[EpisodeRecord]] = field(default_factory=dict)
    _stopped: bool = False
    _stop_reason: str = ""
    _emitted_so_far: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------
    def record_episode(self, rec: EpisodeRecord) -> None:
        """Register one finished episode and re-check stop triggers.

        Returns nothing; caller polls `is_stopped()` after each batch.
        """
        with self._lock:
            self._records.setdefault(rec.model, []).append(rec)
            n_so_far = len(self._records[rec.model])
            # Emit every N
            if n_so_far - self._emitted_so_far.get(rec.model, 0) >= self.emit_every:
                self._emit_locked(rec.model)
                self._emitted_so_far[rec.model] = n_so_far
            # Check triggers
            self._check_triggers_locked(rec.model)

    def finalize(self) -> None:
        """Emit a final cost_tracker.jsonl row for each model (closeout)."""
        with self._lock:
            for model in self._records:
                self._emit_locked(model, final=True)

    # -----------------------------------------------------------
    def _summary_locked(self, model: str) -> dict[str, Any]:
        recs = self._records.get(model, [])
        n = len(recs)
        if n == 0:
            return {}
        in_tok = sum(r.input_tokens for r in recs)
        out_tok = sum(r.output_tokens for r in recs)
        cost = sum(r.cost_usd for r in recs)
        json_retries = sum(r.json_retries_total for r in recs)
        json_calls = sum(r.json_calls_total for r in recs)
        reruns = sum(1 for r in recs if r.rerun)
        in_per_ep = in_tok / n
        out_per_ep = out_tok / n
        cost_per_ep = cost / n
        v2_est = PRICING.get(model, {}).get("v2_est", 0.0)
        deviation_pct = ((cost_per_ep - v2_est) / v2_est * 100.0) if v2_est else 0.0
        json_retry_rate = json_retries / max(1, json_calls)
        rerun_rate = reruns / n
        return {
            "n": n,
            "input_tokens_total": in_tok,
            "output_tokens_total": out_tok,
            "cost_usd_total": cost,
            "input_tokens_per_ep": in_per_ep,
            "output_tokens_per_ep": out_per_ep,
            "cost_usd_per_ep": cost_per_ep,
            "v2_estimate_per_ep": v2_est,
            "deviation_pct": deviation_pct,
            "json_retry_rate": json_retry_rate,
            "rerun_rate": rerun_rate,
        }

    def _emit_locked(self, model: str, final: bool = False) -> None:
        s = self._summary_locked(model)
        if not s:
            return
        record = {
            "ts": jst_now(),
            "phase": self.phase,
            "slice": self.slice_name,
            "model": model,
            "ep_completed_this_phase": s["n"],
            "actual_input_tok_total": s["input_tokens_total"],
            "actual_output_tok_total": s["output_tokens_total"],
            "actual_usd_so_far": round(s["cost_usd_total"], 6),
            "usd_per_ep_running_mean": round(s["cost_usd_per_ep"], 6),
            "deviation_vs_v2_estimate_pct": round(s["deviation_pct"], 2),
            "json_retry_rate_so_far": round(s["json_retry_rate"], 4),
            "ep_rerun_rate_so_far": round(s["rerun_rate"], 4),
            "triggered_stop": self._stopped,
            "stop_reason": self._stop_reason if self._stopped else None,
            "final": bool(final),
        }
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")

    def _output_tok_threshold_locked(self, model: str) -> int:
        """Regime-aware §10.3 output-tokens-per-ep threshold (Director cond #4).

        Uses majority regime tag among recorded eps; defaults to 50k if no tag.
        """
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
        # Only enforce after at least 10 ep (per spec: "跑前 10 ep 后必 cost-validate")
        if s["n"] < 10:
            return
        triggers: list[str] = []
        # 10.1
        ep_cap = PRICING.get(model, {}).get("ep_cap")
        if ep_cap is not None and s["cost_usd_per_ep"] > ep_cap:
            triggers.append(f"{model}_usd_per_ep_exceeded:{s['cost_usd_per_ep']:.4f}>{ep_cap}")
        if PRICING.get(model, {}).get("v2_est") and s["deviation_pct"] > 20.0:
            triggers.append(f"{model}_deviation_pct_exceeded:{s['deviation_pct']:.1f}%>20%")
        # 10.2
        if s["json_retry_rate"] > 0.08:
            triggers.append(f"json_retry_rate_high:{s['json_retry_rate']:.4f}>0.08")
        if s["rerun_rate"] > 0.03:
            triggers.append(f"ep_rerun_rate_high:{s['rerun_rate']:.4f}>0.03")
        # 10.3 (Mode C, Regime-aware per Director cond #4)
        if s["input_tokens_per_ep"] > 500_000:
            triggers.append(f"input_tok_per_ep_high:{s['input_tokens_per_ep']:.0f}>500000")
        out_thresh = self._output_tok_threshold_locked(model)
        if s["output_tokens_per_ep"] > out_thresh:
            triggers.append(f"output_tok_per_ep_high:{s['output_tokens_per_ep']:.0f}>{out_thresh}")
        if triggers:
            self._stopped = True
            self._stop_reason = ";".join(triggers)

    def is_stopped(self) -> bool:
        with self._lock:
            return self._stopped

    def force_stop(self, reason: str) -> None:
        """External-API forced-stop (e.g. Stage 5 cost cap hit)."""
        with self._lock:
            self._stopped = True
            self._stop_reason = (self._stop_reason + ";" if self._stop_reason else "") + reason

    def stop_reason(self) -> str:
        with self._lock:
            return self._stop_reason

    def summary_all(self) -> dict[str, Any]:
        with self._lock:
            return {m: self._summary_locked(m) for m in self._records}
