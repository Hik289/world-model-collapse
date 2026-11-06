from __future__ import annotations
import json, math
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'raw_logs'
ANALYSIS = ROOT / 'analysis'
WSA_THR = 0.5
SCA_THR = 0.3
SCA_WINDOW = 5
JST = timezone(timedelta(hours=9))

def jst_now() -> str:
    return datetime.now(tz=JST).isoformat()

def load_step_episodes(rel_path='data/raw_logs/stage4_step.jsonl'):
    path = ROOT / rel_path
    by_t = defaultdict(list)
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            by_t[d['task_id']].append(d)
    for tid in by_t:
        by_t[tid].sort(key=lambda x: x['step'])
    return dict(by_t)

def load_episode_jsonl(rel_path):
    out = []
    with open(ROOT / rel_path) as f:
        for line in f:
            out.append(json.loads(line))
    return out

def parse_stage4_task_id(task_id: str):
    parts = task_id.split('_')
    sc = int(parts[-3][2:])
    dd = int(parts[-2][2:])
    t_idx = int(parts[-1][1:])
    return (sc, dd, t_idx)

def _rolling_mean(seq, window):
    out = [None] * len(seq)
    cumsum = 0.0
    for i, v in enumerate(seq):
        cumsum += v
        if i >= window:
            cumsum -= seq[i - window]
        if i >= window - 1:
            out[i] = cumsum / window
    return out

def first_ws_collapse_step(steps):
    for s in steps:
        if s.get('world_state_accuracy') is not None and s['world_state_accuracy'] < WSA_THR:
            return s['step']
    return None

def first_sca_collapse_step(steps):
    sca_seq = [1.0 if s.get('self_check_correct') else 0.0 for s in steps]
    rm = _rolling_mean(sca_seq, SCA_WINDOW)
    for i, v in enumerate(rm):
        if v is not None and v < SCA_THR:
            return steps[i]['step']
    return None

def first_combined_ws_collapse(steps):
    a = first_ws_collapse_step(steps)
    b = first_sca_collapse_step(steps)
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)

def first_plan_collapse_step(steps):
    for s in steps:
        if s['step'] < 1:
            continue
        if not s.get('action_valid', True):
            return s['step']
    return None

def percentile_bootstrap_ci(values, stat_fn, B=10000, ci_pct=0.99, rng_seed=20260607):
    if len(values) == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(rng_seed)
    arr = np.asarray(values)
    n = len(arr)
    boots = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boots[b] = stat_fn(arr[idx])
    alpha = 1 - ci_pct
    return (float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2))))

def cluster_bootstrap_ci(cluster_index, values, stat_fn, B=10000, ci_pct=0.95, rng_seed=20260607):
    rng = np.random.default_rng(rng_seed)
    clusters = np.asarray(cluster_index)
    vals = np.asarray(values)
    uniq = np.unique(clusters)
    idx_by_cluster = {c: np.where(clusters == c)[0] for c in uniq}
    K = len(uniq)
    boots = np.empty(B)
    for b in range(B):
        chosen = rng.integers(0, K, size=K)
        sample_idx = np.concatenate([idx_by_cluster[uniq[c]] for c in chosen])
        boots[b] = stat_fn(vals[sample_idx])
    alpha = 1 - ci_pct
    return (float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2))), boots)

def wilson_ci(k: int, n: int, ci_pct: float=0.95):
    if n == 0:
        return (0.0, 1.0)
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - ci_pct) / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(x) for x in o]
        if hasattr(o, 'item'):
            return o.item()
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return o
    Path(path).write_text(json.dumps(_clean(obj), indent=2, sort_keys=True, ensure_ascii=False))

def cell_key(sc: int, dd: int) -> str:
    return f'sc={sc},dd={dd}'
