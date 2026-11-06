import runpy
import sys
from pathlib import Path
sys.path.insert(0, 'analysis/scripts')
import _common
import tier2_lags
import numpy as np
import scipy.stats as stats

def test_env(env, path):
    print(f'=== {env} ===')
    try:
        steps_by_tid = _common.load_step_episodes(path)
    except FileNotFoundError:
        print('Not found')
        return
    rows = []
    for tid, steps in steps_by_tid.items():
        w = _common.first_combined_ws_collapse(steps)
        a = _common.first_plan_collapse_step(steps)
        if w is not None and a is not None:
            lag = w - a
            rows.append({'task_id': tid, 'tau_w': w, 'tau_a': a, 'lag': lag})
    if not rows:
        print('No paired rows')
        return
    print(f'  n_paired: {len(rows)}')
    lags = [r['lag'] for r in rows]
    med = np.median(lags)
    print(f'  median lag: {med} (neg = WS first)')
    n_ws_first = sum((1 for x in lags if x < 0))
    n_act_first = sum((1 for x in lags if x > 0))
    n_sync = sum((1 for x in lags if x == 0))
    res = stats.wilcoxon([-x for x in lags], alternative='greater')
    print(f'  WS-first: {n_ws_first}, Act-first: {n_act_first}, Sync: {n_sync}')
    print(f'  wilcoxon p: {res.pvalue:.2e}')
test_env('graph_nav', 'data/raw_logs/stage4_graph_nav_step.jsonl')
test_env('tool_dag', 'data/raw_logs/stage4_tool_dag_step.jsonl')
test_env('stateful_puzzle', 'data/raw_logs/stage4_step.jsonl')
