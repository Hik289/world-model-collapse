import json
import sys
import os
from pathlib import Path
from collections import defaultdict
import scipy.stats as stats
import numpy as np

def compute_tau_w(steps):
    for i, s in enumerate(steps):
        wsa = s.get('world_state_accuracy', 1.0)
        if i > 0:
            prev_wsa = steps[i - 1].get('world_state_accuracy', 1.0)
            if wsa < 0.5 and prev_wsa - wsa >= 0.15:
                return s['step']
    return None

def compute_tau_a(steps):
    for s in steps:
        if not s.get('action_valid', True) and s['step'] >= 1:
            return s['step']
    return None

def main():
    envs = ['stateful_puzzle', 'graph_nav', 'tool_dag']
    for env in envs:
        if env == 'stateful_puzzle':
            p = 'data/raw_logs/stage4_step.jsonl'
        else:
            p = f'data/raw_logs/stage4_{env}_step.jsonl'
        if not os.path.exists(p):
            print(f'[{env}] SKIP - no step log at {p}')
            continue
        by_tid = defaultdict(list)
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                by_tid[d['task_id']].append(d)
        n_paired = 0
        lags = []
        n_action_first = 0
        n_ws_first = 0
        n_sync = 0
        for tid, steps in by_tid.items():
            steps.sort(key=lambda x: x['step'])
            tau_w = compute_tau_w(steps)
            tau_a = compute_tau_a(steps)
            if tau_w is not None and tau_a is not None:
                n_paired += 1
                lag = tau_w - tau_a
                lags.append(lag)
                if lag < 0:
                    n_ws_first += 1
                elif lag > 0:
                    n_action_first += 1
                else:
                    n_sync += 1
        if n_paired == 0:
            print(f'[{env}] No paired events found.')
            continue
        med_lag = np.median(lags)
        try:
            res = stats.wilcoxon([-x for x in lags], alternative='greater')
            pval = res.pvalue
        except ValueError:
            pval = 1.0
        print(f'=== {env} ===')
        print(f'  n_paired: {n_paired}')
        print(f'  median lag: {med_lag} (negative = WS first)')
        print(f'  Wilcoxon p-val (WS<Action): {pval:.2e}')
        print(f'  Breakdown: WS-first={n_ws_first} Action-first={n_action_first} Sync={n_sync}')
        print()
if __name__ == '__main__':
    main()
