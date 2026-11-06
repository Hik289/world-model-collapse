from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import defaultdict
import statistics
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LOG_DIR = ROOT / 'data' / 'raw_logs'
HERE = Path(__file__).resolve().parent

def mslug(m):
    return m.replace('-', '_').replace('.', '_').replace('/', '_')

def _bad(step):
    return not step.get('action_valid', True) or not step.get('world_consistent', True) or (not step.get('dependency_correct', True))

def tau_a_of(steps):
    for i, s in enumerate(steps):
        if i >= 1 and (not s.get('action_valid', True)):
            return i
    return None

def tau_w_of(steps):
    prev = None
    for i, s in enumerate(steps):
        wsa = s.get('world_state_accuracy', s.get('wsa'))
        if wsa is None:
            prev = wsa
            continue
        if wsa < 0.5 and prev is not None and (prev - wsa >= 0.15):
            return i
        prev = wsa
    return None

def _facts(ws):
    if not isinstance(ws, dict):
        return set()
    out = set()
    for key in ('beliefs', 'locations', 'objects', 'inventory', 'relations', 'open_subgoals', 'completed_subgoals', 'blocked_dependencies'):
        v = ws.get(key)
        if isinstance(v, dict):
            for kk, vv in v.items():
                out.add((key, str(kk), str(vv)))
        elif isinstance(v, (list, set)):
            for item in v:
                out.add((key, str(item)))
    return out

def retention_at_k(steps, k=5):
    observed_at = {}
    num = den = 0
    for i, s in enumerate(steps):
        gold = _facts(s.get('gold_world_state'))
        agent = _facts(s.get('agent_world_state'))
        for f in gold:
            if f not in observed_at:
                observed_at[f] = i
        for f in gold:
            if f in observed_at and i - observed_at[f] >= k:
                den += 1
                if f in agent:
                    num += 1
    return num / den if den else None

def summarize_cells(completed, excluded=(), k=5):
    cells = defaultdict(list)
    for meta, steps in completed:
        cfg = meta['stress_config']
        cells[cfg['state_card'], cfg['dep_density']].append((meta, steps))
    for row in excluded:
        cfg = row['stress_config']
        cells[cfg['state_card'], cfg['dep_density']]
    output = {}
    for (ss, sd), episodes in sorted(cells.items()):
        records = [{'success': bool(meta['final_success']), 'tw': tau_w_of(steps), 'ta': tau_a_of(steps), 'ret': retention_at_k(steps, k)} for meta, steps in episodes]
        tw = [x['tw'] for x in records if x['tw'] is not None]
        ta = [x['ta'] for x in records if x['ta'] is not None]
        ret = [x['ret'] for x in records if x['ret'] is not None]
        paired = [x for x in records if not x['success'] and x['tw'] is not None and (x['ta'] is not None)]
        output[f'sc={ss},dd={sd}'] = {'n': len(records), 'success_rate': statistics.mean((x['success'] for x in records)) if records else None, 'tau_w_median': statistics.median(tw) if tw else None, 'tau_w_n': len(tw), 'tau_a_median': statistics.median(ta) if ta else None, 'tau_a_n': len(ta), f'retention_at_{k}': statistics.mean(ret) if ret else None, 'retention_n': len(ret), 'paired_n': len(paired), 'world_first': sum((x['tw'] < x['ta'] for x in paired)), 'same_step': sum((x['tw'] == x['ta'] for x in paired)), 'action_first': sum((x['tw'] > x['ta'] for x in paired)), 'lead_values': [x['ta'] - x['tw'] for x in paired]}
    return output

def main():
    from src.evaluation.jsonl_logger import load_completed_traces
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--env', required=True)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--log-dir', type=Path, default=LOG_DIR)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.k < 1:
        parser.error('--k must be positive')
    tag = f'{mslug(args.model)}_{args.env}'
    complete, excluded = load_completed_traces(args.log_dir / f'solgrid2d_{tag}_step.jsonl', args.log_dir / f'solgrid2d_{tag}_episode.jsonl')
    output = summarize_cells(complete, excluded, args.k)
    path = args.output or HERE / f'aligned_{tag}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    path.with_suffix('.audit.json').write_text(json.dumps({'model': args.model, 'environment': args.env, 'completed': len(complete), 'excluded': excluded, 'tau_w_rule': 'first zero-based step with WSA < 0.5 and a drop of at least 0.15', 'tau_a_rule': 'first invalid action at zero-based step >= 1', 'retention_rule': 'pooled gold-fact-time recall since first gold appearance, averaged across episodes', 'retention_lag': args.k}, indent=2) + '\n')
    print(f'Wrote {len(output)} cells; {len(complete)} complete episodes, {len(excluded)} excluded')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
