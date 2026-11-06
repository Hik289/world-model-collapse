from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.runner import world_state_facts
FINE_K = [1, 2, 3, 5, 10]

def _facts(ws: dict) -> set[str]:
    try:
        return world_state_facts(ws)
    except Exception:
        return set()

def episode_retention(steps: list[dict]) -> dict:
    steps = sorted(steps, key=lambda r: r.get('step', 0))
    intro: dict[str, int] = {}
    per_k_vals: dict[int, list[float]] = {k: [] for k in FINE_K}
    wsa_vals: list[float] = []
    for rec in steps:
        t = rec.get('step', 0)
        agent_facts = _facts(rec.get('agent_world_state', {}))
        gold_facts = _facts(rec.get('gold_world_state', {}))
        for f in gold_facts:
            if f not in intro:
                intro[f] = t
        if agent_facts or gold_facts:
            inter = len(agent_facts & gold_facts)
            union = len(agent_facts | gold_facts)
            wsa_vals.append(inter / union if union else 1.0)
        for k in FINE_K:
            eligible = {f for f in gold_facts if t - intro.get(f, t) >= k}
            if eligible:
                held = len(agent_facts & eligible)
                per_k_vals[k].append(held / len(eligible))
    out = {}
    for k in FINE_K:
        vals = per_k_vals[k]
        out[f'retention_at_{k}'] = sum(vals) / len(vals) if vals else None
    out['mean_wsa'] = sum(wsa_vals) / len(wsa_vals) if wsa_vals else None
    out['n_steps'] = len(steps)
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--step-log', required=True)
    ap.add_argument('--episode-log', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    step_path = Path(args.step_log)
    ep_path = Path(args.episode_log)
    from src.evaluation.jsonl_logger import load_completed_traces
    completed, excluded = load_completed_traces(args.step_log, args.episode_log)
    steps_by_ep = {str(i): steps for i, (_, steps) in enumerate(completed)}
    ep_meta = {str(i): meta for i, (meta, _) in enumerate(completed)}
    cell_eps: dict[tuple, list[dict]] = defaultdict(list)
    for tid, steps in steps_by_ep.items():
        meta = ep_meta.get(tid, {})
        sc = meta.get('stress_config', {}).get('state_card')
        dd = meta.get('stress_config', {}).get('dep_density')
        T = meta.get('stress_config', {}).get('T')
        model = meta.get('model', '?')
        ret = episode_retention(steps)
        ret['success'] = bool(meta.get('final_success', False))
        ret['task_id'] = meta['task_id']
        cell_eps[model, sc, dd, T].append(ret)
    cells = []
    for (model, sc, dd, T), eps in sorted(cell_eps.items(), key=lambda x: (str(x[0][0]), x[0][1] or 0, x[0][3] or 0)):

        def _mean(key):
            vs = [e[key] for e in eps if e.get(key) is not None]
            return sum(vs) / len(vs) if vs else None
        n = len(eps)
        succ = sum((1 for e in eps if e['success']))
        cell = {'model': model, 'state_card': sc, 'dep_density': dd, 'T': T, 'n_ep': n, 'success_rate': succ / n if n else None, 'mean_wsa': _mean('mean_wsa'), **{f'retention_at_{k}': _mean(f'retention_at_{k}') for k in FINE_K}}
        cells.append(cell)
    out = {'cells': cells, 'fine_k': FINE_K}
    out['excluded'] = excluded
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"{'model':20} {'sc':>5} {'dd':>3} {'T':>4} {'n':>3} {'succ':>6} {'ret@5':>7} {'ret@10':>7} {'WSA':>6}")
    for c in cells:
        r5 = c['retention_at_5']
        r10 = c['retention_at_10']
        w = c['mean_wsa']
        s = c['success_rate']
        print(f"{str(c['model'])[:20]:20} {str(c['state_card']):>5} {str(c['dep_density']):>3} {str(c['T']):>4} {c['n_ep']:>3} {(f'{s:.2f}' if s is not None else '-'):>6} {(f'{r5:.3f}' if r5 is not None else '-'):>7} {(f'{r10:.3f}' if r10 is not None else '-'):>7} {(f'{w:.3f}' if w is not None else '-'):>6}")
    print(f'\nWrote {args.out}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
