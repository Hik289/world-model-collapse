from __future__ import annotations
import argparse
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _PROJ)
sys.path.insert(0, _HERE)
import alfworld_agent_runner as R
SURVEY = Path(_HERE) / 'sc_survey.json'

def build_buckets(rows, n_buckets):
    scs = [r['sc_facts'] for r in rows]
    lo, hi = (min(scs), max(scs))
    width = (hi - lo) / n_buckets
    buckets = []
    for b in range(n_buckets):
        b_lo = lo + b * width
        b_hi = hi if b == n_buckets - 1 else lo + (b + 1) * width
        members = [r for r in rows if b_lo <= r['sc_facts'] <= b_hi] if b == n_buckets - 1 else [r for r in rows if b_lo <= r['sc_facts'] < b_hi]
        buckets.append({'idx': b, 'lo': b_lo, 'hi': b_hi, 'members': members})
    return buckets

def sample_bucket(members, k, rng):
    if len(members) <= k:
        return list(members)
    by_type: dict[str, list] = {}
    for m in members:
        by_type.setdefault(m['task_type'], []).append(m)
    for v in by_type.values():
        rng.shuffle(v)
    picks = []
    types = sorted(by_type)
    ti = 0
    while len(picks) < k and any(by_type.values()):
        t = types[ti % len(types)]
        if by_type[t]:
            picks.append(by_type[t].pop())
        ti += 1
    return picks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--per-bucket', type=int, default=5)
    ap.add_argument('--buckets', type=int, default=7)
    ap.add_argument('--max-steps', type=int, default=40)
    ap.add_argument('--model-key', default='mini')
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / 'checkpoint.json'
    plan_path = run_dir / 'scan_plan.json'
    rows = json.loads(SURVEY.read_text())
    rng = random.Random(args.seed)
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        buckets = build_buckets(rows, args.buckets)
        plan = {'buckets': [], 'episodes': []}
        for b in buckets:
            picks = sample_bucket(b['members'], args.per_bucket, rng)
            plan['buckets'].append({'idx': b['idx'], 'lo': b['lo'], 'hi': b['hi'], 'n_members': len(b['members']), 'n_sampled': len(picks)})
            for m in picks:
                eid = f"{args.model_key}_b{b['idx']}_sc{m['sc_facts']}_{m['task_type']}_{os.path.basename(os.path.dirname(m['game_file']))[:12]}"
                plan['episodes'].append({'episode_id': eid, 'bucket': b['idx'], 'sc_facts': m['sc_facts'], 'task_type': m['task_type'], 'split': m['split'], 'game_file': m['game_file']})
        plan_path.write_text(json.dumps(plan, indent=2))
    ckpt = json.loads(ckpt_path.read_text()) if ckpt_path.exists() else {'done': []}
    done = set(ckpt['done'])
    print(f"COARSE_SCAN plan: {len(plan['episodes'])} episodes, {args.buckets} buckets, {len(done)} already done", flush=True)
    for b in plan['buckets']:
        print(f"  bucket {b['idx']}: SC[{b['lo']:.0f},{b['hi']:.0f}] members={b['n_members']} sampled={b['n_sampled']}", flush=True)
    for ep in plan['episodes']:
        if ep['episode_id'] in done:
            continue
        print(f"RUN {ep['episode_id']} sc={ep['sc_facts']} {ep['task_type']}", flush=True)
        try:
            R.run_episode(ep['game_file'], run_dir, model_key=args.model_key, max_steps=args.max_steps, episode_id=ep['episode_id'])
        except Exception as exc:
            print(f"EPISODE_FAILED {ep['episode_id']}: {exc!r}", flush=True)
        done.add(ep['episode_id'])
        ckpt['done'] = sorted(done)
        ckpt_path.write_text(json.dumps(ckpt, indent=2))
    print('RUNNING METRICS', flush=True)
    subprocess.run([sys.executable, 'experiments/trace_evaluation/trace_v1_metrics.py', str(run_dir)], cwd=_PROJ, check=True)
    aggregate(run_dir, plan)

def aggregate(run_dir: Path, plan: dict):
    coll = json.loads((run_dir / 'derived/metrics_v1/all_episode_metrics.json').read_text())
    by_eid = {e['episode_id']: e for e in coll['episodes']}
    ep_bucket = {e['episode_id']: e for e in plan['episodes']}
    buckets: dict[int, dict] = {}
    per_episode = []
    for eid, m in by_eid.items():
        meta = ep_bucket.get(eid)
        if not meta:
            continue
        b = meta['bucket']
        buckets.setdefault(b, {'ret5': [], 'bp': [], 'ur': [], 'cov': [], 'sc': [], 'succ': []})
        r5 = m.get('mean_retention_at_5')
        if r5 is not None:
            buckets[b]['ret5'].append(r5)
        if m.get('mean_belief_precision') is not None:
            buckets[b]['bp'].append(m['mean_belief_precision'])
        if m.get('mean_new_observation_update_recall') is not None:
            buckets[b]['ur'].append(m['mean_new_observation_update_recall'])
        if m.get('mean_exploration_coverage') is not None:
            buckets[b]['cov'].append(m['mean_exploration_coverage'])
        buckets[b]['sc'].append(meta['sc_facts'])
        buckets[b]['succ'].append(m.get('success'))
        per_episode.append({'episode_id': eid, 'bucket': b, 'sc_facts': meta['sc_facts'], 'task_type': meta['task_type'], 'retention_at_5': r5, 'belief_precision': m.get('mean_belief_precision'), 'update_recall': m.get('mean_new_observation_update_recall'), 'coverage': m.get('mean_exploration_coverage'), 'success': m.get('success')})
    binfo = {b['idx']: b for b in plan['buckets']}
    summary = []
    for b in sorted(buckets):
        d = buckets[b]
        summary.append({'bucket': b, 'sc_range': [round(binfo[b]['lo'], 1), round(binfo[b]['hi'], 1)], 'n_episodes': len(d['sc']), 'mean_sc': round(statistics.mean(d['sc']), 1) if d['sc'] else None, 'mean_retention_at_5': round(statistics.mean(d['ret5']), 4) if d['ret5'] else None, 'n_ret5_defined': len(d['ret5']), 'mean_belief_precision': round(statistics.mean(d['bp']), 4) if d['bp'] else None, 'mean_update_recall': round(statistics.mean(d['ur']), 4) if d['ur'] else None, 'mean_coverage': round(statistics.mean(d['cov']), 4) if d['cov'] else None, 'success_rate': round(sum((bool(x) for x in d['succ'])) / len(d['succ']), 4) if d['succ'] else None})
    out = {'buckets': summary, 'per_episode': sorted(per_episode, key=lambda x: (x['bucket'], x['sc_facts']))}
    (run_dir / 'coarse_scan_summary.json').write_text(json.dumps(out, indent=2))
    print('\n===== COARSE SCAN: Retention@5 vs SC bucket =====', flush=True)
    for s in summary:
        print(f"  b{s['bucket']} SC{s['sc_range']} n={s['n_episodes']} meanSC={s['mean_sc']} ret@5={s['mean_retention_at_5']} (defined {s['n_ret5_defined']}) bp={s['mean_belief_precision']} ur={s['mean_update_recall']} cov={s['mean_coverage']} succ={s['success_rate']}", flush=True)
    plot(run_dir, summary)

def plot(run_dir: Path, summary):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f'PLOT_SKIP matplotlib unavailable: {exc!r}', flush=True)
        return
    pts = [s for s in summary if s['mean_retention_at_5'] is not None]
    xs = [s['mean_sc'] for s in pts]
    ys = [s['mean_retention_at_5'] for s in pts]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(xs, ys, 'o-', color='crimson', lw=2, ms=8, label='Retention@5')
    ax1.set_xlabel('State Cardinality (start gold-fact count)')
    ax1.set_ylabel('mean Retention@5', color='crimson')
    ax1.set_ylim(0, 1.02)
    ax1.tick_params(axis='y', labelcolor='crimson')
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    bp = [s['mean_belief_precision'] for s in pts]
    cov = [s['mean_coverage'] for s in pts]
    ax2.plot(xs, bp, 's--', color='steelblue', alpha=0.7, label='Belief Prec.')
    ax2.plot(xs, cov, '^:', color='seagreen', alpha=0.7, label='Coverage')
    ax2.set_ylabel('Belief Precision / Coverage')
    ax2.set_ylim(0, 1.02)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=9)
    ax1.set_title('gpt-5.4-mini: world-model maintenance vs State Cardinality\n(ALFWorld valid_seen+valid_unseen, coarse scan)')
    fig.tight_layout()
    out = run_dir / 'retention_vs_sc.png'
    fig.savefig(out, dpi=130)
    print(f'PLOT_SAVED {out}', flush=True)
if __name__ == '__main__':
    main()
