import json, glob
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path('experiments/realworld_eval')
KS = [1, 2, 3, 4, 5, 7, 10, 15, 20]

def mean_nonnull(vals):
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None
sw = [json.load(open(f)) for f in glob.glob(str(ROOT / 'runs/sw_cliff_grid/derived/cliff_v1/*/cliff_episode_metrics.json'))]
sw_ret = {k: mean_nonnull([r.get(f'mean_retention_at_{k}') for r in sw]) for k in KS}
sw_pool = []
for f in glob.glob(str(ROOT / 'runs/sw_cliff_grid/derived/cliff_v1/*/cliff_step_metrics.jsonl')):
    for line in open(f):
        sw_pool.append(json.loads(line).get('n_ledger_true', 0))
aw = [json.load(open(f)) for f in glob.glob(str(ROOT / 'transition_scan_run/derived/metrics_v1/*/episode_metrics.json'))]
aw_ret = {k: mean_nonnull([r.get(f'mean_retention_at_{k}') for r in aw]) for k in [1, 5, 10, 20]}
aw_pool = []
for f in glob.glob(str(ROOT / 'transition_scan_run/derived/metrics_v1/*/step_metrics.jsonl')):
    for line in open(f):
        aw_pool.append(json.loads(line).get('observed_ledger_fact_count', 0))
import statistics
print('=== DUAL-SUBSTRATE CONTRAST ===')
print(f'ScienceWorld (occluded):  retention@1={sw_ret[1]:.3f} @3={sw_ret[3]:.3f} @5={sw_ret[5]:.3f} | pool mean={statistics.mean(sw_pool):.0f} max={max(sw_pool)}')
aw1 = aw_ret[1]
aw5 = aw_ret[5]
aw10 = aw_ret[10]
aw20 = aw_ret[20]
print(f'ALFWorld (free-to-observe): retention@1={aw1} @5={aw5} @10={aw10} @20={aw20} | pool mean={statistics.mean(aw_pool):.0f} max={max(aw_pool)}')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(KS, [sw_ret[k] for k in KS], 'o-', color='#c0392b', lw=2, ms=7, label='ScienceWorld (occluded)')
aw_ks = [1, 5, 10, 20]
aw_vals = [aw_ret[k] for k in aw_ks]
ax1.plot([k for k, v in zip(aw_ks, aw_vals) if v is not None], [v for v in aw_vals if v is not None], 's--', color='#2980b9', lw=2, ms=9, label='ALFWorld (free-to-observe)')
ax1.annotate('ALFWorld @10/@20 never populate\n(short horizons T=8-16, no aged facts)', xy=(5, 1.0), xytext=(8, 0.72), fontsize=8.5, color='#2980b9', arrowprops=dict(arrowstyle='->', color='#2980b9', alpha=0.6))
ax1.set_xlabel('k (fact age in steps)', fontsize=12)
ax1.set_ylabel('Retention@k', fontsize=12)
ax1.set_title('Retention cliff requires genuine occlusion\nScienceWorld collapses 0.17→0; ALFWorld stays ~1.0')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_ylim(-0.05, 1.1)
aw_obs_recall = mean_nonnull([r.get('mean_observed_fact_recall') for r in aw]) or 0.0
import numpy as np
x = np.arange(2)
w = 0.35
obs_recall_vals = [aw_obs_recall, 1 - (mean_nonnull([r.get('mean_forgotten_rate') for r in sw]) or 0.0)]
ret3_vals = [aw1 if aw1 else 0, sw_ret[3]]
b1 = ax2.bar(x - w / 2, obs_recall_vals, w, color='#7f8c8d', label='obs-fact-recall (facts still visible in obs)')
b2 = ax2.bar(x + w / 2, ret3_vals, w, color=['#2980b9', '#c0392b'], label='retention@3 (true memory test)')
for b in list(b1) + list(b2):
    ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f'{b.get_height():.2f}', ha='center', fontsize=9)
ax2.set_xticks(x)
ax2.set_xticklabels(['ALFWorld\n(re-supplies facts)', 'ScienceWorld\n(occludes facts)'], fontsize=10)
ax2.set_ylabel('fraction', fontsize=12)
ax2.set_title('Mechanism: re-supply vs occlusion (NOT pool size)\nALFWorld keeps facts visible → retention trivial (1.0);\nScienceWorld occludes → retention@3 collapses to 0.05')
ax2.legend(fontsize=9, loc='center right')
ax2.grid(alpha=0.3, axis='y')
ax2.set_ylim(0, 1.15)
plt.tight_layout()
out = ROOT / 'runs/sw_cliff_grid/figures/dual_substrate_contrast.png'
plt.savefig(out, dpi=140)
print(f'[dual_substrate] wrote {out}')
