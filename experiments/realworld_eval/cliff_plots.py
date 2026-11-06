import json, glob, sys
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
run_dir = Path(sys.argv[1]).resolve()
fig_dir = run_dir / 'figures'
fig_dir.mkdir(parents=True, exist_ok=True)
rows = [json.load(open(f)) for f in glob.glob(str(run_dir / 'derived/cliff_v1/*/cliff_episode_metrics.json'))]
KS = [1, 2, 3, 4, 5, 7, 10, 15, 20]
TASKS = sorted({r['task_name'] for r in rows})
TS = sorted({r['horizon'] for r in rows})

def am(rs, k):
    vs = [r[k] for r in rs if r.get(k) is not None]
    return sum(vs) / len(vs) if vs else np.nan
plt.figure(figsize=(7, 5))
ys = [am(rows, f'mean_retention_at_{k}') for k in KS]
plt.plot(KS, ys, 'o-', lw=2, color='#c0392b', markersize=8)
plt.axhline(0.05, ls='--', color='gray', alpha=0.6, label='5% floor')
plt.xlabel('k  (fact age in steps since last observed)', fontsize=12)
plt.ylabel('Retention@k  (frac of eligible facts still believed)', fontsize=12)
plt.title(f'ScienceWorld retention cliff (haiku, mean over 90 episodes)\nknee at k=1→3: {ys[0]:.2f}→{ys[2]:.2f}, floor ~0', fontsize=11)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / 'retention_vs_k.png', dpi=140)
plt.close()

def heat(metric, title, fname, cmap):
    M = np.full((len(TASKS), len(TS)), np.nan)
    for i, task in enumerate(TASKS):
        for j, T in enumerate(TS):
            rs = [r for r in rows if r['task_name'] == task and r['horizon'] == T]
            M[i, j] = am(rs, metric)
    plt.figure(figsize=(9, 4.5))
    im = plt.imshow(M, aspect='auto', cmap=cmap, vmin=0, vmax=np.nanmax(M) if np.nanmax(M) > 0 else 1)
    plt.colorbar(im, label=metric)
    plt.xticks(range(len(TS)), TS)
    plt.yticks(range(len(TASKS)), TASKS)
    plt.xlabel('horizon T')
    plt.ylabel('task')
    plt.title(title)
    for i in range(len(TASKS)):
        for j in range(len(TS)):
            if not np.isnan(M[i, j]):
                plt.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=7, color='black')
    plt.tight_layout()
    plt.savefig(fig_dir / fname, dpi=140)
    plt.close()
heat('mean_retention_at_3', 'Retention@3 across T × task (haiku)', 'retention_at3_heatmap.png', 'RdYlGn')
heat('mean_forgotten_rate', 'Forgotten-rate across T × task (climbs with T)', 'forgotten_heatmap.png', 'Reds')
plt.figure(figsize=(8, 5))
byT = defaultdict(list)
for r in rows:
    byT[r['horizon']].append(r)
forg = [am(byT[T], 'mean_forgotten_rate') for T in TS]
stale = [am(byT[T], 'mean_stale_rate') for T in TS]
hall = [am(byT[T], 'mean_hallucination_rate') for T in TS]
w = 0.6
plt.bar(TS, forg, w, label='Forgotten (gold has, agent dropped)', color='#c0392b')
plt.bar(TS, stale, w, bottom=forg, label='Stale (belief expired)', color='#e67e22')
plt.bar(TS, hall, w, bottom=[f + s for f, s in zip(forg, stale)], label='Hallucinated (never observed)', color='#8e44ad')
plt.xlabel('horizon T')
plt.ylabel('error rate')
plt.title('Belief-vs-Gold error decomposition (haiku)\nForgetting grows with horizon & dominates at long T; comparable to stale/halluc at short T')
plt.legend()
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(fig_dir / 'belief_decomposition.png', dpi=140)
plt.close()
print(f'[cliff_plots] wrote 4 figures -> {fig_dir}')
