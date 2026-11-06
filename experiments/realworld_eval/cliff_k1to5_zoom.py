import json, glob
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path('experiments/realworld_eval')
KS = [1, 2, 3, 4, 5, 7, 10, 15, 20]
KZOOM = [1, 2, 3, 4, 5]

def load(paths):
    rows = []
    for pat in paths:
        rows += [json.load(open(f)) for f in glob.glob(pat)]
    return rows

def ret(rows, k):
    vs = [r.get(f'mean_retention_at_{k}') for r in rows]
    vs = [x for x in vs if x is not None]
    return sum(vs) / len(vs) if vs else None
haiku = load([str(ROOT / 'runs/sw_cliff_grid/derived/cliff_v1/*/cliff_episode_metrics.json'), str(ROOT / 'runs/sw_cliff_grid_shortT_haiku/derived/cliff_v1/*/cliff_episode_metrics.json')])
sonnet = load([str(ROOT / 'runs/sw_cliff_grid_sonnet/derived/cliff_v1/*/cliff_episode_metrics.json')])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
hz = [ret(haiku, k) for k in KZOOM]
sz = [ret(sonnet, k) for k in KZOOM]
ax1.plot(KZOOM, hz, 'o-', color='#c0392b', lw=2.5, ms=9, label=f'haiku (n={len(haiku)})')
ax1.plot(KZOOM, sz, 's-', color='#27ae60', lw=2.5, ms=9, label=f'sonnet (n={len(sonnet)})')
for k, v in zip(KZOOM, hz):
    if v is not None:
        ax1.annotate(f'{v:.2f}', (k, v), textcoords='offset points', xytext=(0, -14), fontsize=8, color='#c0392b', ha='center')
ax1.axvspan(1, 3, alpha=0.12, color='red')
ax1.text(2.0, 0.55, 'cliff k=1→3\n(−44%, −50% per step)', ha='center', fontsize=8, color='#c0392b')
ax1.set_xlabel('k  (fact age in steps)', fontsize=12)
ax1.set_ylabel('Retention@k', fontsize=12)
ax1.set_xticks(KZOOM)
ax1.set_title('ZOOM k=1..5: the retention cliff (high-resolution)\nhaiku collapses in k=1→3 then floors; sonnet (prelim n=47) stays high')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_ylim(-0.03, 0.75)
hf = [ret(haiku, k) for k in KS]
sf = [ret(sonnet, k) for k in KS]
ax2.plot(KS, hf, 'o-', color='#c0392b', lw=2, ms=7, label='haiku')
ax2.plot(KS, sf, 's-', color='#27ae60', lw=2, ms=7, label='sonnet')
ax2.axvspan(1, 5, alpha=0.1, color='orange')
ax2.text(3, 0.55, 'zoom region\n(left panel)', ha='center', fontsize=8, color='#e67e22')
ax2.set_xlabel('k  (fact age in steps)', fontsize=12)
ax2.set_ylabel('Retention@k', fontsize=12)
ax2.set_title('Full curve (cross-model): haiku cliff vs sonnet plateau')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_ylim(-0.03, 0.75)
plt.tight_layout()
out = ROOT / 'runs/sw_cliff_grid/figures/cliff_k1to5_zoom.png'
plt.savefig(out, dpi=140)
print(f'[cliff_k1to5_zoom] haiku n={len(haiku)} sonnet n={len(sonnet)} -> {out}')
print('haiku k1..5:', [f'{v:.3f}' if v else 'None' for v in hz])
print('sonnet k1..5:', [f'{v:.3f}' if v else 'None' for v in sz])
