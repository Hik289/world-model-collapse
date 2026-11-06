from __future__ import annotations
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
SC_VALUES = [10, 20, 40]
DD_VALUES = [1, 2, 4, 6]
MODELS = ['mini', 'gpt54', 'gpt55']
MODEL_LABEL = {'mini': 'gpt-5.4-mini', 'gpt54': 'gpt-5.4', 'gpt55': 'gpt-5.5'}
HEATMAP_METRICS = [('success_rate', 'Success rate'), ('retention_at_5', 'Retention@5'), ('belief_precision', 'Belief precision'), ('exploration_coverage', 'Exploration coverage')]

def wilson(k: int, n: int, z: float=1.96) -> tuple[float, float, float]:
    if n == 0:
        return (float('nan'), float('nan'), float('nan'))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))

def mean_or_nan(xs):
    xs = [x for x in xs if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return statistics.mean(xs) if xs else float('nan')

def fmt(x, pct=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '  -  '
    if pct:
        return f'{100 * x:4.0f}%'
    return f'{x:.2f}'

def load_episodes(run_dir: Path) -> list[dict]:
    path = run_dir / 'derived/metrics_v1/all_episode_metrics.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    return data['episodes']

def aggregate(episodes: list[dict]) -> dict:
    cells = defaultdict(list)
    for e in episodes:
        cells[e['model_key'], e['state_card'], e['dep_density']].append(e)
    agg = {}
    for (mk, sc, dd), eps in cells.items():
        n = len(eps)
        k_success = sum((1 for e in eps if e['success']))
        p, lo, hi = wilson(k_success, n)
        agg[mk, sc, dd] = {'n_seeds': n, 'success_rate': p, 'success_ci_lo': lo, 'success_ci_hi': hi, 'belief_precision': mean_or_nan([e.get('mean_belief_precision') for e in eps]), 'update_recall': mean_or_nan([e.get('mean_new_observation_update_recall') for e in eps]), 'retention_at_1': mean_or_nan([e.get('mean_retention_at_1') for e in eps]), 'retention_at_5': mean_or_nan([e.get('mean_retention_at_5') for e in eps]), 'retention_at_10': mean_or_nan([e.get('mean_retention_at_10') for e in eps]), 'exploration_coverage': mean_or_nan([e.get('mean_exploration_coverage') for e in eps]), 'contradiction_rate': mean_or_nan([e.get('mean_contradiction_rate') for e in eps]), 'task_predicate_accuracy': mean_or_nan([e.get('mean_task_predicate_accuracy') for e in eps]), 'legacy_mean_wsa': mean_or_nan([e.get('legacy_mean_wsa') for e in eps]), 'mean_steps': mean_or_nan([e.get('steps') for e in eps])}
    return agg

def heatmap_md(agg: dict, mk: str, metric: str, pct: bool) -> str:
    lines = []
    header = '| sc \\ dd | ' + ' | '.join((f'dd={dd}' for dd in DD_VALUES)) + ' |'
    sep = '|' + '---|' * (len(DD_VALUES) + 1)
    lines.append(header)
    lines.append(sep)
    for sc in SC_VALUES:
        row = [f'**sc={sc}**']
        for dd in DD_VALUES:
            cell = agg.get((mk, sc, dd))
            if cell is None:
                row.append(' · ')
            else:
                row.append(fmt(cell.get(metric), pct=pct))
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)

def main() -> None:
    run_dir = Path(sys.argv[1]).resolve()
    episodes = load_episodes(run_dir)
    agg = aggregate(episodes)
    global SC_VALUES, DD_VALUES
    scs = sorted({k[1] for k in agg})
    dds = sorted({k[2] for k in agg})
    if scs:
        SC_VALUES = scs
    if dds:
        DD_VALUES = dds
    agg_out = {f'{mk}|sc{sc}|dd{dd}': v for (mk, sc, dd), v in sorted(agg.items())}
    (run_dir / 'derived/aggregate_by_cell.json').write_text(json.dumps(agg_out, indent=2, sort_keys=True), encoding='utf-8')
    models_present = [m for m in MODELS if any((k[0] == m for k in agg))]
    out = []
    out.append('# TRACE_SWEEP aggregate tables\n')
    out.append(f'Run dir: `{run_dir}`\n')
    out.append(f'Episodes aggregated: {len(episodes)}\n')
    out.append(f"Models present: {', '.join((MODEL_LABEL[m] for m in models_present))}\n")
    for mk in models_present:
        out.append(f'\n## {MODEL_LABEL[mk]}\n')
        n_cells = sum((1 for k in agg if k[0] == mk))
        total_ep = sum((agg[k]['n_seeds'] for k in agg if k[0] == mk))
        out.append(f'cells: {n_cells}/12, episodes: {total_ep}\n')
        for metric, label in HEATMAP_METRICS:
            pct = metric in ('success_rate', 'retention_at_5', 'exploration_coverage')
            out.append(f'\n### {label} ({MODEL_LABEL[mk]})\n')
            out.append(heatmap_md(agg, mk, metric, pct))
            out.append('')
    out.append('\n## Cross-model headline: retention_at_5 (mean over all cells)\n')
    out.append('| model | mean retention@5 | mean success | mean belief_prec | mean coverage |')
    out.append('|---|---|---|---|---|')
    for mk in models_present:
        cells = [v for k, v in agg.items() if k[0] == mk]
        r5 = mean_or_nan([c['retention_at_5'] for c in cells])
        sr = mean_or_nan([c['success_rate'] for c in cells])
        bp = mean_or_nan([c['belief_precision'] for c in cells])
        cov = mean_or_nan([c['exploration_coverage'] for c in cells])
        out.append(f'| {MODEL_LABEL[mk]} | {fmt(r5)} | {fmt(sr, pct=True)} | {fmt(bp)} | {fmt(cov, pct=True)} |')
    report_path = run_dir / 'aggregate_tables.md'
    report_path.write_text('\n'.join(out) + '\n', encoding='utf-8')
    print(f'wrote {report_path}')
    print(f"wrote {run_dir / 'derived/aggregate_by_cell.json'}")
    for mk in models_present:
        cells = [v for k, v in agg.items() if k[0] == mk]
        r5 = mean_or_nan([c['retention_at_5'] for c in cells])
        print(f'{MODEL_LABEL[mk]}: mean retention@5 = {fmt(r5)}  (cells={len(cells)})')
if __name__ == '__main__':
    main()
