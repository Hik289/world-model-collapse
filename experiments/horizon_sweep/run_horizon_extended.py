from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.agents.llm_client import ConfiguredLLMClient
from src.runner.pilot_runner import CellSpec, run_pilot_slice
from src.runner.cost_tracker import CostTracker
from src.evaluation.jsonl_logger import JSONLWriter
HERE = Path(__file__).resolve().parent
LOG_DIR = ROOT / 'data' / 'raw_logs'
COST_DIR = ROOT / 'data' / 'costs'
N_EP = 15
SPECS = {'gpt-5.4-mini': [dict(tag='ext_T', env='graph_nav', sc=20, dd=1, T_vals=[150, 200, 300, 500]), dict(tag='ext_T', env='graph_nav', sc=20, dd=2, T_vals=[150, 200, 300, 500]), dict(tag='ext_T', env='tool_dag', sc=5, dd=3, T_vals=[80, 150, 200, 300]), dict(tag='TxSC', env='graph_nav', sc=10, dd=1, T_vals=[20, 40, 80, 150]), dict(tag='TxSC', env='graph_nav', sc=15, dd=1, T_vals=[20, 40, 80, 150]), dict(tag='TxSC', env='graph_nav', sc=20, dd=1, T_vals=[20, 40, 80, 150]), dict(tag='TxSC', env='graph_nav', sc=25, dd=1, T_vals=[20, 40, 80, 150]), dict(tag='TxDD', env='graph_nav', sc=20, dd=1, T_vals=[20, 40, 80, 150]), dict(tag='TxDD', env='graph_nav', sc=20, dd=2, T_vals=[20, 40, 80, 150]), dict(tag='TxDD', env='graph_nav', sc=20, dd=3, T_vals=[20, 40, 80, 150]), dict(tag='TxDD', env='graph_nav', sc=20, dd=4, T_vals=[20, 40, 80, 150])], 'claude-haiku-4-5': [dict(tag='ext_T', env='graph_nav', sc=10, dd=1, T_vals=[150, 200, 300, 500]), dict(tag='ext_T', env='graph_nav', sc=8, dd=1, T_vals=[80, 150, 200, 300]), dict(tag='TxSC', env='graph_nav', sc=5, dd=1, T_vals=[20, 40, 80, 150]), dict(tag='TxSC', env='graph_nav', sc=8, dd=1, T_vals=[20, 40, 80, 150]), dict(tag='TxSC', env='graph_nav', sc=10, dd=1, T_vals=[20, 40, 80, 150]), dict(tag='TxSC', env='graph_nav', sc=13, dd=1, T_vals=[20, 40, 80, 150])]}

def cell_key(tag, env, sc, dd, model, T):
    return f'{tag}|{env}|sc{sc}|dd{dd}|T{T}|{model}'

def build_cells(env, sc, dd, model, T, n_ep):
    cells = []
    for i in range(n_ep):
        seed = hash(f'ext:{env}:{sc}:{dd}:{model}:{T}:{i}') & 2147483647
        stress = {'state_card': sc, 'dep_density': dd, 'T': T, 'branching': 4, 'obs_noise': 'clean', 'mut_rate': 'static'}
        tid = f'ext_{env}_sc{sc}_dd{dd}_T{T}_{model}_ep{i:03d}'
        cells.append(CellSpec(env_name=env, model=model, stress_config=stress, task_config={'task_id': tid, 'stress_config': stress}, task_seed=seed, decoding_seed=seed ^ 61453, world_regime='horizon_ext', task_id=tid))
    return cells

def load_done(path):
    if path.exists():
        try:
            return set(json.loads(path.read_text()))
        except:
            pass
    return set()

def save_done(path, done):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(done), indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    models = [args.model] if args.model else list(SPECS)
    client = ConfiguredLLMClient(timeout=300.0)
    for model in models:
        done_path = HERE / f"ext_done_{model.replace('-', '_')}.json"
        ep_path = LOG_DIR / f"horizon_ext_{model.replace('-', '_')}_episode.jsonl"
        step_path = LOG_DIR / f"horizon_ext_{model.replace('-', '_')}_step.jsonl"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        done = load_done(done_path)
        specs = SPECS[model]
        total = sum((len(s['T_vals']) for s in specs))
        skip = sum((1 for s in specs for T in s['T_vals'] if cell_key(s['tag'], s['env'], s['sc'], s['dd'], model, T) in done))
        print(f'\n=== {model} === {total} cells ({skip} done)')
        with JSONLWriter(ep_path) as ep_w, JSONLWriter(step_path) as st_w:
            for spec in specs:
                env, sc, dd = (spec['env'], spec['sc'], spec['dd'])
                tag = spec['tag']
                for T in spec['T_vals']:
                    key = cell_key(tag, env, sc, dd, model, T)
                    if key in done:
                        print(f'  SKIP {key}')
                        continue
                    print(f'  [{tag}] {env} sc={sc} dd={dd} T={T:<4} (n={N_EP})')
                    cells = build_cells(env, sc, dd, model, T, N_EP)
                    ct = CostTracker(out_path=COST_DIR / 'horizon_ext_cost.jsonl', phase='horizon_ext', slice_name=key, emit_every=5)
                    outcomes = run_pilot_slice(cells=cells, client=client, step_jsonl_path=step_path, episode_jsonl_path=ep_path, cost_tracker=ct, n_workers=args.workers)
                    succ = sum((1 for o in outcomes if o.success))
                    p = succ / max(len(outcomes), 1)
                    done.add(key)
                    save_done(done_path, done)
                    mk = '***CLIFF' if p < 0.1 else '↓↓' if p < 0.3 else '↓' if p < 0.5 else ''
                    print(f'     p̂={p:.3f} ({succ}/{len(outcomes)}) {mk}')
                    _print_partial(ep_path, model, env, sc, dd, tag)
    print('\n✅ Done')

def _print_partial(ep_path, model, env, sc_fix, dd_fix, tag):
    import re
    data = {}
    try:
        for line in open(ep_path):
            d = json.loads(line)
            m = re.match('ext_(.+)_sc(\\d+)_dd(\\d+)_T(\\d+)_.+_ep\\d+', d.get('task_id', ''))
            if m and m.group(1) == env and (int(m.group(2)) == sc_fix) and (int(m.group(3)) == dd_fix):
                T = int(m.group(4))
                data.setdefault(T, []).append(d.get('final_success', False))
    except:
        return
    if not data:
        return
    row = f'    → {env} sc={sc_fix} dd={dd_fix}: '
    for T in sorted(data):
        v = data[T]
        if len(v) == N_EP:
            p = sum(v) / len(v)
            row += f'T{T}={p:.2f} '
    print(row)
if __name__ == '__main__':
    main()
