from __future__ import annotations
import argparse, json, sys, time
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
LOG_DIR.mkdir(parents=True, exist_ok=True)
N_EP = 15
SWEEP = {'gpt-5.4-mini': {'envs': ['graph_nav'], 'sc_vals': [30, 40, 60, 80, 100, 150], 'dd_vals': [1, 2, 3, 4], 'workers': 4}, 'claude-haiku-4-5': {'envs': ['graph_nav'], 'sc_vals': [5, 10, 20, 40, 60], 'dd_vals': [1, 2, 3, 4], 'workers': 1}}

def cell_key(env, sc, dd, model):
    return f'{env}|sc{sc}|dd{dd}|{model}'

def build_cells(env, sc, dd, model):
    cells = []
    for i in range(N_EP):
        seed = hash(f'scext:{env}:{sc}:{dd}:{model}:{i}') & 2147483647
        stress = {'state_card': sc, 'dep_density': dd, 'T': 40, 'branching': 4, 'obs_noise': 'clean', 'mut_rate': 'static'}
        tid = f'scext_{env}_sc{sc}_dd{dd}_{model}_ep{i:03d}'
        cells.append(CellSpec(env_name=env, model=model, stress_config=stress, task_config={'task_id': tid, 'stress_config': stress}, task_seed=seed, decoding_seed=seed ^ 48879, world_regime='sc_extended', task_id=tid))
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
    ap.add_argument('--model', required=True, choices=list(SWEEP))
    args = ap.parse_args()
    model = args.model
    cfg = SWEEP[model]
    done_path = HERE / f"scext_done_{model.replace('-', '_').replace('.', '_')}.json"
    ep_path = LOG_DIR / f"scext_{model.replace('-', '_').replace('.', '_')}_episode.jsonl"
    st_path = LOG_DIR / f"scext_{model.replace('-', '_').replace('.', '_')}_step.jsonl"
    done = load_done(done_path)
    total = len(cfg['envs']) * len(cfg['sc_vals']) * len(cfg['dd_vals'])
    skip = sum((1 for env in cfg['envs'] for sc in cfg['sc_vals'] for dd in cfg['dd_vals'] if cell_key(env, sc, dd, model) in done))
    print(f'=== {model} === {total} cells ({skip} done, {total - skip} remaining)')
    client = ConfiguredLLMClient(timeout=600.0)
    with JSONLWriter(ep_path) as ep_w, JSONLWriter(st_path) as st_w:
        for env in cfg['envs']:
            for sc in cfg['sc_vals']:
                for dd in cfg['dd_vals']:
                    key = cell_key(env, sc, dd, model)
                    if key in done:
                        print(f'  SKIP {key}')
                        continue
                    print(f'  {env} sc={sc:<4} dd={dd}  (n={N_EP})', flush=True)
                    cells = build_cells(env, sc, dd, model)
                    ct = CostTracker(out_path=COST_DIR / 'scext_cost.jsonl', phase='sc_extended', slice_name=key, emit_every=5)
                    outcomes = run_pilot_slice(cells=cells, client=client, step_jsonl_path=st_path, episode_jsonl_path=ep_path, cost_tracker=ct, n_workers=cfg['workers'])
                    succ = sum((1 for o in outcomes if o.success))
                    p = succ / max(len(outcomes), 1)
                    done.add(key)
                    save_done(done_path, done)
                    mk = '***CLIFF' if p < 0.1 else '↓↓' if p < 0.3 else '↓' if p < 0.6 else ''
                    print(f'    → p̂={p:.2f} ({succ}/{len(outcomes)}) {mk}', flush=True)
                    if 'haiku' in model:
                        time.sleep(5)
    print('\n=== FINAL TABLE ===')
    _print_table(ep_path, model, cfg['sc_vals'], cfg['dd_vals'], cfg['envs'][0])
    print('✅ Done')

def _print_table(ep_path, model, sc_vals, dd_vals, env):
    import re
    data = {}
    try:
        for line in open(ep_path):
            d = json.loads(line)
            m = re.match(f'scext_{re.escape(env)}_sc(\\d+)_dd(\\d+)_.+_ep\\d+', d.get('task_id', ''))
            if m:
                sc, dd = (int(m.group(1)), int(m.group(2)))
                data.setdefault((sc, dd), []).append(d.get('final_success', False))
    except:
        pass
    print(f'  sc\\dd ' + '  '.join((f'dd={dd}' for dd in dd_vals)))
    for sc in sc_vals:
        row = f'  sc={sc:<5}'
        for dd in dd_vals:
            v = data.get((sc, dd))
            if v and len(v) >= N_EP:
                p = sum(v) / len(v)
                mk = '***' if p < 0.1 else '↓↓' if p < 0.3 else '↓' if p < 0.6 else '  '
                row += f'  {p:.2f}{mk}'
            elif v:
                row += f'  ({len(v)})'
            else:
                row += '    — '
        print(row)
if __name__ == '__main__':
    main()
