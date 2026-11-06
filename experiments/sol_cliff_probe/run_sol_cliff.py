from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
import time
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
COST_DIR.mkdir(parents=True, exist_ok=True)
ENV = 'stateful_puzzle'
MIN_FREE_GB = 15.0

def free_gb(path: str='/') -> float:
    return shutil.disk_usage(path).free / 1000000000.0

def mslug(model: str) -> str:
    return model.replace('-', '_').replace('.', '_')

def cell_key(model: str, sc: int, dd: int, T: int) -> str:
    return f'{ENV}|{model}|sc{sc}|dd{dd}|T{T}'

def load_done(path: Path) -> set[str]:
    if path.exists():
        try:
            return set(json.loads(path.read_text()))
        except Exception:
            pass
    return set()

def save_done(path: Path, done: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(sorted(done), indent=2))
    tmp.replace(path)

def _phat_from_log(ep_path: Path, model: str, sc: int, dd: int, T: int) -> float | None:
    if not ep_path.exists():
        return None
    succ = tot = 0
    prefix = f'solcliff_{ENV}_sc{sc}_dd{dd}_T{T}_{mslug(model)}_ep'
    try:
        with ep_path.open(encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if str(r.get('task_id', '')).startswith(prefix):
                    tot += 1
                    if r.get('final_success'):
                        succ += 1
    except Exception:
        return None
    return succ / tot if tot else None

def build_cells(model: str, sc: int, dd: int, T: int, n_ep: int) -> list[CellSpec]:
    cells = []
    for i in range(n_ep):
        seed = int(hashlib.sha256(f'solcliff:{model}:{sc}:{dd}:{T}:{i}'.encode()).hexdigest()[:8], 16)
        stress = {'state_card': sc, 'dep_density': dd, 'T': T, 'branching': 4, 'obs_noise': 'clean', 'mut_rate': 'static'}
        tid = f'solcliff_{ENV}_sc{sc}_dd{dd}_T{T}_{mslug(model)}_ep{i:03d}'
        cells.append(CellSpec(env_name=ENV, model=model, stress_config=stress, task_config={'task_id': tid, 'stress_config': stress}, task_seed=seed, decoding_seed=seed ^ 48879, world_regime='sol_cliff_probe', task_id=tid))
    return cells

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--axis', choices=['sc', 'T'], default='sc', help='which axis is being swept (for logging/checkpoint naming)')
    ap.add_argument('--sc', default='40,80,160,320', help='comma list of state_card values (sweep if axis=sc, else single)')
    ap.add_argument('--dd', type=int, default=4)
    ap.add_argument('--T', default='40', help='comma list of T values (sweep if axis=T, else single)')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--timeout', type=float, default=900.0)
    ap.add_argument('--auto-escalate', action='store_true', help='keep climbing the SC ladder (then T ladder) until a cliff appears or ceiling is hit. For the sol target run.')
    ap.add_argument('--flat-thresh', type=float, default=0.6, help="success p̂ >= this == 'flat' (no cliff) → keep escalating")
    args = ap.parse_args()
    model = args.model
    sc_vals = [int(x) for x in str(args.sc).split(',') if x.strip()]
    T_vals = [int(x) for x in str(args.T).split(',') if x.strip()]
    dd = args.dd
    SC_LADDER = [40, 80, 160, 320, 640, 1280]
    T_LADDER = [40, 80, 120, 160, 240, 320]
    T_FIXED_SC_FOR_STAGE2 = 160
    tag = f'{mslug(model)}_{args.axis}'
    done_path = HERE / f'done_{tag}.json'
    ep_path = LOG_DIR / f'sol_cliff_{tag}_episode.jsonl'
    st_path = LOG_DIR / f'sol_cliff_{tag}_step.jsonl'
    done = load_done(done_path)
    if args.auto_escalate:
        queue = [(sc, T_vals[0]) for sc in sc_vals]
        mode = 'sc'
    else:
        queue = [(sc, T) for sc in sc_vals for T in T_vals]
        mode = args.axis
    print(f'=== sol_cliff {model} axis={args.axis} auto={args.auto_escalate} ===')
    print(f'    initial queue={queue} dd={dd} | n={args.n} workers={args.workers}')
    print(f'    free disk: {free_gb():.0f} GB')

    def run_one_cell(ep_w, st_w, sc, T) -> float | None:
        key = cell_key(model, sc, dd, T)
        if key in done:
            print(f'  SKIP {key}', flush=True)
            return _phat_from_log(ep_path, model, sc, dd, T)
        fg = free_gb()
        if fg < MIN_FREE_GB:
            print(f'  [DISK-GUARD] free {fg:.1f}GB < {MIN_FREE_GB}GB — STOP before {key}.', flush=True)
            return None
        print(f'  {ENV} sc={sc:<5} dd={dd} T={T:<4} (n={args.n}) free={fg:.0f}GB', flush=True)
        t0 = time.time()
        cells = build_cells(model, sc, dd, T, args.n)
        ct = CostTracker(out_path=COST_DIR / 'sol_cliff_cost.jsonl', phase='sol_cliff', slice_name=key, emit_every=5)
        client = ConfiguredLLMClient(timeout=args.timeout)
        outcomes = run_pilot_slice(cells=cells, client=client, step_jsonl_path=st_path, episode_jsonl_path=ep_path, cost_tracker=ct, n_workers=args.workers)
        if any((o.error for o in outcomes)):
            raise RuntimeError('Cell incomplete; transport failures are excluded from success rates')
        succ = sum((1 for o in outcomes if o.success))
        p = succ / max(len(outcomes), 1)
        done.add(key)
        save_done(done_path, done)
        dt = time.time() - t0
        mk = '***CLIFF' if p < 0.1 else '↓↓' if p < 0.3 else '↓' if p < 0.6 else 'flat'
        print(f'    → success p̂={p:.2f} ({succ}/{len(outcomes)}) {mk}  [{dt / 60:.1f}min]', flush=True)
        if 'haiku' in model:
            time.sleep(5)
        return p
    n_initial = len(queue)
    with JSONLWriter(ep_path) as ep_w, JSONLWriter(st_path) as st_w:
        if not args.auto_escalate:
            for sc, T in queue:
                if run_one_cell(ep_w, st_w, sc, T) is None:
                    return 2
        else:
            qi = 0
            while qi < len(queue):
                sc, T = queue[qi]
                is_dynamic = qi >= n_initial
                qi += 1
                p = run_one_cell(ep_w, st_w, sc, T)
                if p is None:
                    return 2
                if is_dynamic and p < args.flat_thresh:
                    print(f'  [ESCALATE] cliff at sc={sc} T={T} (p̂={p:.2f} < {args.flat_thresh}). Stop escalation; critical point on the {mode} axis.', flush=True)
                    break
                if qi >= len(queue):
                    if p < args.flat_thresh:
                        print(f'  [ESCALATE] last tier sc={sc} T={T} cliffed (p̂={p:.2f}); escalation complete.', flush=True)
                        break
                    if mode == 'sc':
                        higher = [s for s in SC_LADDER if s > sc]
                        if higher:
                            print(f'  [ESCALATE] flat at sc={sc}; climbing SC → sc={higher[0]}', flush=True)
                            queue.append((higher[0], T))
                        else:
                            print(f'  [ESCALATE] SC ceiling {sc} still flat; switching to T axis at fixed sc={T_FIXED_SC_FOR_STAGE2}', flush=True)
                            mode = 'T'
                            for tt in [t for t in T_LADDER if t > T_vals[0]]:
                                queue.append((T_FIXED_SC_FOR_STAGE2, tt))
                    elif mode == 'T':
                        print(f'  [ESCALATE] T axis exhausted at T={T}, sol still flat. Reached ceiling of both axes.', flush=True)
    print(f'\n✅ Done {model} axis={args.axis}. Logs: {ep_path.name}, {st_path.name}')
    print(f'   Next: python3 -m experiments.sol_cliff_probe.compute_retention --step-log {st_path} --episode-log {ep_path} --out {HERE}/{tag}_retention.json')
    return 0
if __name__ == '__main__':
    sys.exit(main())
