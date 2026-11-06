from __future__ import annotations
import argparse, json, sys, time, traceback
from pathlib import Path
sys.path.insert(0, '.')
sys.path.insert(0, 'experiments/realworld_eval')
import scienceworld_agent_runner as R
T_GRID = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 44, 48, 52, 56]
TASKS = ['boil', 'freeze', 'melt', 'change-the-state-of-matter-of', 'chemistry-mix', 'use-thermometer', 'measure-melting-point-known-substance', 'measure-melting-point-unknown-substance', 'power-component', 'power-component-renewable-vs-nonrenewable-energy', 'test-conductivity', 'test-conductivity-of-unknown-substances', 'grow-plant', 'grow-fruit', 'chemistry-mix-paint-secondary-color']

def log(run_dir: Path, msg: str) -> None:
    ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with (run_dir / 'grid.log').open('a', encoding='utf-8') as f:
        f.write(line + '\n')

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--model-key', default='haiku')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_root = run_dir / 'raw'
    cells = [(task, Th) for task in TASKS for Th in T_GRID]
    total = len(cells)
    log(run_dir, f'GRID_START total={total} T={T_GRID} tasks={TASKS} seed={args.seed} model={args.model_key}')
    done = 0
    for i, (task, Th) in enumerate(cells):
        if args.nshards > 1 and i % args.nshards != args.shard:
            continue
        eid = f'{args.model_key}_{task}_T{Th:02d}_s{args.seed}'
        summ = raw_root / eid / 'summary.json'
        if summ.exists() and summ.stat().st_size > 0:
            done += 1
            continue
        log(run_dir, f'[{i + 1}/{total}] RUN {eid}')
        for attempt in range(3):
            try:
                res = R.run_episode(task, run_dir, variation=args.seed, model_key=args.model_key, max_steps=Th, episode_id=eid, block_focus=True)
                done += 1
                log(run_dir, f"[{i + 1}/{total}] DONE {eid} steps={res.get('steps')} stuck={res.get('stuck_in_loop')}")
                break
            except Exception as e:
                log(run_dir, f'[{i + 1}/{total}] ERR {eid} attempt={attempt} {type(e).__name__}: {e}')
                (run_dir / 'grid_errors.log').open('a').write(f'{eid} attempt={attempt}\n{traceback.format_exc()}\n')
                time.sleep(15 * (attempt + 1))
        (run_dir / 'grid_checkpoint.json').write_text(json.dumps({'done': done, 'total': total, 'last': eid, 'idx': i + 1}, indent=2))
    log(run_dir, f'GRID_COMPLETE done={done}/{total}')
if __name__ == '__main__':
    main()
