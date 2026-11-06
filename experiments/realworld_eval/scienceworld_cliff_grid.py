from __future__ import annotations
import argparse
import json
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0, '.')
sys.path.insert(0, 'experiments/realworld_eval')
import scienceworld_agent_runner as R
T_GRID = [10, 15, 20, 25, 30, 35, 40, 45, 50]
TASKS = ['boil', 'freeze', 'melt', 'change-the-state-of-matter-of', 'chemistry-mix']

def log(run_dir: Path, msg: str) -> None:
    ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with (run_dir / 'grid.log').open('a', encoding='utf-8') as f:
        f.write(line + '\n')

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--model-key', default='haiku')
    ap.add_argument('--block-focus', action='store_true', default=True, help='drop terminating `focus` meta-action so episodes run the full horizon (retention is measured during operation, not task completion which haiku never reaches on these tasks)')
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_root = run_dir / 'raw'
    cells = []
    for task in TASKS:
        for T in T_GRID:
            for seed in range(args.seeds):
                cells.append((task, T, seed))
    total = len(cells)
    log(run_dir, f'GRID_START total={total} T={T_GRID} tasks={TASKS} seeds={args.seeds} model={args.model_key}')
    done = 0
    for i, (task, T, seed) in enumerate(cells):
        eid = f'{args.model_key}_{task}_T{T:02d}_s{seed}'
        ep_dir = raw_root / eid
        summ = ep_dir / 'summary.json'
        if summ.exists() and summ.stat().st_size > 0:
            done += 1
            log(run_dir, f'[{i + 1}/{total}] SKIP {eid} (done)')
            continue
        log(run_dir, f'[{i + 1}/{total}] RUN {eid} task={task} T={T} seed={seed}')
        for attempt in range(3):
            try:
                res = R.run_episode(task, run_dir, variation=seed, model_key=args.model_key, max_steps=T, episode_id=eid, block_focus=args.block_focus)
                done += 1
                log(run_dir, f"[{i + 1}/{total}] DONE {eid} success={res.get('success')} steps={res.get('steps')}")
                break
            except Exception as e:
                log(run_dir, f'[{i + 1}/{total}] ERR {eid} attempt={attempt} {type(e).__name__}: {e}')
                (run_dir / 'grid_errors.log').open('a').write(f'{eid} attempt={attempt}\n{traceback.format_exc()}\n')
                time.sleep(15 * (attempt + 1))
        (run_dir / 'grid_checkpoint.json').write_text(json.dumps({'done': done, 'total': total, 'last': eid, 'idx': i + 1}, indent=2))
    log(run_dir, f'GRID_COMPLETE done={done}/{total}')
if __name__ == '__main__':
    main()
