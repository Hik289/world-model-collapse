from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _PROJ)
sys.path.insert(0, _HERE)
import alfworld_common as AC
import alfworld_agent_runner as R
T_GRID = [8, 12, 16, 20, 25, 30, 40]
N_TASKS = 6

def pick_tasks(n: int) -> list[str]:
    seen = AC.list_games('valid_seen')
    wants = ['pick_and_place_simple', 'pick_clean_then_place_in_recep', 'pick_cool_then_place_in_recep', 'pick_heat_then_place_in_recep', 'pick_two_obj_and_place', 'look_at_obj_in_light']
    picks = []
    for w in wants:
        for g in seen:
            if w in g and g not in picks:
                picks.append(g)
                break
    return picks[:n]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--model-key', default='haiku')
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / 'checkpoint.json'
    done = set()
    if ckpt.exists():
        done = set(json.loads(ckpt.read_text()).get('done', []))
    tasks = pick_tasks(N_TASKS)
    jobs = []
    for T in T_GRID:
        for gi, game in enumerate(tasks):
            eid = f'{args.model_key}_T{T:02d}_g{gi}'
            jobs.append((eid, game, T))
    print(f'TRANSITION_SCAN plan: {len(jobs)} episodes ({len(T_GRID)} T-levels x {len(tasks)} tasks), model={args.model_key}', flush=True)
    print(f'  T grid: {T_GRID}', flush=True)
    for i, (eid, game, T) in enumerate(jobs):
        if eid in done:
            print(f'[{i + 1}/{len(jobs)}] SKIP {eid}', flush=True)
            continue
        print(f'[{i + 1}/{len(jobs)}] RUN {eid} T={T}', flush=True)
        try:
            R.run_episode(game, str(run_dir), model_key=args.model_key, max_steps=T, episode_id=eid)
            done.add(eid)
            ckpt.write_text(json.dumps({'done': sorted(done)}, indent=2))
        except Exception as exc:
            print(f'[{i + 1}/{len(jobs)}] FAIL {eid}: {exc!r}', flush=True)
    print(f'TRANSITION_SCAN done: {len(done)}/{len(jobs)}', flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
