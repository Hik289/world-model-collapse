from __future__ import annotations
import json
import os
import sys
import traceback
from pathlib import Path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _PROJ)
sys.path.insert(0, _HERE)
import alfworld_common as AC
import alfworld_gold_state as G
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(_HERE) / 'sc_survey.json'

def main():
    done = {}
    if OUT.exists():
        for row in json.loads(OUT.read_text()):
            done[row['game_file']] = row
    rows = list(done.values())
    for split in ('valid_seen', 'valid_unseen'):
        games = AC.list_games(split)
        print(f'{split}: {len(games)} games', flush=True)
        for i, g in enumerate(games):
            if g in done:
                continue
            try:
                env = AC.make_single_env(g, max_steps=10)
                _obs, infos = env.reset()
                facts = infos['facts'][0]
                sc = G.compute_sc(facts)
                row = {'game_file': g, 'split': split, 'task_type': G.task_type_from_gamefile(g), 'sc_facts': sc['sc_facts'], 'n_objects': sc['n_objects'], 'n_receptacles': sc['n_receptacles']}
                rows.append(row)
                env.close()
            except Exception as exc:
                print(f'  SKIP {i} {g[-40:]}: {exc!r}', flush=True)
                traceback.print_exc()
                continue
            if i % 25 == 0:
                OUT.write_text(json.dumps(rows, indent=2))
                print(f"  {split} {i}/{len(games)} sc={row['sc_facts']} {row['task_type']}", flush=True)
    OUT.write_text(json.dumps(rows, indent=2))
    scs = sorted((r['sc_facts'] for r in rows))
    print(f'TOTAL {len(rows)} games. SC min={scs[0]} max={scs[-1]} median={scs[len(scs) // 2]}', flush=True)
if __name__ == '__main__':
    main()
