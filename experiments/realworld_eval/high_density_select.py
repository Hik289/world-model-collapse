from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.abspath(os.path.join(_HERE, '..', '..'))
for p in (_PROJ, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import alfworld_common as AC
import alfworld_gold_state as G
MULTI_OBJECT_TYPES = {'clean', 'cool', 'heat', 'pick_two'}

def true_density(game_file: str) -> dict:
    env = AC.make_single_env(game_file, max_steps=10)
    try:
        _obs, infos = env.reset()
        facts = infos['facts'][0]
        sc = G.compute_sc(facts)
        return sc
    finally:
        try:
            env.close()
        except Exception:
            pass

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--survey-json', default=os.path.join(_HERE, 'sc_survey.json'))
    ap.add_argument('--min-recep', type=int, default=40)
    ap.add_argument('--min-obj', type=int, default=50)
    ap.add_argument('--top', type=int, default=12)
    ap.add_argument('--out', default=os.path.join(_HERE, 'high_density_tasks.json'))
    args = ap.parse_args()
    survey = json.load(open(args.survey_json))
    cand = [g for g in survey if g['n_receptacles'] >= args.min_recep and g['n_objects'] >= args.min_obj]
    print(f'first-pass (n_recep>={args.min_recep} & n_obj>={args.min_obj}): {len(cand)} / {len(survey)}', flush=True)
    enriched = []
    for g in cand:
        sc = true_density(g['game_file'])
        n_in = sc['n_in_receptacle']
        multi = g['task_type'] in MULTI_OBJECT_TYPES
        score = n_in * (1.25 if multi else 1.0)
        row = {**g, 'n_in_receptacle': n_in, 'sc_facts_recompute': sc['sc_facts'], 'multi_object_type': multi, 'density_score': score}
        enriched.append(row)
        print(f"  {g['task_type']:16s} n_recep={g['n_receptacles']:3d} n_obj={g['n_objects']:3d} n_in_recep={n_in:3d} score={score:.1f} {g['game_file'].split('/json_2.1.1/')[1][:50]}", flush=True)
    enriched.sort(key=lambda r: -r['density_score'])
    out = {'criteria': {'min_receptacles': args.min_recep, 'min_objects': args.min_obj, 'multi_object_types': sorted(MULTI_OBJECT_TYPES), 'ranking': 'density_score = n_in_receptacle * (1.25 if multi_object else 1.0)', 'rationale': 'n_in_receptacle is the number of loc_contains facts = the ceiling on simultaneous container-contents the agent can be forced to hold. Native retention was flat 1.0 because the agent only opened 3-10 non-empty containers; ranking by n_in_receptacle + forced-survey exposes the whole ceiling.'}, 'n_candidates': len(enriched), 'tasks': enriched}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f'\nTOP {args.top} by density_score:', flush=True)
    for r in enriched[:args.top]:
        print(f"  score={r['density_score']:6.1f} n_in_recep={r['n_in_receptacle']:3d} {r['task_type']:16s} {r['game_file'].split('/json_2.1.1/')[1][:55]}", flush=True)
    print(f'\nwrote {args.out}', flush=True)
if __name__ == '__main__':
    main()
