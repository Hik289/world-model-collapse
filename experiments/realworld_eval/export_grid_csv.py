import json, glob, os, sys, csv
sys.path.insert(0, '.')
sys.path.insert(0, 'experiments/trace_evaluation')
import trace_v1_metrics as T
from src.evaluation.runner import world_state_facts
KS = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
OUT = 'experiments/realworld_eval/GRID_EXPORT.csv'

def episode_row(ep_dir, model):
    mani = json.load(open(ep_dir + 'manifest.json'))
    summ = json.load(open(ep_dir + 'summary.json'))
    subgoal_ids = set(mani['public_task_spec'])
    ledger = T.ObservedLedger()
    steps = [json.loads(l) for l in open(ep_dir + 'steps.jsonl') if l.strip()]
    ret = {k: [] for k in KS}
    bp, forg, stale, hall = ([], [], [], [])
    for rec in steps:
        t = rec['step']
        agent = T.filter_belief_facts(world_state_facts(rec['updater']['canonical_state']), subgoal_ids)
        gold = T.filter_belief_facts(world_state_facts(rec['evaluator_only']['gold_before']), subgoal_ids)
        obs = rec['model_visible']['observation_partial_state']
        ledger.update(obs, t, subgoal_ids)
        ledger_all = {f for f in ledger.facts if not ledger.is_layout(f)}
        ledger_true = ledger_all & gold
        correct = agent & gold
        wrong = agent - gold
        st = wrong & ledger_all
        hl = wrong - ledger_all
        forgot = ledger_true - agent
        bp.append(len(correct) / len(agent) if agent else 1.0)
        forg.append(len(forgot) / len(ledger_true) if ledger_true else 0.0)
        stale.append(len(st) / len(agent) if agent else 0.0)
        hall.append(len(hl) / len(agent) if agent else 0.0)
        for k in KS:
            elig = {f for f in ledger_true if t - ledger.last_seen.get(f, t) >= k and (not ledger.is_layout(f))}
            if elig:
                ret[k].append(len(agent & elig) / len(elig))

    def m(x):
        return round(sum(x) / len(x), 4) if x else ''
    row = {'model': model, 'task': mani.get('task_name'), 'horizon_T': mani['horizon'], 'state_card': mani['state_card'], 'n_rooms': mani.get('sc_components', {}).get('n_rooms'), 'steps': summ['steps'], 'success': int(summ['success']), 'stuck_in_loop': int(summ.get('stuck_in_loop', 0)), 'belief_precision': m(bp), 'forgotten_rate': m(forg), 'stale_rate': m(stale), 'hallucination_rate': m(hall)}
    for k in KS:
        row[f'retention_at_{k}'] = m(ret[k])
    return row

def main():
    rows = []
    for model, rel in [('haiku', 'sw_grid_dense_haiku'), ('sonnet', 'sw_grid_dense_sonnet'), ('sonnet45', 'sw_grid_dense_sonnet45'), ('opus', 'sw_grid_dense_opus'), ('opus46', 'sw_grid_dense_opus46'), ('opus48', 'sw_grid_dense_opus48'), ('gpt56luna', 'sw_grid_dense_gpt56luna'), ('gpt56sol', 'sw_grid_dense_gpt56sol')]:
        for ep_dir in sorted(glob.glob(f'experiments/realworld_eval/runs/{rel}/raw/*/')):
            if not os.path.exists(ep_dir + 'steps.jsonl') or not os.path.exists(ep_dir + 'summary.json'):
                continue
            try:
                rows.append(episode_row(ep_dir, model))
            except Exception as e:
                print(f'[WARN] {ep_dir}: {e}', file=sys.stderr)
    if not rows:
        print('no data')
        return
    cols = list(rows[0].keys())
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f'[export] {len(rows)} episodes -> {OUT}')
    from collections import Counter
    c = Counter((r['model'] for r in rows))
    print('per-model:', dict(c))
if __name__ == '__main__':
    main()
