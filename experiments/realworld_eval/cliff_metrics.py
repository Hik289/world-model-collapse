from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, '.')
sys.path.insert(0, 'experiments/trace_evaluation')
import trace_v1_metrics as T
from src.evaluation.runner import world_state_facts
FINE_K = (1, 2, 3, 4, 5, 7, 10, 15, 20)

def compute_episode_cliff(episode_dir: Path, derived_root: Path) -> dict:
    manifest = T.load_json(episode_dir / 'manifest.json')
    summary = T.load_json(episode_dir / 'summary.json')
    task_spec = manifest['public_task_spec']
    subgoal_ids = set(task_spec)
    ledger = T.ObservedLedger()
    with (episode_dir / 'steps.jsonl').open(encoding='utf-8') as f:
        steps = [json.loads(line) for line in f if line.strip()]
    out_dir = derived_root / manifest['episode_id']
    out_dir.mkdir(parents=True, exist_ok=True)
    step_rows = []
    for rec in steps:
        t = rec['step']
        obs = rec['model_visible']['observation_partial_state']
        agent_state = rec['updater']['canonical_state']
        gold = rec['evaluator_only']['gold_before']
        obs_facts = ledger.update(obs, t, subgoal_ids)
        agent_facts = T.filter_belief_facts(world_state_facts(agent_state), subgoal_ids)
        gold_facts = T.filter_belief_facts(world_state_facts(gold), subgoal_ids)
        ledger_all = {f for f in ledger.facts if not ledger.is_layout(f)}
        ledger_true = ledger_all & gold_facts
        correct = agent_facts & gold_facts
        forgotten = ledger_true - agent_facts
        wrong = agent_facts - gold_facts
        stale = wrong & ledger_all
        hallucinated = wrong - ledger_all
        ret = {}
        for k in FINE_K:
            eligible = {fact for fact in ledger_true if t - ledger.last_seen.get(fact, t) >= k and (not ledger.is_layout(fact))}
            ret[f'retention_at_{k}'] = T.safe_div(len(agent_facts & eligible), len(eligible)) if eligible else None
        row = {'step': t, 'n_agent': len(agent_facts), 'n_gold': len(gold_facts), 'n_ledger_true': len(ledger_true), 'n_correct': len(correct), 'n_forgotten': len(forgotten), 'n_stale': len(stale), 'n_hallucinated': len(hallucinated), 'belief_precision': T.safe_div(len(correct), len(agent_facts)), 'forgotten_rate': T.safe_div(len(forgotten), len(ledger_true), empty=0.0), 'stale_rate': T.safe_div(len(stale), len(agent_facts), empty=0.0), 'hallucination_rate': T.safe_div(len(hallucinated), len(agent_facts), empty=0.0), **ret, '_ex_forgotten': sorted(forgotten)[:3], '_ex_stale': sorted(stale)[:3], '_ex_hallucinated': sorted(hallucinated)[:3]}
        T.append_jsonl(out_dir / 'cliff_step_metrics.jsonl', row)
        step_rows.append(row)

    def m(name):
        v = [r[name] for r in step_rows if r.get(name) is not None]
        return sum(v) / len(v) if v else None
    ep = {'episode_id': manifest['episode_id'], 'model_key': manifest['model_key'], 'task_name': manifest.get('task_name', manifest['episode_id']), 'horizon': manifest['horizon'], 'state_card': manifest['state_card'], 'n_rooms': manifest.get('sc_components', {}).get('n_rooms'), 'task_seed': manifest.get('task_seed', manifest.get('variation')), 'success': summary['success'], 'steps': summary['steps'], 'mean_belief_precision': m('belief_precision'), 'mean_forgotten_rate': m('forgotten_rate'), 'mean_stale_rate': m('stale_rate'), 'mean_hallucination_rate': m('hallucination_rate'), **{f'mean_retention_at_{k}': m(f'retention_at_{k}') for k in FINE_K}}
    T.dump_json(out_dir / 'cliff_episode_metrics.json', ep)
    return ep

def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('usage: cliff_metrics.py <run_dir>')
    run_dir = Path(sys.argv[1]).resolve()
    raw_root = run_dir / 'raw'
    derived_root = run_dir / 'derived' / 'cliff_v1'
    derived_root.mkdir(parents=True, exist_ok=True)
    all_ep = []
    for ep_dir in sorted(raw_root.iterdir()):
        if not (ep_dir / 'steps.jsonl').exists():
            continue
        if not (ep_dir / 'summary.json').exists():
            continue
        try:
            all_ep.append(compute_episode_cliff(ep_dir, derived_root))
        except Exception as e:
            print(f'[WARN] {ep_dir.name}: {e}', file=sys.stderr)
    T.dump_json(derived_root / 'all_cliff_episode_metrics.json', all_ep)
    print(f'[cliff_metrics] wrote {len(all_ep)} episodes -> {derived_root}')
if __name__ == '__main__':
    main()
