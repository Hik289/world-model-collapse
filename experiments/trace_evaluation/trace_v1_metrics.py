from __future__ import annotations
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any
sys.path.insert(0, '.')
from src.evaluation.runner import world_state_facts
METRIC_SCHEMA = 'worldmodelphase.metrics_v1'

def load_json(path: Path) -> Any:
    with path.open(encoding='utf-8') as f:
        return json.load(f)

def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write('\n')
    os.replace(tmp, path)

def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + '\n')

def safe_div(a: float, b: float, empty: float=1.0) -> float:
    return a / b if b else empty

def fact_prefix(fact: str) -> str:
    return fact.split(':', 1)[0]

def is_subgoal_object_fact(fact: str, subgoal_ids: set[str]) -> bool:
    parts = fact.split(':')
    if fact.startswith('obj:') and len(parts) >= 2:
        return parts[1] in subgoal_ids
    if fact.startswith('obj_prop:') and len(parts) >= 2:
        return parts[1] in subgoal_ids
    return False

def filter_belief_facts(facts: set[str], subgoal_ids: set[str]) -> set[str]:
    out = set()
    for fact in facts:
        if fact.startswith('blocked_dep:') or fact.startswith('belief:'):
            continue
        if is_subgoal_object_fact(fact, subgoal_ids):
            continue
        out.add(fact)
    return out

def filter_coverage_facts(facts: set[str], subgoal_ids: set[str]) -> set[str]:
    out = set()
    for fact in facts:
        if fact.startswith(('blocked_dep:', 'belief:', 'open_sg:', 'done_sg:', 'inv:')):
            continue
        if is_subgoal_object_fact(fact, subgoal_ids):
            continue
        if fact.startswith('rel:agent:at:'):
            continue
        out.add(fact)
    return out

def slot_of(fact: str) -> str | None:
    if fact.startswith('obj:') and ':type=' in fact:
        return fact.split('=', 1)[0]
    if fact.startswith('obj_prop:') and '=' in fact:
        return fact.split('=', 1)[0]
    if fact.startswith('loc:') and ':type=' in fact:
        return fact.split('=', 1)[0]
    if fact.startswith('rel:agent:at:'):
        return 'rel:agent:at'
    if fact.startswith('open_sg:'):
        return 'sg_status:' + fact.split(':', 1)[1]
    if fact.startswith('done_sg:'):
        return 'sg_status:' + fact.split(':', 1)[1]
    return None

def contradictory_assertions(agent: set[str], gold: set[str]) -> set[str]:
    gold_by_slot = {}
    for fact in gold:
        slot = slot_of(fact)
        if slot is not None:
            gold_by_slot.setdefault(slot, set()).add(fact)
    contradictions = set()
    for fact in agent:
        if fact in gold:
            continue
        slot = slot_of(fact)
        if slot is not None and gold_by_slot.get(slot):
            contradictions.add(fact)
        elif fact.startswith(('loc_contains:', 'inv:')):
            contradictions.add(fact)
    return contradictions

class ObservedLedger:

    def __init__(self, sticky_prefixes: set[str] | None=None, layout_prefixes: set[str] | None=None):
        self.facts: set[str] = set()
        self.first_seen: dict[str, int] = {}
        self.last_seen: dict[str, int] = {}
        self.sticky_prefixes: set[str] = set(sticky_prefixes or ())
        self.layout_prefixes: set[str] = set(layout_prefixes or ())

    def _remove_prefix(self, prefix: str) -> None:
        self.facts = {f for f in self.facts if not f.startswith(prefix)}

    def is_layout(self, fact: str) -> bool:
        return any((fact.startswith(p) for p in self.layout_prefixes))

    def update(self, obs: dict, step: int, subgoal_ids: set[str]) -> set[str]:
        obs_facts = filter_belief_facts(world_state_facts(obs), subgoal_ids)
        loc_contains_sticky = 'loc_contains:' in self.sticky_prefixes
        objects = obs.get('objects') if isinstance(obs.get('objects'), dict) else {}
        for oid in objects:
            self._remove_prefix(f'obj:{oid}:')
            self._remove_prefix(f'obj_prop:{oid}:')
        locations = obs.get('locations') if isinstance(obs.get('locations'), dict) else {}
        for lid, meta in locations.items():
            self._remove_prefix(f'loc:{lid}:')
            if not loc_contains_sticky:
                self._remove_prefix(f'loc_contains:{lid}:')
        self._remove_prefix('inv:')
        self._remove_prefix('open_sg:')
        self._remove_prefix('done_sg:')
        self._remove_prefix('rel:agent:at:')
        if loc_contains_sticky:
            observed_objs: set[str] = set()
            inv = obs.get('inventory') if isinstance(obs.get('inventory'), list) else []
            observed_objs |= {str(i) for i in inv}
            for lid, meta in locations.items():
                meta_d = meta if isinstance(meta, dict) else {}
                for c in meta_d.get('contents') or []:
                    observed_objs |= {str(c)}
            if observed_objs:
                suffixes = tuple((f':{oid}' for oid in observed_objs))
                self.facts = {f for f in self.facts if not (f.startswith('loc_contains:') and f.endswith(suffixes) and (f not in obs_facts))}
            self.facts |= obs_facts
            for fact in obs_facts:
                new = fact not in self.first_seen
                self.first_seen.setdefault(fact, step)
                if fact.startswith('loc_contains:'):
                    if new:
                        self.last_seen[fact] = step
                else:
                    self.last_seen[fact] = step
            return obs_facts
        self.facts |= obs_facts
        for fact in obs_facts:
            self.first_seen.setdefault(fact, step)
            self.last_seen[fact] = step
        return obs_facts

def task_predicates(task_spec: dict) -> list[str]:
    return sorted({predicate for meta in task_spec.values() for predicate in meta.get('all_of') or []})

def gold_predicate_truth(predicate: str, gold: dict) -> bool:
    missing = {p for dep in gold.get('blocked_dependencies') or [] if isinstance(dep, dict) for p in dep.get('missing') or []}
    return predicate not in missing

def agent_predicate_truth(predicate: str, state: dict) -> bool:
    objects = state.get('objects') if isinstance(state.get('objects'), dict) else {}
    inventory = state.get('inventory') if isinstance(state.get('inventory'), list) else []
    locations = state.get('locations') if isinstance(state.get('locations'), dict) else {}
    relations = state.get('relations') if isinstance(state.get('relations'), list) else []
    if predicate.startswith('hold(') and predicate.endswith(')'):
        return predicate[5:-1] in inventory
    if predicate.startswith('open(') and predicate.endswith(')'):
        cid = predicate[5:-1]
        meta = objects.get(cid) if isinstance(objects.get(cid), dict) else {}
        props = meta.get('props') if isinstance(meta.get('props'), dict) else {}
        return props.get('open') is True
    if predicate.startswith('switch_on(') and predicate.endswith(')'):
        sid = predicate[10:-1]
        meta = objects.get(sid) if isinstance(objects.get(sid), dict) else {}
        props = meta.get('props') if isinstance(meta.get('props'), dict) else {}
        return props.get('on') is True
    if predicate.startswith('visited(') and predicate.endswith(')'):
        rid = predicate[8:-1]
        if rid in locations:
            return True
        return any((isinstance(r, dict) and r.get('subj') == 'agent' and (r.get('rel') == 'at') and (r.get('obj') == rid) for r in relations))
    return False

def mean_or_none(values: list[float | None]) -> float | None:
    xs = [x for x in values if x is not None and math.isfinite(x)]
    return statistics.mean(xs) if xs else None

def compute_episode(episode_dir: Path, derived_root: Path) -> dict:
    manifest = load_json(episode_dir / 'manifest.json')
    summary = load_json(episode_dir / 'summary.json')
    task_spec = manifest['public_task_spec']
    subgoal_ids = set(task_spec)
    predicates = task_predicates(task_spec)
    if manifest.get('benchmark') == 'alfworld':
        ledger = ObservedLedger(sticky_prefixes={'loc_contains:'}, layout_prefixes={'loc:', 'obj:', 'obj_prop:', 'rel:agent:at:', 'inv:', 'open_sg:', 'done_sg:'})
    else:
        ledger = ObservedLedger()
    step_metrics = []
    with (episode_dir / 'steps.jsonl').open(encoding='utf-8') as f:
        steps = [json.loads(line) for line in f if line.strip()]
    out_dir = derived_root / manifest['episode_id']
    out_steps = out_dir / 'step_metrics.jsonl'
    if out_steps.exists():
        out_steps.unlink()
    previous_action = None
    for rec in steps:
        t = rec['step']
        obs = rec['model_visible']['observation_partial_state']
        agent_state = rec['updater']['canonical_state']
        gold = rec['evaluator_only']['gold_before']
        obs_facts = ledger.update(obs, t, subgoal_ids)
        agent_facts = filter_belief_facts(world_state_facts(agent_state), subgoal_ids)
        gold_facts = filter_belief_facts(world_state_facts(gold), subgoal_ids)
        ledger_true = ledger.facts & gold_facts
        correct = agent_facts & gold_facts
        precision = safe_div(len(correct), len(agent_facts))
        global_recall = safe_div(len(correct), len(gold_facts))
        observed_recall = safe_div(len(agent_facts & ledger_true), len(ledger_true))
        update_recall = safe_div(len(agent_facts & obs_facts), len(obs_facts))
        false_facts = agent_facts - gold_facts
        contradiction_facts = contradictory_assertions(agent_facts, gold_facts)
        coverage_gold = filter_coverage_facts(gold_facts, subgoal_ids)
        coverage_ledger = filter_coverage_facts(ledger_true, subgoal_ids)
        exploration_coverage = safe_div(len(coverage_ledger), len(coverage_gold), empty=0.0)
        retention_values = {}
        for k in (1, 5, 10, 20):
            eligible = {fact for fact in ledger_true if t - ledger.last_seen.get(fact, t) >= k and (not ledger.is_layout(fact))}
            retention_values[f'retention_at_{k}'] = safe_div(len(agent_facts & eligible), len(eligible)) if eligible else None
        gold_truth = {p: gold_predicate_truth(p, gold) for p in predicates}
        agent_truth = {p: agent_predicate_truth(p, agent_state) for p in predicates}
        task_correct = sum((agent_truth[p] == gold_truth[p] for p in predicates))
        true_predicates = {p for p in predicates if gold_truth[p]}
        task_true_recall = safe_div(sum((agent_truth[p] for p in true_predicates)), len(true_predicates))
        action = rec['transition']['action']
        metric = {'schema': METRIC_SCHEMA + '.step', 'episode_id': manifest['episode_id'], 'step': t, 'belief_precision': precision, 'global_gold_recall': global_recall, 'observed_fact_recall': observed_recall, 'new_observation_update_recall': update_recall, 'false_fact_rate': safe_div(len(false_facts), len(agent_facts), empty=0.0), 'contradiction_rate': safe_div(len(contradiction_facts), len(agent_facts), empty=0.0), 'exploration_coverage': exploration_coverage, 'task_predicate_accuracy': safe_div(task_correct, len(predicates)), 'task_true_predicate_recall': task_true_recall, 'action_valid': rec['evaluator_only']['action_valid'], 'action_repeated_from_previous': action == previous_action, 'agent_fact_count': len(agent_facts), 'gold_fact_count': len(gold_facts), 'observed_ledger_fact_count': len(ledger_true), 'new_observation_fact_count': len(obs_facts), 'false_fact_count': len(false_facts), 'contradiction_count': len(contradiction_facts), **retention_values}
        append_jsonl(out_steps, metric)
        step_metrics.append(metric)
        previous_action = action

    def vals(name):
        return [m.get(name) for m in step_metrics]
    episode_metrics = {'schema': METRIC_SCHEMA + '.episode', 'episode_id': manifest['episode_id'], 'model': manifest['model'], 'model_key': manifest['model_key'], 'state_card': manifest['state_card'], 'dep_density': manifest['dep_density'], 'horizon': manifest['horizon'], 'task_seed': manifest['task_seed'], 'success': summary['success'], 'steps': summary['steps'], 'completed_subgoals': summary['completed_subgoals'], 'total_subgoals': summary['total_subgoals'], 'action_validity_rate': summary['action_validity_rate'], 'legacy_mean_wsa': summary['mean_legacy_wsa'], 'legacy_final_wsa': summary['final_legacy_wsa'], 'mean_belief_precision': mean_or_none(vals('belief_precision')), 'final_belief_precision': vals('belief_precision')[-1], 'mean_observed_fact_recall': mean_or_none(vals('observed_fact_recall')), 'final_observed_fact_recall': vals('observed_fact_recall')[-1], 'mean_new_observation_update_recall': mean_or_none(vals('new_observation_update_recall')), 'mean_false_fact_rate': mean_or_none(vals('false_fact_rate')), 'final_false_fact_rate': vals('false_fact_rate')[-1], 'mean_contradiction_rate': mean_or_none(vals('contradiction_rate')), 'final_contradiction_rate': vals('contradiction_rate')[-1], 'mean_exploration_coverage': mean_or_none(vals('exploration_coverage')), 'final_exploration_coverage': vals('exploration_coverage')[-1], 'mean_task_predicate_accuracy': mean_or_none(vals('task_predicate_accuracy')), 'final_task_predicate_accuracy': vals('task_predicate_accuracy')[-1], 'mean_task_true_predicate_recall': mean_or_none(vals('task_true_predicate_recall')), 'final_task_true_predicate_recall': vals('task_true_predicate_recall')[-1], 'mean_retention_at_1': mean_or_none(vals('retention_at_1')), 'mean_retention_at_5': mean_or_none(vals('retention_at_5')), 'mean_retention_at_10': mean_or_none(vals('retention_at_10')), 'mean_retention_at_20': mean_or_none(vals('retention_at_20')), 'repeated_action_rate': safe_div(sum((bool(x) for x in vals('action_repeated_from_previous'))), len(step_metrics), empty=0.0), 'updater_parse_failures': summary['updater_parse_failures'], 'planner_parse_failures': summary['planner_parse_failures'], 'transport_retries': summary['transport_retries']}
    dump_json(out_dir / 'episode_metrics.json', episode_metrics)
    return episode_metrics

def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('usage: trace_v1_metrics.py <run_dir>')
    run_dir = Path(sys.argv[1]).resolve()
    raw_root = run_dir / 'raw'
    derived_root = run_dir / 'derived' / 'metrics_v1'
    derived_root.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    errors = []
    for episode_dir in sorted((p for p in raw_root.iterdir() if p.is_dir())):
        if not (episode_dir / 'summary.json').exists():
            continue
        try:
            all_metrics.append(compute_episode(episode_dir, derived_root))
        except Exception as exc:
            errors.append({'episode_id': episode_dir.name, 'error': repr(exc)})
    dump_json(derived_root / 'all_episode_metrics.json', {'schema': METRIC_SCHEMA + '.collection', 'episode_count': len(all_metrics), 'episodes': all_metrics, 'errors': errors})
    print(json.dumps({'metric_schema': METRIC_SCHEMA, 'episodes_computed': len(all_metrics), 'errors': errors, 'output': str(derived_root)}, sort_keys=True))
if __name__ == '__main__':
    main()
