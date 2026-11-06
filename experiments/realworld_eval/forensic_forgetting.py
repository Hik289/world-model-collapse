from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, '.')
sys.path.insert(0, 'experiments/trace_evaluation')
from src.evaluation.runner import world_state_facts

def facts_of(state):
    try:
        return set(world_state_facts(state))
    except Exception:
        return set()

def analyze_episode(ep_dir: Path):
    steps = [json.loads(l) for l in (ep_dir / 'steps.jsonl').open() if l.strip()]
    manifest = json.load((ep_dir / 'manifest.json').open())
    rows = []
    dropped_examples = []
    max_in_tok = 0
    any_length_finish = False
    for rec in steps:
        t = rec['step']
        mv = rec['model_visible']
        u = rec['updater']
        prev = mv.get('prev_agent_world_state', {})
        prev_facts = facts_of(prev)
        new_facts = facts_of(u.get('canonical_state', {}))
        tr = u.get('transport', {})
        usage = tr.get('usage') or {}
        in_tok = usage.get('input_tokens') or usage.get('prompt_tokens') or 0
        out_tok = usage.get('output_tokens') or usage.get('completion_tokens') or 0
        finish = tr.get('finish_reason')
        if finish == 'length':
            any_length_finish = True
        max_in_tok = max(max_in_tok, in_tok)
        dropped = prev_facts - new_facts
        dropped_mem = {f for f in dropped if f.startswith(('loc_contains:', 'obj:', 'obj_prop:', 'rel:')) and (not f.startswith('rel:agent:at:'))}
        rows.append({'step': t, 'prev_n': len(prev_facts), 'new_n': len(new_facts), 'in_tok': in_tok, 'out_tok': out_tok, 'finish': finish, 'dropped_mem_n': len(dropped_mem), 'valid_json': u.get('valid_json'), 'fallback': u.get('fallback_used')})
        if dropped_mem and len(dropped_examples) < 5:
            dropped_examples.append({'step': t, 'example_dropped_facts': sorted(dropped_mem)[:4], 'prev_had': len(prev_facts), 'kept': len(new_facts), 'in_tok': in_tok, 'out_tok': out_tok, 'finish': finish})
    return {'episode': manifest['episode_id'], 'horizon': manifest['horizon'], 'task': manifest.get('task_name'), 'state_card': manifest['state_card'], 'max_input_tokens': max_in_tok, 'any_output_length_truncation': any_length_finish, 'total_dropped_mem_events': sum((r['dropped_mem_n'] for r in rows)), 'all_valid_json': all((r['valid_json'] for r in rows)), 'any_fallback': any((r['fallback'] for r in rows)), 'smoking_gun_examples': dropped_examples, 'rows': rows}

def main():
    run_dir = Path(sys.argv[1]).resolve()
    raw = run_dir / 'raw'
    eps = []
    for ep in sorted(raw.iterdir()):
        if (ep / 'steps.jsonl').exists() and (ep / 'manifest.json').exists():
            try:
                eps.append(analyze_episode(ep))
            except Exception as e:
                print(f'[WARN] {ep.name}: {e}', file=sys.stderr)
    out = run_dir / 'derived' / 'forensic_forgetting.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(eps, indent=2))
    max_tok = max((e['max_input_tokens'] for e in eps), default=0)
    any_trunc = any((e['any_output_length_truncation'] for e in eps))
    all_valid = all((e['all_valid_json'] for e in eps))
    any_fb = any((e['any_fallback'] for e in eps))
    total_drops = sum((e['total_dropped_mem_events'] for e in eps))
    print(f'=== FORENSIC VERDICT ({len(eps)} episodes) ===')
    print(f'max updater input_tokens (any step)   : {max_tok}  (haiku ctx = 200,000)')
    print(f"any output finish_reason=='length'    : {any_trunc}  (True => output truncation)")
    print(f'all updater outputs valid_json        : {all_valid}')
    print(f'any fallback_used                     : {any_fb}')
    print(f"total 'fed-back-but-dropped' mem events: {total_drops}")
    print()
    print('>>> If max_input_tokens << 200k AND no length-truncation AND valid_json')
    print(">>> AND many fed-back-but-dropped events => model WON'T store (real forgetting)")
    print()
    shown = 0
    for e in eps:
        for ex in e['smoking_gun_examples']:
            if shown >= 6:
                break
            print(f"[{e['episode']}] step {ex['step']}: model was FED {ex['prev_had']} facts (in_tok={ex['in_tok']}, out_tok={ex['out_tok']}, finish={ex['finish']}, valid JSON), rewrote only {ex['kept']}, DROPPED e.g. {ex['example_dropped_facts']}")
            shown += 1
        if shown >= 6:
            break
if __name__ == '__main__':
    main()
