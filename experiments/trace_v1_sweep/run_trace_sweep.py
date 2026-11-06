from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / 'experiments/trace_evaluation/trace_v1_runner.py'
METRICS = PROJECT_ROOT / 'experiments/trace_evaluation/trace_v1_metrics.py'
SC_VALUES = [10, 20, 40]
DD_VALUES = [1, 2, 4, 6]
MODEL_PRIORITY = ['mini', 'gpt54', 'gpt55']

def horizon_for(sc: int) -> int:
    return 100 if sc == 40 else 40

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def episode_id(model: str, sc: int, dd: int, horizon: int, seed: int) -> str:
    suffix = '' if seed == 0 else f'_s{seed}'
    return f'{model}_sc{sc}_dd{dd}_T{horizon}{suffix}'

def is_done(run_dir: Path, model: str, sc: int, dd: int, horizon: int, seed: int) -> bool:
    eid = episode_id(model, sc, dd, horizon, seed)
    return (run_dir / 'raw' / eid / 'summary.json').exists()

def build_jobs(models: list[str], seeds: int, sc_values: list[int] | None=None, dd_values: list[int] | None=None) -> list[dict]:
    sc_list = sc_values if sc_values is not None else SC_VALUES
    dd_list = dd_values if dd_values is not None else DD_VALUES
    jobs = []
    for model in models:
        for seed in range(seeds):
            for sc in sc_list:
                for dd in dd_list:
                    horizon = horizon_for(sc)
                    jobs.append({'model': model, 'sc': sc, 'dd': dd, 'horizon': horizon, 'seed': seed, 'episode_id': episode_id(model, sc, dd, horizon, seed)})
    return jobs

def run_episode(run_dir: Path, job: dict, log_dir: Path) -> dict:
    eid = job['episode_id']
    log_path = log_dir / f'{eid}.log'
    cmd = [sys.executable, str(RUNNER), job['model'], str(job['sc']), str(job['dd']), str(job['horizon']), str(run_dir), str(job['seed'])]
    started = time.time()
    with log_path.open('w', encoding='utf-8') as logf:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    ok = proc.returncode == 0 and is_done(run_dir, job['model'], job['sc'], job['dd'], job['horizon'], job['seed'])
    return {'episode_id': eid, 'ok': ok, 'returncode': proc.returncode, 'elapsed_s': elapsed}

def write_checkpoint(ckpt_path: Path, state: dict) -> None:
    tmp = ckpt_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding='utf-8')
    tmp.replace(ckpt_path)

def run_metrics(run_dir: Path, log_dir: Path) -> None:
    log_path = log_dir / 'metrics.log'
    with log_path.open('w', encoding='utf-8') as logf:
        subprocess.run([sys.executable, str(METRICS), str(run_dir)], cwd=str(PROJECT_ROOT), stdout=logf, stderr=subprocess.STDOUT)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default=None)
    ap.add_argument('--models', default=','.join(MODEL_PRIORITY))
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--sc-values', default=None, help='comma-separated sc list (default 10,20,40)')
    ap.add_argument('--dd-values', default=None, help='comma-separated dd list (default 1,2,4,6)')
    ap.add_argument('--metrics-only', action='store_true')
    ap.add_argument('--shard', default='0/1', help='i/n: this worker runs jobs where (index %% n == i). Shards are non-overlapping, so multiple workers on one run_dir are safe. Priority order is preserved within each shard (mini first).')
    args = ap.parse_args()
    shard_i, shard_n = (int(x) for x in args.shard.split('/'))
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        run_dir = PROJECT_ROOT / 'experiments/trace_evaluation/trace_runs' / f"trace_sweep_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / f'checkpoint_shard{shard_i}of{shard_n}.json'
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    for m in models:
        assert m in MODEL_PRIORITY, f'unknown model {m}'
    models = [m for m in MODEL_PRIORITY if m in models]
    sc_values = [int(x) for x in args.sc_values.split(',')] if args.sc_values else None
    dd_values = [int(x) for x in args.dd_values.split(',')] if args.dd_values else None
    all_jobs = build_jobs(models, args.seeds, sc_values, dd_values)
    jobs = [j for idx, j in enumerate(all_jobs) if idx % shard_n == shard_i]
    manifest = {'schema': 'worldmodelphase.trace_sweep.manifest', 'created_at': utc_now(), 'run_dir': str(run_dir), 'models': models, 'sc_values': SC_VALUES, 'dd_values': DD_VALUES, 'seeds': args.seeds, 'horizon_policy': {'sc10': 40, 'sc20': 40, 'sc40': 100}, 'total_jobs': len(jobs), 'runner': str(RUNNER), 'metrics': str(METRICS), 'note': 'seeds = distinct task_index within (sc,dd); seed 0 == probe anchor'}
    write_checkpoint(run_dir / 'sweep_manifest.json', manifest)
    if args.metrics_only:
        run_metrics(run_dir, log_dir)
        print(f'metrics-only done for {run_dir}')
        return
    completed, skipped, failed = ([], [], [])
    total = len(jobs)
    for i, job in enumerate(jobs, 1):
        if is_done(run_dir, job['model'], job['sc'], job['dd'], job['horizon'], job['seed']):
            skipped.append(job['episode_id'])
            print(f"[{i}/{total}] SKIP (done) {job['episode_id']}", flush=True)
            continue
        print(f"[{i}/{total}] RUN {job['episode_id']}", flush=True)
        result = run_episode(run_dir, job, log_dir)
        if result['ok']:
            completed.append(job['episode_id'])
            print(f"[{i}/{total}] OK {job['episode_id']} ({result['elapsed_s']:.0f}s)", flush=True)
        else:
            failed.append({'episode_id': job['episode_id'], **result})
            print(f"[{i}/{total}] FAIL {job['episode_id']} rc={result['returncode']}", flush=True)
        write_checkpoint(ckpt_path, {'updated_at': utc_now(), 'total': total, 'completed_this_run': completed, 'skipped_already_done': skipped, 'failed': failed, 'progress': f'{i}/{total}'})
        next_model = jobs[i]['model'] if i < total else None
        if shard_i == 0 and next_model != job['model']:
            print(f"--- model {job['model']} block done; running metrics ---", flush=True)
            run_metrics(run_dir, log_dir)
    if shard_i == 0:
        run_metrics(run_dir, log_dir)
    print(json.dumps({'run_dir': str(run_dir), 'completed_this_run': len(completed), 'skipped': len(skipped), 'failed': len(failed), 'failed_ids': [f['episode_id'] for f in failed]}, indent=2))
if __name__ == '__main__':
    main()
