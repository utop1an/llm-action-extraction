"""Wait for capacity and retry only zero-request token-limit rejections."""
import collections
import concurrent.futures
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import openai_batch_experiment as batch

path = ROOT / 'results/batch/rebuttal_20261027_v1/matrix.json'
clients = {model: batch.get_openai_client(model).with_options(timeout=30, max_retries=0)
    for model in ['gpt-5.4', 'gpt-5.4-mini']}
active = {'validating', 'in_progress', 'finalizing', 'cancelling'}

def now():
    return datetime.now(ZoneInfo('Australia/Sydney')).isoformat(timespec='seconds')

def refresh(run):
    return clients[run['model']].batches.retrieve(run['batch_id']).model_dump()

batch.write_json(path.parent / 'queue_worker.json', {
    'pid': os.getpid(), 'started_at': now(), 'status': 'running',
    'purpose': 'Collect completed outputs and sequentially resubmit only the capacity-rejected, unexecuted batches until all 12 are accepted.',
})
connection_failures = 0
while True:
    matrix = batch.load_manifest(path)
    # A manifest is persisted immediately after create; recover its ID if a
    # previous worker stopped before the aggregate matrix was updated.
    for run in matrix['runs']:
        manifest = batch.load_manifest(Path(run['manifest']))
        if manifest.get('batch_id'):
            run['batch_id'] = manifest['batch_id']
        elif manifest.get('submission_history'):
            batch.submit(SimpleNamespace(manifest=run['manifest']))
            run['batch_id'] = batch.load_manifest(Path(run['manifest']))['batch_id']
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            remote_states = list(pool.map(refresh, matrix['runs']))
    except Exception as exc:
        connection_failures += 1
        print(json.dumps({'status_check_error': type(exc).__name__, 'attempt': connection_failures}), flush=True)
        if connection_failures >= 10:
            raise
        time.sleep(60)
        continue
    connection_failures = 0
    checked = now()
    for run, remote in zip(matrix['runs'], remote_states):
        run.update(status=remote['status'], request_counts=remote.get('request_counts'), errors=remote.get('errors'))
        manifest_path = Path(run['manifest'])
        manifest = batch.load_manifest(manifest_path)
        manifest.update(batch_status=remote['status'], request_counts=remote.get('request_counts'),
            errors=remote.get('errors'), output_file_id=remote.get('output_file_id'),
            error_file_id=remote.get('error_file_id'), last_checked_at=checked)
        batch.write_json(manifest_path, manifest)
        if remote['status'] == 'completed' and not manifest.get('collected_at'):
            print(f"COLLECT: {run['model']} {run['condition']}", flush=True)
            local_output = manifest_path.parent / 'output.jsonl'
            batch.collect(SimpleNamespace(manifest=str(manifest_path),
                output_jsonl=str(local_output) if local_output.exists() else None,
                output_file_id=None, result_solver=None))
            manifest = batch.load_manifest(manifest_path)
            output_rows = batch.read_jsonl(local_output)
            response_models = collections.Counter((row.get('response') or {}).get('body', {}).get('model') for row in output_rows)
            run['collection'] = {'result_dir': manifest['result_dir'], 'failure_count': manifest['failure_count'],
                'response_models': dict(response_models), 'collected_at': manifest['collected_at']}
    matrix['last_checked_at'] = checked
    matrix['status_counts'] = dict(collections.Counter(run['status'] for run in matrix['runs']))
    batch.write_json(path.parent / 'remote_status.json', {'checked_at': checked, 'batches': remote_states})
    pending = []
    unexpected = []
    for run in matrix['runs']:
        if run['status'] in {'failed', 'expired', 'cancelled'}:
            errors = (run.get('errors') or {}).get('data') or []
            counts = run.get('request_counts') or {}
            if run['status'] == 'failed' and errors and all(e.get('code') == 'token_limit_exceeded' for e in errors) and all(counts.get(k, 0) == 0 for k in ['total', 'completed', 'failed']):
                pending.append(run)
            else:
                unexpected.append(run)
    matrix['pending_capacity'] = [{'model': r['model'], 'condition': r['condition']} for r in pending]
    matrix['status'] = 'waiting_for_capacity' if pending else 'all_batches_accepted'
    batch.write_json(path, matrix)
    print(json.dumps({'checked_at': checked, 'status_counts': matrix['status_counts'],
        'completed_requests': sum((r.get('request_counts') or {}).get('completed', 0) for r in matrix['runs']),
        'pending_capacity': matrix['pending_capacity']}), flush=True)
    if unexpected:
        matrix['status'] = 'requires_attention'
        batch.write_json(path, matrix)
        print('Unexpected terminal failure; no automatic retry: ' + json.dumps(unexpected), flush=True)
        raise SystemExit(1)
    if not pending and all(r['status'] != 'validating' for r in matrix['runs']):
        matrix['all_batches_accepted_at'] = now()
        batch.write_json(path, matrix)
        print('ALL 12 BATCHES ACCEPTED; 5040 total planned requests.', flush=True)
        batch.write_json(path.parent / 'queue_worker.json', {'pid': os.getpid(), 'status': 'complete', 'finished_at': now()})
        break
    for model in clients:
        eligible = [r for r in pending if r['model'] == model]
        busy = any(r['model'] == model and r['status'] in active for r in matrix['runs'])
        if not eligible or busy:
            continue
        run = eligible[0]
        manifest_path = Path(run['manifest'])
        manifest = batch.load_manifest(manifest_path)
        assert hashlib.sha256(Path(manifest['input_jsonl']).read_bytes()).hexdigest() == run['input_sha256']
        history = manifest.setdefault('submission_history', [])
        if len(history) >= 3:
            raise SystemExit('Repeated capacity rejection even when this matrix has no other active job for the model; investigate before retrying again.')
        history.append({key: manifest.get(key) for key in ['batch_id', 'batch_status', 'input_file_id',
            'submission_started_at', 'submitted_at', 'last_checked_at', 'request_counts', 'errors']})
        for key in ['batch_id', 'batch_status', 'submission_started_at', 'submitted_at', 'request_counts', 'errors', 'output_file_id', 'error_file_id']:
            manifest.pop(key, None)
        batch.write_json(manifest_path, manifest)
        print(f"RETRY AFTER CAPACITY RELEASE: {model} {run['condition']} (same input file, 420 requests)", flush=True)
        batch.submit(SimpleNamespace(manifest=str(manifest_path)))
        manifest = batch.load_manifest(manifest_path)
        run.update(batch_id=manifest['batch_id'], status=manifest['batch_status'],
            request_counts=None, errors=None, submission_history=history)
        batch.write_json(path, matrix)
    time.sleep(60)
