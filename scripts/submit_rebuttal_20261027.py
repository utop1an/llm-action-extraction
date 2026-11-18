"""Submit the preflight-checked matrix, preserving every remote job ID."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import openai_batch_experiment as batch

path = ROOT / 'results/batch/rebuttal_20261027_v1/matrix.json'
matrix = batch.load_manifest(path)
assert matrix['preflight']['requests_checked'] == 5040
assert len(matrix['runs']) == 12
for baseline in matrix['baseline_files']:
    assert hashlib.sha256(Path(baseline['path']).read_bytes()).hexdigest() == baseline['sha256']
for source in matrix['source_files']:
    assert hashlib.sha256((ROOT / source['path']).read_bytes()).hexdigest() == source['sha256']
for run in matrix['runs']:
    manifest = batch.load_manifest(Path(run['manifest']))
    assert manifest['request_count'] == 420
    assert manifest['model_name'] == run['api_model']
    assert hashlib.sha256(Path(manifest['input_jsonl']).read_bytes()).hexdigest() == run['input_sha256']

matrix['status'] = 'submitting'
batch.write_json(path, matrix)
for index, run in enumerate(matrix['runs'], 1):
    print(f"[{index}/12] {run['model']} {run['condition']} (420 requests)", flush=True)
    try:
        batch.submit(SimpleNamespace(manifest=run['manifest']))
    except Exception as exc:
        run['submission_error'] = {'type': type(exc).__name__, 'message': str(exc)[:1200]}
        run['status'] = 'submission_error'
        matrix['status'] = 'submission_error'
        batch.write_json(path, matrix)
        print(json.dumps(run['submission_error']), flush=True)
        raise SystemExit(1)
    manifest = batch.load_manifest(Path(run['manifest']))
    run.update(batch_id=manifest['batch_id'], input_file_id=manifest['input_file_id'], status=manifest['batch_status'])
    batch.write_json(path, matrix)
    print(json.dumps({'batch_id': run['batch_id'], 'status': run['status']}), flush=True)

matrix['status'] = 'submitted'
matrix['submitted_at'] = datetime.now(ZoneInfo('Australia/Sydney')).isoformat(timespec='seconds')
batch.write_json(path, matrix)
print('SUBMITTED: 12 batches, 5040 requests. ' + str(path), flush=True)
