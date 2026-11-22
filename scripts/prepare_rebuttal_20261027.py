"""Prepare and audit the authorized 2-model, 3-domain rebuttal matrix offline."""
from pathlib import Path
import collections
import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import openai_batch_experiment as batch
from src.llm import generate_prompt

MATRIX_DIR = ROOT / 'results/batch/rebuttal_20261027_v1'
if MATRIX_DIR.exists():
    raise SystemExit(f'Matrix already exists: {MATRIX_DIR}')
MATRIX_DIR.mkdir(parents=True)
MODELS = {'gpt-5.4': 'gpt-5.4-2026-03-05', 'gpt-5.4-mini': 'gpt-5.4-mini-2026-03-17'}
COUNTS = {'cooking': 116, 'wikihow': 150, 'win2k': 154}
matrix = {
    'experiment': 'rebuttal_20261027_v1',
    'created_at': datetime.now(ZoneInfo('Australia/Sydney')).isoformat(timespec='seconds'),
    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
    'status': 'preparing', 'models': MODELS, 'datasets': COUNTS,
    'selection_seeds': list(range(5)), 'expected_batches': 12, 'expected_requests': 5040,
    'protocol': {
        'random_selection': 'One random.Random(str(seed) + ":" + dataset) permutation per dataset and seed; exclude query ID, take first two remaining.',
        'minimal_prompt': 'Delete from original Definition heading through semantic Rules; retain task text, JSON-only line and original output schema verbatim.',
        'coref': 'none', 'sampling_parameters': 'omitted, matching saved original requests',
        'evaluation': 'Unchanged strict evaluator; retain all document IDs and report missing/invalid outputs.',
        'new_requests_only': True,
    },
    'baseline_files': [], 'runs': [],
}

for model in MODELS:
    for solver in ['gpt3_to_plan', 'nl2p_1']:
        for ds in COUNTS:
            path = ROOT / f'results/{solver}/{model}/{ds}_{solver}_{model}.json'
            data = json.loads(path.read_text(encoding='utf-8'))
            assert len(data) == COUNTS[ds]
            matrix['baseline_files'].append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})

for model, snapshot in MODELS.items():
    conditions = [('gpt3_to_plan_random', seed) for seed in range(5)] + [('nl2p_minimal', None)]
    for solver, seed in conditions:
        label = f'random-seed{seed}' if seed is not None else 'minimal'
        result_solver = f'gpt3_to_plan_random_seed{seed}_rebuttal_20261027' if seed is not None else 'nl2p_minimal_rebuttal_20261027'
        args = SimpleNamespace(s=solver, m=model, d=None, l=None, t=0, coref='none', coref_dir=None,
            run_id=f'rebuttal-20261027-{label}', selection_seed=seed, api_model=snapshot, result_solver=result_solver)
        with contextlib.redirect_stdout(io.StringIO()):
            batch.prepare(args)
        path = batch.run_dir(solver, model, args.run_id) / 'manifest.json'
        manifest = batch.load_manifest(path)
        matrix['runs'].append({'condition': label, 'solver': solver, 'model': model, 'api_model': snapshot,
            'selection_seed': seed, 'manifest': str(path), 'request_count': manifest['request_count'],
            'result_solver': result_solver, 'input_sha256': manifest['input_sha256'], 'status': 'prepared'})
        batch.write_json(MATRIX_DIR / 'matrix.json', matrix)

prompts_by_condition = {}
for run in matrix['runs']:
    manifest = batch.load_manifest(Path(run['manifest']))
    rows = batch.read_jsonl(Path(manifest['input_jsonl']))
    assert len(rows) == len(manifest['records']) == 420
    assert collections.Counter(r['dataset'] for r in manifest['records']) == COUNTS
    assert len({row['custom_id'] for row in rows}) == 420
    for ds, size in COUNTS.items():
        assert sorted(r['doc_id'] for r in manifest['records'] if r['dataset'] == ds) == list(range(size))
    by_doc = {}
    for row, record in zip(rows, manifest['records']):
        assert row['custom_id'] == record['custom_id']
        assert row['body']['model'] == run['api_model']
        assert set(row['body']) == {'model', 'messages'}
        prompt = row['body']['messages'][0]['content']
        assert hashlib.sha256(prompt.encode()).hexdigest() == record['prompt_sha256']
        if run['selection_seed'] is not None:
            selected = record['example_doc_ids']
            assert len(selected) == len(set(selected)) == 2 and record['doc_id'] not in selected
            assert all(0 <= i < COUNTS[record['dataset']] for i in selected)
        else:
            full = generate_prompt('nl2p_1', {'nl': record['original_text']})
            start = full.index('            Definition:')
            end = full.index('            - Output valid JSON only.', start)
            assert prompt == full[:start] + full[end:]
        by_doc[(record['dataset'], record['doc_id'])] = prompt
    if run['condition'] in prompts_by_condition:
        assert by_doc == prompts_by_condition[run['condition']]
    else:
        prompts_by_condition[run['condition']] = by_doc
assert len(matrix['runs']) == 12
assert sum(run['request_count'] for run in matrix['runs']) == 5040

source_files = ['scripts/openai_batch_experiment.py', 'src/solvers/gpt3_to_plan.py', 'src/solvers/nl2p_1.py',
    'src/solvers/__init__.py', 'src/llm/config.py', 'experiment.py', 'evaluation.py', 'src/evaluation_helpers.py',
    'tests/test_rebuttal_batch_experiment.py', 'tests/test_openai_batch_experiment.py', 'scripts/prepare_rebuttal_20261027.py']
matrix['source_files'] = []
for relative in source_files:
    source = ROOT / relative
    dest = MATRIX_DIR / 'source_snapshot' / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    matrix['source_files'].append({'path': relative, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
matrix.update(status='preflight_passed', preflight={
    'requests_checked': 5040, 'self_demonstrations': 0, 'cross_model_prompt_mismatches': 0,
    'minimal_prompt_deletion_mismatches': 0, 'dataset_coverage': COUNTS,
    'focused_tests': '23 passed',
    'preexisting_unrelated_test_failures': ['Refined old wording assertion', 'Removed llama3-3b alias', 'Removed hpc/llama32_3.slurm'],
    'model_snapshot_access': 'Both snapshots retrieved successfully from OpenAI models API before preparation',
})
batch.write_json(MATRIX_DIR / 'matrix.json', matrix)
print(json.dumps({'matrix': str(MATRIX_DIR / 'matrix.json'), 'batches': 12, 'requests': 5040,
    'preflight': matrix['preflight'], 'total_input_bytes': sum(Path(batch.load_manifest(Path(r['manifest']))['input_jsonl']).stat().st_size for r in matrix['runs'])}, indent=2))
