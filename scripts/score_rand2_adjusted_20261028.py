"""Offline adjusted diagnostics of the five saved, complete mini rand2 seeds."""
from pathlib import Path
import json
import statistics
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_nl2p_repeats import score, METRICS, ADJUSTED_METRICS, sha
from scripts.openai_batch_experiment import DATASETS, dataset_path

MODEL = 'gpt-5.4-mini'
DIRECTORY = ROOT / 'results/rebuttal_20261027' / MODEL
MATRIX = ROOT / 'results/batch/rebuttal_20261027_v1/matrix.json'
TARGET = DIRECTORY / 'rand2_adjusted_20261028.json'
CHECKPOINT = DIRECTORY / 'rand2_adjusted_20261028_progress.json'
assert not TARGET.exists(), 'Reuse the saved report instead of recomputing'
matrix = json.loads(MATRIX.read_text(encoding='utf-8'))
strict_path = DIRECTORY / 'comparison.json'
strict = json.loads(strict_path.read_text(encoding='utf-8'))
for relative in ['evaluation.py', 'src/evaluation_helpers.py']:
    assert sha(ROOT / relative) == sha(MATRIX.parent / 'source_snapshot' / relative)
assert strict['evaluator_sha256'] == sha(ROOT / 'evaluation.py')
assert strict['helpers_sha256'] == sha(ROOT / 'src/evaluation_helpers.py')
for item in strict['inputs']:
    assert sha(item['path']) == item['sha256']
    assert sha(Path(item['path']).with_suffix('.pkl')) == item['pkl_sha256']

report = {
    'model': MODEL, 'model_snapshot': matrix['models'][MODEL],
    'created_at': datetime.now(ZoneInfo('Australia/Sydney')).isoformat(timespec='seconds'),
    'method': 'GPT3-to-Plan with two randomly selected demonstrations per query, excluding that query',
    'selection_seeds': matrix['selection_seeds'], 'requests': 2100,
    'source_strict_report': str(strict_path), 'source_strict_sha256': sha(strict_path),
    'evaluator_sha256': sha(ROOT / 'evaluation.py'),
    'helpers_sha256': sha(ROOT / 'src/evaluation_helpers.py'),
    'scoring_wrapper_sha256': sha(ROOT / 'scripts/evaluate_nl2p_repeats.py'),
    'analysis_script_sha256': sha(__file__), 'metrics': [], 'inputs': [], 'summary': [],
    'notes': [
        'Only existing saved predictions are scored; no inference requests or retries.',
        'Original evaluator and matching are unchanged; every strict metric must reproduce the saved strict report.',
        'Mean and sample standard deviation (ddof=1) over five corpus-level seed scores, not three same-prompt repetitions.',
        'Adjusted argument scores are prediction-dependent diagnostics on evaluator-matched actions, not adjusted action F1.',
        'Audit counts are raw denominator deductions before the original true-positive lower-bound clamp, not independent annotation-error counts.'
    ]
}
for run in matrix['runs']:
    if run['model'] != MODEL or not run['condition'].startswith('random-seed'):
        continue
    manifest = json.loads(Path(run['manifest']).read_text(encoding='utf-8'))
    assert manifest['batch_status'] == 'completed' and manifest['collected_at']
    assert manifest['request_counts'] == {'total': 420, 'completed': 420, 'failed': 0}
    assert manifest['failure_count'] == 0 and manifest['model_name'] == report['model_snapshot']
    assert sha(manifest['input_jsonl']) == manifest['input_sha256']
    for dataset, digest in manifest['dataset_sha256'].items():
        assert sha(dataset_path(DATASETS[dataset])) == digest
    solver = manifest['result_solver']
    for dataset, count in matrix['datasets'].items():
        print(f"Scoring {run['condition']} {dataset}", flush=True)
        stem = Path(manifest['result_dir']) / f'{dataset}_{solver}_{MODEL}'
        metrics, provenance = score(stem, dataset, solver, MODEL, count, adjusted=True)
        reference = next(x for x in strict['metrics'] if x['condition'] == run['condition'] and x['dataset'] == dataset)
        for metric in METRICS:
            assert abs(metrics[metric] - reference[metric]) < 1e-12, (run['condition'], dataset, metric)
        assert metrics['missing_predictions'] == 0
        assert metrics['empty_predictions'] == reference['empty_predictions']
        report['metrics'].append({'condition': run['condition'], 'selection_seed': manifest['selection_seed'],
                                  'dataset': dataset, 'batch_id': manifest['batch_id'], **metrics})
        report['inputs'].append(provenance)
        CHECKPOINT.write_text(json.dumps({**report, 'complete': False}, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'condition': run['condition'], 'dataset': dataset,
                          'adjusted_argument_f1': metrics['adjusted_argument_f1'] * 100}), flush=True)
assert len(report['metrics']) == 15
report['complete'] = True
for dataset in ['win2k', 'cooking', 'wikihow']:
    rows = sorted((x for x in report['metrics'] if x['dataset'] == dataset), key=lambda x: x['selection_seed'])
    item = {'dataset': dataset}
    for metric in METRICS[:6] + ADJUSTED_METRICS:
        values = [x[metric] * 100 for x in rows]
        item[metric] = {'seeds': values, 'mean': statistics.mean(values), 'sample_std': statistics.stdev(values)}
    item['raw_adjustment_audit'] = {key: [x['adjustment_audit'][key] for x in rows]
                                  for key in ['gold_deductions', 'prediction_deductions', 'affected_matched_actions']}
    report['summary'].append(item)
for item in report['inputs']:
    assert sha(item['path']) == item['json_sha256']
    assert sha(Path(item['path']).with_suffix('.pkl')) == item['pkl_sha256']
assert sha(strict_path) == report['source_strict_sha256']
with TARGET.open('x', encoding='utf-8') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'report': str(TARGET), 'summary': report['summary']}, ensure_ascii=False), flush=True)
