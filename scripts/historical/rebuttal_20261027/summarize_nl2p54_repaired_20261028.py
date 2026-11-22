"""Summarize completed, frozen GPT-5.4 repeat evaluations without rescoring."""
import hashlib
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.result_paths import recorded_path

OUT = ROOT / 'results/nl2p_repeats_20261027_v1/repaired/gpt-5.4'
AUDIT = ROOT / 'results/batch/nl2p_repeats_20261027_v1/repairs/completion_audit_20261028.json'
STRICT = ['action_precision', 'action_recall', 'action_f1',
          'argument_precision', 'argument_recall', 'argument_f1']
ADJUSTED = ['adjusted_argument_precision', 'adjusted_argument_recall', 'adjusted_argument_f1']
DATASETS = [('win2k', 'WHS'), ('cooking', 'CT'), ('wikihow', 'WHG')]

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stats(values):
    return {'values': values, 'mean': statistics.mean(values),
            'sample_sd_ddof1': statistics.stdev(values)}

def display(value):
    return f"{value['mean']:.2f} +/- {value['sample_sd_ddof1']:.2f}"

def main():
    strict_path, adjusted_path = OUT / 'comparison.json', OUT / 'comparison_adjusted.json'
    strict, adjusted, audit = read(strict_path), read(adjusted_path), read(AUDIT)
    assert strict['complete_three_repeat_comparison'] and adjusted['complete_three_repeat_comparison']
    assert len(strict['metrics']) == len(adjusted['metrics']) == 18
    assert audit['total_successes'] == 2520 and audit['parse_failures'] == 0
    index = lambda report: {(x['dataset'], x['condition'], x['repeat']): x for x in report['metrics']}
    si, ai = index(strict), index(adjusted)
    assert set(si) == set(ai)
    for key in si:
        for metric in STRICT:
            assert abs(si[key][metric] - ai[key][metric]) < 1e-12
    for report in [strict, adjusted]:
        for entry in report['inputs']:
            path = recorded_path(entry['path'])
            assert sha(path) == entry['json_sha256']
            assert sha(path.with_suffix('.pkl')) == entry['pkl_sha256']
    summaries, differences, exemptions = [], [], []
    for dataset, label in DATASETS:
        for condition in ['minimal', 'full']:
            row = {'dataset': dataset, 'label': label, 'condition': condition, 'n': 3}
            for metric in STRICT + ADJUSTED:
                source = ai if metric in ADJUSTED else si
                row[metric] = stats([100 * source[(dataset, condition, n)][metric] for n in [1, 2, 3]])
            summaries.append(row)
            audits = [ai[(dataset, condition, n)]['adjustment_audit'] for n in [1, 2, 3]]
            exemptions.append({'dataset': dataset, 'label': label, 'condition': condition,
                'per_repeat': audits,
                **{key: sum(x[key] for x in audits) for key in
                   ['gold_deductions', 'prediction_deductions', 'affected_matched_actions']}})
        row = {'dataset': dataset, 'label': label, 'comparison': 'full minus minimal'}
        for metric in STRICT + ADJUSTED:
            source = ai if metric in ADJUSTED else si
            values = [100 * (source[(dataset, 'full', n)][metric] - source[(dataset, 'minimal', n)][metric]) for n in [1, 2, 3]]
            row[metric] = {**stats(values), 'positive_rounds': sum(v > 0 for v in values),
                          'negative_rounds': sum(v < 0 for v in values),
                          'range_pp': [min(values), max(values)]}
        differences.append(row)
    notes = [
        'All scores are percentages; differences are percentage points; standard deviation uses ddof=1.',
        'Three repeats are descriptive and do not alone establish statistical significance.',
        'Adjusted metrics are argument-only, prediction-dependent diagnostics on evaluator-matched actions; they do not replace strict scores.',
        'Exemptions are raw denominator deductions before the evaluator true-positive floor, not independently verified annotation errors.',
        'The original request bodies contain only model and messages, retaining the original model snapshot and omitted sampling fields.',
        'Retain original successes and the first successful authorized repair; no best-of selection, no fourth round, no successful sample reruns.',
        'All 569 executed repairs address exact HTTP429 insufficient_quota / credit_balance_exhausted failures; no server failures were repaired.',
        'Repairs occurred later than the original batches; the legacy _server_repaired directory name does not imply server errors.',
    ]
    result = {'model': 'gpt-5.4', 'snapshot': 'gpt-5.4-2026-03-05',
        'created_at': datetime.now(ZoneInfo('Australia/Sydney')).isoformat(timespec='seconds'),
        'summary': summaries, 'paired_differences': differences, 'adjustment_exemptions': exemptions,
        'completion_audit': audit, 'notes': notes,
        'sources': [{'path': str(path), 'sha256': sha(path)} for path in [strict_path, adjusted_path, AUDIT]]}
    text = ['GPT-5.4: three-repeat Minimal versus full NL2P', '', *notes, '',
        'Final coverage: 2520/2520 (100%); 0 API failures remaining; 0 parse failures.',
        'Original successes preserved: 1951. Executed repair requests: 569.',
        'Repair submission attempts: 4; 3 executed, 1 rejected with explicit 0/0/0 counts.', '',
        'Strict metrics: mean +/- sample SD across three repeats',
        'Dataset | Condition | Action P | Action R | Action F1 | Argument P | Argument R | Argument F1']
    for row in summaries:
        text.append(' | '.join([row['label'], row['condition']] + [display(row[k]) for k in STRICT]))
    text += ['', 'Adjusted argument diagnostics: mean +/- sample SD',
             'Dataset | Condition | Adjusted argument P | Adjusted argument R | Adjusted argument F1']
    for row in summaries:
        text.append(' | '.join([row['label'], row['condition']] + [display(row[k]) for k in ADJUSTED]))
    text += ['', 'Paired full-minus-minimal differences (rounds 1, 2, 3; percentage points)',
             'Dataset | Metric | Round 1 | Round 2 | Round 3 | Mean +/- SD | Positive rounds']
    for row in differences:
        for metric in STRICT + ADJUSTED:
            v = row[metric]
            text.append(' | '.join([row['label'], metric] + [f'{x:+.4f}' for x in v['values']] +
                                  [display(v), f"{v['positive_rounds']}/3"]))
    text += ['', 'Adjustment audit (raw deductions; three repeats)',
             'Dataset | Condition | Gold deductions r1/r2/r3 | Prediction deductions r1/r2/r3 | Affected matched actions r1/r2/r3']
    for row in exemptions:
        text.append(' | '.join([row['label'], row['condition']] +
            ['/'.join(str(x[k]) for x in row['per_repeat']) for k in
             ['gold_deductions', 'prediction_deductions', 'affected_matched_actions']]))
    text += ['', 'Failure and repair audit (Australia/Sydney timestamps)',
             'Run | Original successes | Original exact credit-429 failures | Repair attempts | Executed repair requests | Final coverage']
    for row in audit['runs']:
        text.append(f"r{row['repeat']} {row['condition']} | {row['original_successes_preserved']} | "
                    f"{row['original_http429_insufficient_quota_credit_balance_exhausted']} | "
                    f"{row['repair_attempt_count']} | {row['repair_requests_executed']} | 420/420")
        for attempt in row['repair_attempts']:
            text.append(f"  Attempt {attempt['attempt']}: {attempt['batch_id']}; "
                f"submitted {attempt['submitted_requests']}; counts {attempt['request_counts']}; "
                f"created {attempt['created_at']}; completed {attempt['completed_at']}; failed {attempt['failed_at']}; "
                f"zero-execution capacity rejection={attempt['zero_execution_capacity_rejection']}")
        text.append(f"  Provenance: {row['provenance_path']}")
    text += ['', 'Sources:'] + [f"{x['path']} SHA256={x['sha256']}" for x in result['sources']]
    (OUT / 'completion_summary_20261028.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (OUT / 'summary_20261028.txt').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(json.dumps({'summary': summaries, 'differences': differences,
                      'exemptions': exemptions, 'report': str(OUT / 'summary_20261028.txt')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
