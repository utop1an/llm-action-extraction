"""Recheck frozen RiCoRecA artifacts, replay and text-only inference without training.

Writes only a new review directory. Historical data, checkpoints and receipts stay
unchanged. Run from the repository root after any other GPU audit has finished.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from audit_ceasdrl_risec import audit_replay, read, sha
from ceasdrl_spans import SpanDqn, SpanFeatures, evaluate_spans
from evaluate_ceasdrl_holdout import independent_char_scores
from evaluate_ceasdrl_spans import assemble_predictions
from prepare_ricoreca_data import convert_record, published_records
from prepare_ricoreca_split import make_split
from risec_spans import make_span_examples, predicted_span_contexts


def check_features(directory):
    manifest = read(directory / 'manifest.json')
    directories = [directory] + [(directory / p).resolve()
                                 for p in manifest.get('cache_directories', [])]
    for key, expected in manifest['feature_sha256'].items():
        path = next(p / (key + '.npy') for p in directories
                    if (p / (key + '.npy')).is_file())
        assert sha(path) == expected, str(path)
        values = np.load(path)
        assert values.shape[1] == 868 and np.isfinite(values).all()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Review directory already exists')
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    data = root / 'data/procedural/ricoreca'
    run = root / 'results/ceasdrl_ricoreca_64_16_20_seed0'
    final = run / 'final'
    report = {'status': 'running', 'no_training_performed': True,
              'started_at_utc': datetime.now(timezone.utc).isoformat()}
    rows = {s: [json.loads(line) for line in (data / 'converted' / (s + '.jsonl'))
               .read_text(encoding='utf-8').splitlines()] for s in ('train', 'dev', 'test')}
    split = read(data / 'converted/split.json')
    regenerated = make_split([r for group in rows.values() for r in group], split['split_seed'])
    assert regenerated['document_ids'] == split['document_ids']
    for s, group in rows.items():
        assert [r['id'] for r in group] == split['document_ids'][s]
    for source in split['source_files']:
        assert sha(data / 'raw' / source['path']) == source['sha256']
    released, _ = published_records(data / 'raw')
    raw = {r['meta']['id']: r for p in (data / 'raw').rglob('Annotator_P/prodigy_*.json')
           for r in json.loads(p.read_text(encoding='utf-8-sig'))}
    for group in rows.values():
        for row in group:
            assert convert_record(raw[row['id']], released[row['id']]) == {
                k: v for k, v in row.items() if k != 'split'}
    conversion = read(data / 'converted/conversion_audit.json')
    for name, expected in conversion['source_sha256'].items():
        assert sha(root / 'scripts' / name) == expected
    manifest_path = data / 'features_train_dev/manifest.json'
    manifest = check_features(manifest_path.parent)
    hashes = {s: sha(data / 'converted' / (s + '.jsonl')) for s in rows}
    assert hashes == conversion['dataset_sha256']
    assert manifest['dataset_sha256'] == {s: hashes[s] for s in ('train', 'dev')}
    result = read(final / 'receipt.json')
    freeze = read(final / 'freeze.json')
    assert result['status'] == 'completed'
    assert not result['inference_uses_gold_labels'] and not result['test_evaluated_during_training']
    assert hashes == result['dataset_sha256'] == freeze['dataset_sha256']
    assert sha(final / 'freeze.json') == result['freeze_sha256']
    assert sha(data / 'converted/split.json') == freeze['split_sha256']
    assert freeze['frozen_at_utc'] < result['completed_at_utc']
    for name, expected in freeze['evaluation_source_sha256'].items():
        assert sha(root / 'scripts' / name) == expected
    report.update({'split_counts': split['counts'], 'split_seed': split['split_seed'],
                   'split_reproduced': True, 'duplicate_ids_and_normalized_texts': 0,
                   'source_files_checked': len(split['source_files']),
                   'all_converted_records_reproduced_from_raw': True,
                   'training': {}, 'inference_cache_hashes': {}})
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for mode in ('act', 'arg'):
        receipt = read(run / mode / 'receipt.json')
        assert receipt['status'] == 'completed'
        assert receipt['test'] is None and not receipt['test_evaluated']
        assert receipt['weights_changed'] and receipt['checkpoint_reload_predictions_equal']
        assert receipt['losses_and_gradients_finite']
        assert receipt['feature_manifest_sha256'] == sha(manifest_path)
        assert receipt['epochs_completed'] == 10 and all(h['complete'] for h in receipt['history'])
        assert receipt['best_epoch'] == max(receipt['history'], key=lambda h: h['dev']['f1'])['epoch']
        assert receipt['dev']['f1'] == max(h['dev']['f1'] for h in receipt['history'])
        assert manifest['examples'][mode]['test'] == []
        for s in ('train', 'dev'):
            assert make_span_examples(rows[s], mode) == manifest['examples'][mode][s]
        # Also validates every test span fits the declared context and BIO scheme;
        # this is a data coverage check, never fed into the model below.
        make_span_examples(rows['test'], mode)
        for name, expected in receipt['source_sha256'].items():
            assert sha(root / name) == expected, name
        assert sha(run / mode / 'receipt.json') == freeze['training_receipt_sha256'][mode]
        assert sha(run / mode / 'best.pt') == result['checkpoint_sha256'][mode] == freeze['checkpoint_sha256'][mode]
        last = torch.load(run / mode / 'last.pt', map_location='cpu', weights_only=False)
        best = torch.load(run / mode / 'best.pt', map_location='cpu', weights_only=False)
        assert best['epoch'] == receipt['best_epoch']
        assert best['configuration'] == last['configuration'] == receipt['configuration']
        assert last['training_steps'] == receipt['training_steps']
        assert last['updates'] == receipt['optimizer_updates']
        assert all(torch.isfinite(v).all() for v in best['state_dict'].values())
        assert all(v.data_ptr() != last['target_state'][k].data_ptr() for k, v in last['state_dict'].items())
        replay = audit_replay(last, manifest['examples'][mode]['train'], 600 if mode == 'act' else 100)
        assert set(last['visited']) == {e['id'] for e in manifest['examples'][mode]['train']}
        compatibility = read(run / mode / 'trained_keras_compatibility.json')
        assert compatibility['passed'] and compatibility['num_q_outputs'] == 3
        assert compatibility['real_validation_contexts'] == 4
        report['training'][mode] = {'best_epoch': best['epoch'], 'epochs_completed': 10,
                                   'training_steps': receipt['training_steps'],
                                   'optimizer_updates': receipt['optimizer_updates'],
                                   'dev_f1': receipt['dev']['f1'], 'replay': replay,
                                   'configuration': receipt['configuration'],
                                   'prior_keras_compatibility_receipt': compatibility}
        del last, best
        feat_path = final / (mode + '_features')
        feat = check_features(feat_path)
        assert not feat['uses_gold_action_or_argument_labels']
        assert all('labels' not in e and 'gold_spans' not in e for e in feat['examples'][mode]['test'])
        assert sha(feat_path / 'manifest.json') == result['inference_feature_manifest_sha256'][mode]
        report['inference_cache_hashes'][mode] = len(feat['feature_sha256'])
        print(mode + ': source, checkpoint, features and 50,000 replay rows checked', flush=True)
    plain = {r['id']: {k: r[k] for k in ('words', 'text', 'token_offsets')} for r in rows['test']}
    for mode in ('act', 'arg'):
        device = torch.device(report['training'][mode]['configuration']['device'])
        features = SpanFeatures(final / (mode + '_features'), mode, device)
        saved = [{k: v for k, v in e.items() if k != 'index'} for e in features.examples['test']]
        if mode == 'act':
            contexts = saved
            for e in contexts:
                assert e['tokens'] == plain[e['doc_id']]['words']
        else:
            assert contexts == saved
        assert all('labels' not in e and 'gold_spans' not in e for e in contexts)
        model = SpanDqn(mode).to(device)
        model.load_state_dict(torch.load(run / mode / 'best.pt', map_location=device, weights_only=False)['state_dict'])
        _, predictions = evaluate_spans(model, features, contexts, batch_size=7)
        if mode == 'act':
            contexts = predicted_span_contexts(plain, predictions)
        del model, features
    fresh = assemble_predictions(plain, contexts, predictions)
    assert fresh == read(final / 'predictions.json')
    metrics = independent_char_scores({r['id']: r for r in rows['test']}, fresh)
    assert metrics == result['metrics'] == result['independent_character_scoring']
    report.update({'status': 'passed', 'metrics': metrics,
                   'source_and_checkpoint_hashes_match': True,
                   'test_absent_from_training_features': True,
                   'inference': {'documents': len(fresh), 'predicted_actions': len(contexts),
                                 'annotations_stripped': True, 'batch_size': 7,
                                 'original_batch_size': 32,
                                 'fresh_all_document_predictions_identical': True},
                   'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                   'review_script_sha256': sha(Path(__file__))})
    for name, value in [('audit', report), ('fresh_predictions', fresh)]:
        (args.output / (name + '.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'metrics': metrics, 'inference': report['inference']}, indent=2))


if __name__ == '__main__':
    main()
