"""Read-only verification of the fixed RiCoRecA experiment and final predictions."""
import argparse
import hashlib
import json
from pathlib import Path

try:
    from .evaluate_ceasdrl_holdout import independent_char_scores
    from .prepare_ricoreca_data import published_records, convert_record
    from .prepare_ricoreca_split import make_split
except ImportError:
    from evaluate_ceasdrl_holdout import independent_char_scores
    from prepare_ricoreca_data import published_records, convert_record
    from prepare_ricoreca_split import make_split


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('data/procedural/ricoreca'))
    p.add_argument('--run', type=Path, default=Path('results/ceasdrl_ricoreca_64_16_20_seed0'))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = args.data_root / 'converted'
    rows = {s: [json.loads(x) for x in (data / (s + '.jsonl')).read_text(encoding='utf-8').splitlines()]
            for s in ('train', 'dev', 'test')}
    split = json.loads((data / 'split.json').read_text(encoding='utf-8'))
    regenerated = make_split([r for group in rows.values() for r in group], split['split_seed'])
    assert regenerated['document_ids'] == split['document_ids']
    for name in rows:
        assert [r['id'] for r in rows[name]] == split['document_ids'][name]
    for source in split['source_files']:
        assert sha(args.data_root / 'raw' / source['path']) == source['sha256']
    released, _ = published_records(args.data_root / 'raw')
    original = {r['meta']['id']: r for f in (args.data_root / 'raw').rglob('Annotator_P/prodigy_*.json')
                for r in json.loads(f.read_text(encoding='utf-8-sig'))}
    for group in rows.values():
        for row in group:
            reconstructed = convert_record(original[row['id']], released[row['id']])
            assert reconstructed == {k: v for k, v in row.items() if k != 'split'}
    feature_root = args.data_root / 'features_train_dev'
    manifest_path = feature_root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    dirs = [feature_root] + [(feature_root / p).resolve() for p in manifest['cache_directories']]
    for key, expected in manifest['feature_sha256'].items():
        path = next(p / (key + '.npy') for p in dirs if (p / (key + '.npy')).is_file())
        assert sha(path) == expected
    assert set(manifest['dataset_sha256']) == {'train', 'dev'}
    compatibility = {}
    for mode in ('act', 'arg'):
        assert manifest['examples'][mode]['test'] == []
        receipt = json.loads((args.run / mode / 'receipt.json').read_text(encoding='utf-8'))
        assert receipt['status'] == 'completed' and receipt['test'] is None and not receipt['test_evaluated']
        assert receipt['feature_manifest_sha256'] == sha(manifest_path)
        assert receipt['weights_changed'] and receipt['checkpoint_reload_predictions_equal']
        assert receipt['losses_and_gradients_finite']
        compatibility[mode] = json.loads((args.run / mode / 'trained_keras_compatibility.json').read_text(encoding='utf-8'))
        assert compatibility[mode]['passed'] and compatibility[mode]['num_q_outputs'] == 3
        assert compatibility[mode]['real_validation_contexts'] == 4
        assert compatibility[mode]['mode'] == mode
        train = manifest['examples'][mode]['train']
        assert receipt['unique_train_examples_visited'] == len(train)
        assert receipt['training_steps'] == receipt['epochs_completed'] * sum(len(x['tokens']) for x in train)
        assert all(h['complete'] for h in receipt['history'])
        assert receipt['best_epoch'] == max(receipt['history'], key=lambda x: x['dev']['f1'])['epoch']
        for path, expected in receipt['source_sha256'].items():
            assert sha(root / path) == expected, path
    final = args.run / 'final'
    report = json.loads((final / 'receipt.json').read_text(encoding='utf-8'))
    freeze = json.loads((final / 'freeze.json').read_text(encoding='utf-8'))
    assert report['status'] == 'completed' and not report['inference_uses_gold_labels']
    assert not report['test_evaluated_during_training']
    assert report['freeze_sha256'] == sha(final / 'freeze.json')
    for mode in ('act', 'arg'):
        assert sha(args.run / mode / 'best.pt') == report['checkpoint_sha256'][mode] == freeze['checkpoint_sha256'][mode]
        assert sha(args.run / mode / 'receipt.json') == freeze['training_receipt_sha256'][mode]
        feat = json.loads((final / (mode + '_features') / 'manifest.json').read_text(encoding='utf-8'))
        assert not feat['uses_gold_action_or_argument_labels']
        assert all('labels' not in x and 'gold_spans' not in x for x in feat['examples'][mode]['test'])
        assert sha(final / (mode + '_features') / 'manifest.json') == report['inference_feature_manifest_sha256'][mode]
    for name, expected in freeze['evaluation_source_sha256'].items():
        assert sha(root / 'scripts' / name) == expected
    assert freeze['frozen_at_utc'] < report['completed_at_utc']
    predictions = json.loads((final / 'predictions.json').read_text(encoding='utf-8'))
    metrics = independent_char_scores({r['id']: r for r in rows['test']}, predictions)
    assert metrics == report['metrics'] == report['independent_character_scoring']
    result = {'status': 'passed', 'split_reproduced': True, 'all_converted_records_reproduced_from_raw': True,
              'all_training_embedding_hashes_valid': True, 'train_dev_test_disjoint': True,
              'all_train_examples_completed_each_epoch': True, 'dev_only_checkpoint_selection': True,
              'test_absent_from_training_features': True, 'label_free_test_decoding': True,
              'source_and_checkpoint_hashes_match': True, 'independent_scores_match': True,
              'upstream_keras_forward_compatible': True, 'keras_compatibility': compatibility,
              'metrics': metrics, 'audit_script_sha256': sha(Path(__file__))}
    (final / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
