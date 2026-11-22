"""Freeze both dev-selected checkpoints, then evaluate a sealed text-only holdout."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

try:
    from .evaluate_ceasdrl_spans import assemble_predictions, evaluate_predictions
    from .prepare_procedural_data import tokenize
    from .risec_spans import embedding_key, predicted_span_contexts, count_scores
except ImportError:
    from evaluate_ceasdrl_spans import assemble_predictions, evaluate_predictions
    from prepare_procedural_data import tokenize
    from risec_spans import embedding_key, predicted_span_contexts, count_scores


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def independent_char_scores(records, predictions):
    """Independent scoring from source character offsets, including missed actions."""
    gold_actions, gold_arguments, pred_actions, pred_arguments = set(), set(), set(), set()
    for doc_id, row in records.items():
        for action in row['actions']:
            key = (doc_id, *action['char_span'])
            assert row['text'][key[1]:key[2]] == action['text']
            gold_actions.add(key)
            for arg in action['explicit_arguments']:
                assert row['text'][arg['char_span'][0]:arg['char_span'][1]] == arg['target_text']
                gold_arguments.add(key + tuple(arg['char_span']))
    if len(predictions) != len(records) or {p['doc_id'] for p in predictions} != set(records):
        raise ValueError('Prediction document mismatch')
    for row in predictions:
        text = records[row['doc_id']]['text']
        for action in row['actions']:
            key = (row['doc_id'], *action['char_span'])
            if text[key[1]:key[2]] != action['name']:
                raise ValueError('Predicted action name does not match source occurrence')
            pred_actions.add(key)
            for arg in action['arguments']:
                if text[arg['char_span'][0]:arg['char_span'][1]] != arg['name']:
                    raise ValueError('Predicted argument name does not match source occurrence')
                pred_arguments.add(key + tuple(arg['char_span']))
    return {'action_name': count_scores(len(gold_actions & pred_actions), len(pred_actions), len(gold_actions)),
            'linked_argument_name': count_scores(len(gold_arguments & pred_arguments), len(pred_arguments), len(gold_arguments))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--action-run', type=Path, required=True)
    p.add_argument('--argument-run', type=Path, required=True)
    p.add_argument('--legacy-python', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if (args.output / 'receipt.json').exists():
        raise ValueError('Refusing to repeat completed test evaluation')
    root = Path(__file__).resolve().parents[1]
    split = json.loads((args.data / 'split.json').read_text(encoding='utf-8'))
    data_hashes = {s: sha(args.data / (s + '.jsonl')) for s in ('train', 'dev', 'test')}
    audit = json.loads((args.data / 'conversion_audit.json').read_text(encoding='utf-8'))
    if audit['status'] != 'passed' or audit['dataset_sha256'] != data_hashes:
        raise ValueError('Dataset conversion audit mismatch')
    for name, expected in audit['source_sha256'].items():
        if sha(root / 'scripts' / name) != expected:
            raise ValueError('Conversion source changed: ' + name)
    manifest_path = args.features / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['dataset_sha256'] != {s: data_hashes[s] for s in ('train', 'dev')}:
        raise ValueError('Training feature data mismatch')
    if any(s['test'] for s in manifest['examples'].values()):
        raise ValueError('Training features include the test set')
    stage_paths = {'act': args.action_run, 'arg': args.argument_run}
    stages, checkpoint_hashes = {}, {}
    for mode, path in stage_paths.items():
        receipt = json.loads((path / 'receipt.json').read_text(encoding='utf-8'))
        if receipt['status'] != 'completed' or receipt['test_evaluated'] or receipt['test'] is not None:
            raise ValueError('Training incomplete or test already evaluated during training')
        if receipt['feature_manifest_sha256'] != sha(manifest_path):
            raise ValueError('Training manifest changed')
        if receipt['configuration']['mode'] != mode or not receipt['configuration']['span_labels']:
            raise ValueError('Wrong model stage or output labels')
        if not all(h['complete'] for h in receipt['history']):
            raise ValueError('Partial training epochs are not an experiment')
        best = max(receipt['history'], key=lambda h: h['dev']['f1'])
        if receipt['best_epoch'] != best['epoch']:
            raise ValueError('Checkpoint not chosen by validation F1')
        for name, expected in receipt['source_sha256'].items():
            if sha(root / name) != expected:
                raise ValueError('Training source changed: ' + name)
        checkpoint_hashes[mode] = sha(path / 'best.pt')
        stages[mode] = receipt
    freeze = {'checkpoint_sha256': checkpoint_hashes, 'dataset_sha256': data_hashes,
              'split_sha256': sha(args.data / 'split.json'),
              'training_receipt_sha256': {m: sha(p / 'receipt.json') for m, p in stage_paths.items()},
              'best_epochs': {m: r['best_epoch'] for m, r in stages.items()},
              'evaluation_source_sha256': {n: sha(root / 'scripts' / n) for n in (
                  'evaluate_ceasdrl_holdout.py', 'evaluate_ceasdrl_spans.py', 'predict_ceasdrl_text.py',
                  'risec_spans.py', 'ceasdrl_spans.py', 'ceasdrl_torch.py', 'prepare_procedural_data.py')}}
    args.output.mkdir(parents=True, exist_ok=True)
    freeze_path = args.output / 'freeze.json'
    if freeze_path.exists():
        old = json.loads(freeze_path.read_text(encoding='utf-8'))
        if {k: v for k, v in old.items() if k != 'frozen_at_utc'} != freeze:
            raise ValueError('Frozen evaluation configuration changed')
        freeze = old
    else:
        freeze['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(freeze_path, freeze)
    print('CHECKPOINTS_FROZEN ' + json.dumps(checkpoint_hashes), flush=True)

    # Inference receives text-derived tokens only. Gold is used below solely to score.
    rows = [json.loads(x) for x in (args.data / 'test.jsonl').read_text(encoding='utf-8').splitlines()]
    records = {r['id']: r for r in rows}
    if len(records) != len(rows) or set(records) != set(split['document_ids']['test']):
        raise ValueError('Held-out document mismatch')
    plain, examples = {}, []
    for row in rows:
        tokens = tokenize(row['text'])
        words, offsets = [t[0] for t in tokens], [[t[1], t[2]] for t in tokens]
        if words != row['words'] or offsets != row['token_offsets']:
            raise ValueError('Inference tokenizer differs from training')
        plain[row['id']] = {'text': row['text'], 'words': words, 'token_offsets': offsets}
        examples.append({'id': row['id'], 'doc_id': row['id'], 'tokens': words, 'embedding_key': embedding_key(words)})
    import torch
    from ceasdrl_spans import SpanDqn, SpanFeatures, evaluate_spans
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    feature_hashes = {}
    for mode in ('act', 'arg'):
        device = torch.device(stages[mode]['configuration']['device'])
        output = (args.output / (mode + '_features')).resolve()
        caches = [args.features.resolve()]
        if mode == 'arg':
            caches.append((args.output / 'act_features').resolve())
        if any('labels' in e or 'gold_spans' in e for e in examples):
            raise ValueError('Gold labels entered inference')
        job = {'mode': mode, 'output': str(output), 'caches': [str(x) for x in caches], 'examples': examples}
        job_path = (args.output / (mode + '_embedding_job.json')).resolve()
        write_json(job_path, job)
        print('INFERENCE_FEATURES ' + mode + ' ' + str(len(examples)), flush=True)
        subprocess.run([str(args.legacy_python), str(root / 'scripts/predict_ceasdrl_text.py'), '--embed-job', str(job_path)], check=True)
        features = SpanFeatures(output, mode, device)
        if features.manifest['uses_gold_action_or_argument_labels']:
            raise ValueError('Gold labels in inference manifest')
        checkpoint = torch.load(stage_paths[mode] / 'best.pt', map_location=device, weights_only=False)
        if checkpoint['epoch'] != stages[mode]['best_epoch']:
            raise ValueError('Loaded checkpoint epoch mismatch')
        model = SpanDqn(mode).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        _, predictions = evaluate_spans(model, features, features.examples['test'])
        write_json(args.output / (mode + '_spans.json'), predictions)
        feature_hashes[mode] = sha(output / 'manifest.json')
        if mode == 'act':
            examples = predicted_span_contexts(plain, predictions)
        del model, features
        torch.cuda.empty_cache()
    predictions = assemble_predictions(plain, examples, predictions)
    metrics, errors = evaluate_predictions(records, predictions)
    independent = independent_char_scores(records, predictions)
    if metrics != independent:
        raise ValueError('Independent character-span scoring disagrees')
    if {m: sha(p / 'best.pt') for m,p in stage_paths.items()} != checkpoint_hashes:
        raise ValueError('Checkpoint changed during evaluation')
    report = {'status': 'completed', 'dataset': 'RiCoRecA',
              'claim': 'cEASDRL CNN/DQN BIO adaptation trained from scratch; not an original-paper score reproduction.',
              'protocol': 'document_holdout_64_16_20', 'cross_validation': False,
              'split_documents': split['counts'], 'split_seed': split['split_seed'], 'training_seed': 0,
              'metrics': metrics, 'independent_character_scoring': independent,
              'gold_scope': audit['gold_scope'], 'matching': 'Micro F1 of exact complete spans/names at document occurrences.',
              'argument_matching': 'The complete action occurrence and its linked complete argument occurrence must both match. Missed actions leave their arguments as false negatives.',
              'inference_uses_gold_labels': False, 'test_evaluated_during_training': False,
              'selection': 'Each stage independently selected by dev exact-span F1 before any test inference.',
              'best_epochs': freeze['best_epochs'], 'epochs_completed': {m:r['epochs_completed'] for m,r in stages.items()},
              'training_and_inference_devices': {m:r['configuration']['device'] for m,r in stages.items()},
              'dataset_sha256': data_hashes, 'checkpoint_sha256': checkpoint_hashes,
              'inference_feature_manifest_sha256': feature_hashes, 'freeze_sha256': sha(freeze_path),
              'completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'limitations': ['Single fixed split and training seed; no cross-validation mean or standard deviation.',
                             'Task projection excludes control predicates, their exclusive members, and implicit entities.',
                             'BIO boundaries and balanced replay adapt the original binary cEASDRL method.',
                             'Scores across RISeC and RiCoRecA are not a controlled dataset-quality comparison.']}
    for name, value in [('predictions', predictions), ('errors', errors), ('receipt', report)]:
        write_json(args.output / (name + '.json'), value)
    print('COMPLETED ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
