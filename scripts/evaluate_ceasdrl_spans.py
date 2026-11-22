"""Evaluate the complete text -> predicted actions -> linked arguments pipeline."""
import argparse
import hashlib
import json
from pathlib import Path

try:
    from .risec_spans import span_name, count_scores, token_span
except ImportError:
    from risec_spans import span_name, count_scores, token_span


def named_span(record, span):
    return {'name': span_name(record, span), 'token_span': list(span),
            'char_span': [record['token_offsets'][span[0]][0], record['token_offsets'][span[1] - 1][1]]}


def assemble_predictions(records, contexts, argument_predictions):
    if len(contexts) != len(argument_predictions):
        raise ValueError('Argument context/prediction count mismatch')
    output = {doc_id: {'doc_id': doc_id, 'text': row['text'], 'actions': []} for doc_id, row in records.items()}
    for ex, pred in zip(contexts, argument_predictions):
        if ex['id'] != pred['id'] or ex['doc_id'] != pred['doc_id']:
            raise ValueError('Argument context/prediction identity mismatch')
        row = records[ex['doc_id']]
        action = named_span(row, ex['action_span'])
        action['arguments'] = [named_span(row, [s + ex['window_start'], e + ex['window_start']])
                               for s, e in pred['predicted_spans']]
        output[ex['doc_id']]['actions'].append(action)
    return list(output.values())


def name_key(doc_id, span, name):
    return (doc_id, span[0], span[1], ' '.join(name.split()).casefold())


def evaluate_predictions(records, predictions):
    if len(predictions) != len(records) or {r['doc_id'] for r in predictions} != set(records):
        raise ValueError('Evaluation document set mismatch')
    gold_acts, gold_args, pred_acts, pred_args = set(), set(), set(), set()
    for doc_id, row in records.items():
        for act in row['actions']:
            key = name_key(doc_id, token_span(act['trigger_tokens']), act['text'])
            gold_acts.add(key)
            for arg in act['explicit_arguments']:
                gold_args.add(key + name_key(doc_id, token_span(arg['token_indices']), arg['target_text'])[1:])
    for row in predictions:
        for act in row['actions']:
            key = name_key(row['doc_id'], act['token_span'], act['name'])
            pred_acts.add(key)
            for arg in act['arguments']:
                pred_args.add(key + name_key(row['doc_id'], arg['token_span'], arg['name'])[1:])
    metrics = {'action_name': count_scores(len(gold_acts & pred_acts), len(pred_acts), len(gold_acts)),
               'linked_argument_name': count_scores(len(gold_args & pred_args), len(pred_args), len(gold_args))}
    errors = {'action_false_positives': sorted(pred_acts - gold_acts), 'action_false_negatives': sorted(gold_acts - pred_acts),
              'argument_false_positives': sorted(pred_args - gold_args), 'argument_false_negatives': sorted(gold_args - pred_args)}
    return metrics, errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--action-run', type=Path, required=True)
    p.add_argument('--argument-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    from ceasdrl_spans import SpanFeatures, SpanDqn, evaluate_spans
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Refusing to overwrite pipeline evaluation')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    rows = [json.loads(x) for x in args.data.read_text(encoding='utf-8').splitlines()]
    records = {r['id']: r for r in rows}
    features = SpanFeatures(args.features, 'arg', device)
    manifest = features.manifest
    data_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    action_prediction_path = args.action_run / 'test_predictions.json'
    if manifest['uses_gold_action_or_argument_labels'] or manifest['dataset_sha256']['test'] != data_hash:
        raise ValueError('Inference features are not label-free or use different data')
    if manifest['action_predictions_sha256'] != hashlib.sha256(action_prediction_path.read_bytes()).hexdigest():
        raise ValueError('Inference actions differ from selected action model output')
    stage_receipts = {mode: json.loads((path / 'receipt.json').read_text(encoding='utf-8'))
                      for mode, path in [('act', args.action_run), ('arg', args.argument_run)]}
    for receipt in stage_receipts.values():
        if receipt['status'] != 'completed' or receipt['dataset_sha256']['test'] != data_hash or not receipt['configuration']['span_labels']:
            raise ValueError('Training incomplete, labels wrong, or test data differs')
    model = SpanDqn('arg').to(device)
    checkpoint = torch.load(args.argument_run / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    # No reference labels are present in these examples; diagnostic scores unused.
    _, argument_predictions = evaluate_spans(model, features, features.examples['test'])
    predictions = assemble_predictions(records, features.examples['test'], argument_predictions)
    metrics, errors = evaluate_predictions(records, predictions)
    if metrics['action_name'] != stage_receipts['act']['test']:
        raise ValueError('Action names and action span scores disagree')
    report = {'status': 'completed', 'claim': 'cEASDRL adapted with BIO span decisions; not original paper reproduction.',
              'metrics': metrics, 'test_documents': len(records), 'split_documents': {'train': 126, 'dev': 56, 'test': 78},
              'matching': 'Micro F1; complete normalized name and exact occurrence boundaries.',
              'argument_matching': 'Both action and linked argument must match; missing actions retain all their gold arguments as false negatives.',
              'gold_scope': 'Explicit argument links in canonical RISeC conversion; no roles or ES/EX/OP; implicit and condition-clause links excluded.',
              'inference_uses_gold_labels': False, 'gold_name_expansion': False,
              'best_epochs': {mode: r['best_epoch'] for mode, r in stage_receipts.items()},
              'completed_epochs': {mode: r['epochs_completed'] for mode, r in stage_receipts.items()},
              'selection': 'Each stage selected by official dev exact-span F1; test never used for checkpoint selection.',
              'dataset_sha256': data_hash,
              'checkpoint_sha256': {mode: hashlib.sha256((path / 'best.pt').read_bytes()).hexdigest()
                                    for mode, path in [('act', args.action_run), ('arg', args.argument_run)]},
              'feature_manifest_sha256': hashlib.sha256((args.features / 'manifest.json').read_bytes()).hexdigest(),
              'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in ['evaluate_ceasdrl_spans.py', 'risec_spans.py', 'ceasdrl_spans.py']}}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [('receipt', report), ('predictions', predictions), ('errors', errors)]:
        (args.output / (name + '.json')).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
