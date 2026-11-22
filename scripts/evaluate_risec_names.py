"""Evaluate actual predicted names against complete RISeC names, without gold expansion.

Names are casefolded and whitespace-normalized. Occurrence positions distinguish
repeated names, and each argument must belong to the correctly predicted action.
"""
import argparse
import hashlib
import json
from pathlib import Path


def normalize_name(value):
    return ' '.join(value.split()).casefold()


def score_sets(predicted, gold):
    right = len(predicted & gold)
    return {'precision': right / len(predicted) if predicted else 0.,
            'recall': right / len(gold) if gold else 0.,
            'f1': 2 * right / (len(predicted) + len(gold)) if predicted or gold else 0.,
            'right': right, 'predicted': len(predicted), 'gold': len(gold)}


def evaluate_names(records, predictions):
    gold_actions, gold_links, predicted_actions, predicted_links = set(), set(), set(), set()
    for row in records:
        for action in row['actions']:
            key = (row['id'], action['trigger_tokens'][0], normalize_name(action['text']))
            gold_actions.add(key)
            for argument in action['explicit_arguments']:
                gold_links.add(key + (argument['token_indices'][0], normalize_name(argument['target_text'])))
    for row in predictions:
        key = (row['doc_id'], row['action_index'], normalize_name(row['action_token']))
        predicted_actions.add(key)
        if len(row['argument_indices']) != len(row['argument_first_tokens']):
            raise ValueError('Argument names and positions have different lengths')
        for position, name in zip(row['argument_indices'], row['argument_first_tokens']):
            predicted_links.add(key + (position, normalize_name(name)))
    return {'action_name': score_sets(predicted_actions, gold_actions),
            'linked_argument_name': score_sets(predicted_links, gold_links)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(x) for x in args.data.read_text(encoding='utf-8').splitlines()]
    predictions = json.loads(args.predictions.read_text(encoding='utf-8'))
    report = {'status': 'completed', 'metrics': evaluate_names(records, predictions),
              'matching': 'Complete name, casefold and whitespace normalization; occurrence-aware exact matching.',
              'argument_matching': 'Correct action name AND correct linked argument name; all gold links are in recall denominator.',
              'prediction_scope': 'Existing checkpoints output one token per action/argument; names are NOT expanded using gold spans.',
              'gold_scope': 'Full annotated action names and explicit argument names; no ES/EX/OP or role categories.',
              'argument_training': 'Uses best checkpoint from three completed epochs; training interrupted during epoch four.',
              'dataset_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
              'prediction_sha256': hashlib.sha256(args.predictions.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('Refusing to overwrite name-evaluation receipt')
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
