"""Score explicit action/argument pairs after propagating predicted actions."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from ceasdrl_torch import CnnDqn, Features, evaluate


def set_score(predicted, gold):
    right = len(predicted & gold)
    return {'precision': right / len(predicted) if predicted else 0.,
            'recall': right / len(gold) if gold else 0.,
            'f1': 2 * right / (len(predicted) + len(gold)) if predicted or gold else 0.,
            'right': right, 'predicted': len(predicted), 'gold': len(gold)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--argument-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Refusing to overwrite pipeline results')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    features = Features(args.features, 'arg', torch.device('cuda'))
    if features.manifest['uses_gold_action_or_argument_labels']:
        raise ValueError('Pipeline inputs contain gold-conditioned actions')
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != features.manifest['dataset_sha256']['test']:
        raise ValueError('Test dataset mismatch')
    model = CnnDqn('arg').cuda()
    checkpoint_path = args.argument_run / 'best.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    _, outputs = evaluate(model, features, features.examples['test'])
    records = {r['id']: r for r in [json.loads(x) for x in args.data.read_text(encoding='utf-8').splitlines()]}
    predicted_pairs, predicted_frames = set(), set()
    readable = []
    for ex, out in zip(features.examples['test'], outputs):
        doc_id, head = ex['doc_id'], ex['absolute_action_index']
        arguments = sorted(set(i + ex['window_start'] for i in out['predicted_indices']))
        predicted_pairs.update((doc_id, head, a) for a in arguments)
        predicted_frames.add((doc_id, head, tuple(arguments)))
        readable.append({'doc_id': doc_id, 'action_index': head, 'action_token': records[doc_id]['words'][head],
                         'argument_indices': arguments, 'argument_first_tokens': [records[doc_id]['words'][i] for i in arguments]})
    gold_pairs, gold_frames = set(), set()
    for doc_id, row in records.items():
        for action in row['actions']:
            head = action['trigger_tokens'][0]
            arguments = sorted(set(a['token_indices'][0] for a in action['explicit_arguments']))
            gold_pairs.update((doc_id, head, a) for a in arguments)
            gold_frames.add((doc_id, head, tuple(arguments)))
    receipt = {'status': 'completed', 'uses_predicted_actions': True,
               'argument_checkpoint_epoch': checkpoint.get('epoch'),
               'test_documents': len(records), 'predicted_actions': len(predicted_frames), 'gold_actions': len(gold_frames),
               'action_argument_pair': set_score(predicted_pairs, gold_pairs),
               'exact_action_argument_set': set_score(predicted_frames, gold_frames),
               'false_positive_pairs': [list(x) for x in sorted(predicted_pairs - gold_pairs)],
               'false_negative_pairs': [list(x) for x in sorted(gold_pairs - predicted_pairs)],
               'test_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
               'argument_checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
               'action_predictions_sha256': features.manifest['action_predictions_sha256'],
               'limitations': ['First-token projections, not exact multiword spans.',
                               'Role agnostic; implicit arguments and condition clauses excluded.',
                               'Missing action predictions contribute false negative argument pairs.',
                               'Argument inference excludes predicted action heads, not gold heads.',
                               'No ES/EX/OP categories.']}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    (args.output / 'predictions.json').write_text(json.dumps(readable, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in receipt.items() if not k.startswith('false_')}, indent=2))


if __name__ == '__main__':
    main()
