"""Legacy py36: embed argument contexts around PREDICTED test actions only.

No gold action/argument labels are used to construct inference contexts.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

def predicted_contexts(records, predictions):
    if set(records) != set(r['doc_id'] for r in predictions):
        raise ValueError('Action prediction document set mismatch')
    examples, contexts = [], {}
    for row in predictions:
        doc_id = row['doc_id']
        words = records[doc_id]['words']
        heads = row['predicted_indices']
        if len(set(heads)) != len(heads) or any(not 0 <= h < len(words) for h in heads):
            raise ValueError('Invalid predicted action locations')
        for head in heads:
            start = max(0, min(head - 50, len(words) - 100))
            stop = min(len(words), start + 100)
            tokens = words[start:stop]
            key = hashlib.sha256(json.dumps(tokens, ensure_ascii=False).encode('utf-8')).hexdigest()
            contexts[key] = tokens
            examples.append({'id': '{}:pred{}'.format(doc_id, head), 'doc_id': doc_id,
                             'tokens': tokens, 'positive_indices': [], 'embedding_key': key,
                             'act_ind': head - start, 'act_inds': [h - start for h in heads if start <= h < stop],
                             'absolute_action_index': head, 'window_start': start})
    return examples, contexts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--action-predictions', type=Path, required=True)
    p.add_argument('--existing-features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    from ceasdrl_local import load_embeddings
    records = {r['id']: r for r in [json.loads(x) for x in args.data.read_text(encoding='utf-8').splitlines()]}
    predictions = json.loads(args.action_predictions.read_text(encoding='utf-8'))
    examples, contexts = predicted_contexts(records, predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    embeddings, exported = None, 0
    for key, tokens in contexts.items():
        destination = args.output / (key + '.npy')
        existing = args.existing_features / (key + '.npy')
        if existing.exists():
            shutil.copyfile(str(existing), str(destination))
        else:
            if embeddings is None:
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                import torch
                from flair.data import Sentence, Token
                torch.set_num_threads(4)
                embeddings = load_embeddings()
            sent = Sentence()
            for word in tokens:
                sent.add_token(Token(word))
            with torch.no_grad():
                embeddings.embed(sent)
            array = np.stack([t.embedding.detach().cpu().numpy() for t in sent]).astype(np.float32)
            np.save(str(destination), array)
            exported += 1
        matrix = np.load(str(destination))
        if matrix.shape != (len(tokens), 868) or not np.isfinite(matrix).all():
            raise ValueError('Invalid prediction context embeddings')
    manifest = {'dimension': 868, 'embedding': 'GloVe-100 + ELMo-small-768',
                'examples': {'arg': {'train': [], 'dev': [], 'test': examples}},
                'dataset_sha256': {'test': hashlib.sha256(args.data.read_bytes()).hexdigest()},
                'action_predictions_sha256': hashlib.sha256(args.action_predictions.read_bytes()).hexdigest(),
                'new_embedding_contexts': exported, 'unique_contexts': len(contexts),
                'uses_gold_action_or_argument_labels': False,
                'skipped': {'arg': {'train': [], 'dev': [], 'test': []}}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'predicted_actions': len(examples), 'contexts': len(contexts), 'new_contexts': exported}))


if __name__ == '__main__':
    main()
