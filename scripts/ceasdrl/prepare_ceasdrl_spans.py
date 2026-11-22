"""Prepare full BIO training targets or predicted-action inference contexts.

Run with legacy py36 when new ELMo contexts must be computed. Existing frozen
embeddings are reused by token-content hash; labels are regenerated from spans.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from risec_spans import make_span_examples, predicted_span_contexts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--existing-features', type=Path, nargs='+', required=True)
    p.add_argument('--action-predictions', type=Path)
    args = p.parse_args()
    import numpy as np
    if (args.output / 'manifest.json').exists():
        raise ValueError('Refusing to overwrite feature manifest')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'label_scheme': 'BIO', 'labels': {'O': 0, 'B': 1, 'I': 2},
                'embedding': 'GloVe-100 + ELMo-small-768', 'dimension': 868,
                'dataset_sha256': {}, 'examples': {}, 'skipped': {},
                'cache_directories': [os.path.relpath(str(p.resolve()), str(args.output.resolve())) for p in args.existing_features],
                'uses_gold_action_or_argument_labels': args.action_predictions is None}
    if args.action_predictions:
        rows = [json.loads(x) for x in args.data.read_text(encoding='utf-8').splitlines()]
        predictions = json.loads(args.action_predictions.read_text(encoding='utf-8'))
        examples = predicted_span_contexts({r['id']: r for r in rows}, predictions)
        manifest['examples'] = {'arg': {'train': [], 'dev': [], 'test': examples}}
        manifest['skipped'] = {'arg': {'train': [], 'dev': [], 'test': []}}
        manifest['dataset_sha256']['test'] = hashlib.sha256(args.data.read_bytes()).hexdigest()
        manifest['action_predictions_sha256'] = hashlib.sha256(args.action_predictions.read_bytes()).hexdigest()
    else:
        for mode in ['act', 'arg']:
            manifest['examples'][mode], manifest['skipped'][mode] = {}, {}
            for split in ['train', 'dev', 'test']:
                path = args.data / (split + '.jsonl')
                rows = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
                manifest['examples'][mode][split] = make_span_examples(rows, mode)
                manifest['skipped'][mode][split] = []
                manifest['dataset_sha256'][split] = hashlib.sha256(path.read_bytes()).hexdigest()
    contexts = {ex['embedding_key']: ex['tokens'] for splits in manifest['examples'].values() for group in splits.values() for ex in group}
    embeddings, exported, hashes = None, 0, {}
    for key, tokens in contexts.items():
        files = [p / (key + '.npy') for p in [args.output] + args.existing_features]
        filename = next((p for p in files if p.is_file()), None)
        if filename is None:
            if embeddings is None:
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                import torch
                from flair.data import Sentence, Token
                from ceasdrl_local import load_embeddings
                torch.set_num_threads(4)
                embeddings = load_embeddings()
            sentence = Sentence()
            for word in tokens:
                sentence.add_token(Token(word))
            with torch.no_grad():
                embeddings.embed(sentence)
            matrix = np.stack([t.embedding.detach().cpu().numpy() for t in sentence]).astype(np.float32)
            filename = args.output / (key + '.npy')
            np.save(str(filename), matrix)
            exported += 1
        matrix = np.load(str(filename))
        if matrix.shape != (len(tokens), 868) or not np.isfinite(matrix).all():
            raise ValueError('Invalid contextual embeddings')
        hashes[key] = hashlib.sha256(filename.read_bytes()).hexdigest()
    manifest.update({'feature_sha256': hashes, 'unique_contexts': len(contexts), 'new_embedding_contexts': exported})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'contexts': len(contexts), 'new_embeddings': exported,
                      'examples': {m: {s: len(ex) for s, ex in splits.items()} for m, splits in manifest['examples'].items()}}))


if __name__ == '__main__':
    main()
