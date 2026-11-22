"""Export the original cached GloVe/ELMo embeddings using the legacy environment.

No model fitting: train/dev/test contextual representations are independent.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from ceasdrl_local import load_embeddings, make_examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import numpy as np
    import torch
    from flair.data import Sentence, Token
    torch.set_num_threads(4)
    embeddings = load_embeddings()
    args.output.mkdir(parents=True, exist_ok=True)
    contexts = {}
    manifest = {'embedding': 'GloVe-100 + ELMo-small-768', 'dimension': 868,
                'context': 'Whole example, exactly as ceasdrl_local.py',
                'dataset_sha256': {}, 'examples': {}, 'skipped': {}}
    for mode in ['act', 'arg']:
        manifest['examples'][mode], manifest['skipped'][mode] = {}, {}
        for split in ['train', 'dev', 'test']:
            path = args.data / (split + '.jsonl')
            manifest['dataset_sha256'][split] = hashlib.sha256(path.read_bytes()).hexdigest()
            records = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
            examples, skipped = make_examples(records, mode, 600 if mode == 'act' else 100)
            for example in examples:
                tokens = example['tokens']
                key = hashlib.sha256(json.dumps(tokens, ensure_ascii=False).encode('utf-8')).hexdigest()
                contexts[key] = tokens
                example['embedding_key'] = key
            manifest['examples'][mode][split] = examples
            manifest['skipped'][mode][split] = skipped
    start = time.time()
    for i, (key, tokens) in enumerate(contexts.items()):
        path = args.output / (key + '.npy')
        if path.exists():
            matrix = np.load(str(path))
            if matrix.shape != (len(tokens), 868) or not np.isfinite(matrix).all():
                raise ValueError('Invalid cached features: ' + str(path))
        else:
            sent = Sentence()
            for word in tokens:
                sent.add_token(Token(word))
            with torch.no_grad():
                embeddings.embed(sent)
            matrix = np.stack([t.embedding.detach().cpu().numpy() for t in sent]).astype(np.float32)
            if matrix.shape != (len(tokens), 868) or not np.isfinite(matrix).all():
                raise ValueError('Invalid embedding shape/values')
            np.save(str(path), matrix)
        if (i + 1) % 50 == 0:
            print('Embeddings {}/{} elapsed {:.1f}s'.format(i + 1, len(contexts), time.time() - start), flush=True)
    manifest['unique_contexts'] = len(contexts)
    manifest['elapsed_seconds'] = time.time() - start
    manifest['feature_sha256'] = {k: hashlib.sha256((args.output / (k + '.npy')).read_bytes()).hexdigest() for k in contexts}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('FEATURE_EXPORT_COMPLETE {} contexts {:.1f}s'.format(len(contexts), time.time() - start), flush=True)


if __name__ == '__main__':
    main()
