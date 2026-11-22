"""Cache frozen GloVe/ELMo features by document, using the legacy Flair runtime."""
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

from ceasdrl_local import ROOT, load_embeddings


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import numpy as np
    import torch
    from flair.data import Sentence, Token
    torch.set_num_threads(4)
    emb = load_embeddings()
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'data/easdrl/indices/ceasdrl_two_source_v1.json').read_text())
    for domain, spec in manifest['domains'].items():
        source = (ROOT / spec['source']).read_bytes()
        assert hashlib.sha256(source).hexdigest() == spec['sha256']
        docs = pickle.loads(source, encoding='latin1')
        for i, doc in enumerate(docs):
            path = output / ('%s_%d.npz' % (domain, i))
            if path.exists():
                continue
            matrices = {}
            def embed(tokens):
                sentence = Sentence()
                for token in tokens:
                    sentence.add_token(Token(token))
                with torch.no_grad():
                    emb.embed(sentence)
                matrix = np.stack([t.embedding.detach().cpu().numpy() for t in sentence])
                assert matrix.shape == (len(tokens), 868) and np.isfinite(matrix).all()
                return matrix
            sentences = doc['sents']
            assert sum(sentences, []) == doc['words']
            matrices['act'] = np.concatenate([embed(s) for s in sentences if s])
            for j, sentence in enumerate(sentences):
                if sentence:
                    matrices['arg_%d' % j] = embed((sentences[j-1] if j else []) + sentence)
            temp = path.with_suffix('.tmp.npz')
            np.savez(str(temp), **matrices)
            temp.replace(path)
            if (i+1) % 20 == 0:
                print(domain, i+1, '/', len(docs), flush=True)
    print('FEATURES_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
