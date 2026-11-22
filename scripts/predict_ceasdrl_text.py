"""Extract complete actions and arguments from an unannotated UTF-8 text file.

The main environment runs the trained CNNs. A legacy Python subprocess computes
missing frozen GloVe/ELMo vectors offline. No annotations or API calls are needed.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def embed_job(path):
    """Private subprocess entry point compatible with the cached py36 embedder."""
    import numpy as np
    job = json.loads(path.read_text(encoding='utf-8'))
    output = Path(job['output'])
    output.mkdir(parents=True, exist_ok=True)
    caches = [Path(p) for p in job['caches']]
    examples = job['examples']
    embeddings, hashes, new = None, {}, 0
    for ex in examples:
        key, tokens = ex['embedding_key'], ex['tokens']
        if key in hashes:
            continue
        filename = next((p / (key + '.npy') for p in [output] + caches if (p / (key + '.npy')).is_file()), None)
        if filename is None:
            if embeddings is None:
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                import torch
                from flair.data import Sentence, Token
                from ceasdrl_local import load_embeddings
                torch.set_num_threads(4)
                embeddings = load_embeddings()
            sentence = Sentence()
            for token in tokens:
                sentence.add_token(Token(token))
            with torch.no_grad():
                embeddings.embed(sentence)
            matrix = np.stack([t.embedding.detach().cpu().numpy() for t in sentence]).astype(np.float32)
            filename = output / (key + '.npy')
            np.save(str(filename), matrix)
            new += 1
        matrix = np.load(str(filename))
        if matrix.shape != (len(tokens), 868) or not np.isfinite(matrix).all():
            raise ValueError('Invalid contextual embedding')
        hashes[key] = hashlib.sha256(filename.read_bytes()).hexdigest()
    write_json(output / 'manifest.json', {'label_scheme': 'BIO', 'dimension': 868,
               'embedding': 'GloVe-100 + ELMo-small-768', 'uses_gold_action_or_argument_labels': False,
               'cache_directories': [os.path.relpath(str(p), str(output)) for p in caches],
               'examples': {job['mode']: {'test': examples}}, 'feature_sha256': hashes,
               'new_embedding_contexts': new})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--text-file', type=Path, required=True)
    p.add_argument('--action-checkpoint', type=Path, required=True)
    p.add_argument('--argument-checkpoint', type=Path, required=True)
    p.add_argument('--legacy-python', type=Path, required=True)
    p.add_argument('--embedding-cache', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Refusing to overwrite prediction output')
    if not args.legacy_python.is_file():
        raise FileNotFoundError('Legacy embedding environment does not exist')
    from prepare_procedural_data import tokenize
    from risec_spans import embedding_key, predicted_span_contexts
    from evaluate_ceasdrl_spans import assemble_predictions
    import torch
    from ceasdrl_spans import SpanFeatures, SpanDqn, evaluate_spans
    with args.text_file.open(encoding='utf-8', newline='') as f:
        text = f.read()
    tokens = tokenize(text)
    if not 0 < len(tokens) <= 600:
        raise ValueError('This trained model supports 1 to 600 tokens per document')
    doc_id = args.text_file.stem
    record = {'text': text, 'words': [t[0] for t in tokens], 'token_offsets': [[t[1], t[2]] for t in tokens]}
    examples = [{'id': doc_id, 'doc_id': doc_id, 'tokens': record['words'], 'embedding_key': embedding_key(record['words'])}]
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions = None
    hashes = {}
    for mode, checkpoint_path in [('act', args.action_checkpoint), ('arg', args.argument_checkpoint)]:
        directory = (args.output / (mode + '_features')).resolve()
        caches = [p.resolve() for p in args.embedding_cache]
        if mode == 'arg':
            caches.append((args.output / 'act_features').resolve())
        job_path = (args.output / (mode + '_embedding_job.json')).resolve()
        write_json(job_path, {'mode': mode, 'output': str(directory), 'caches': [str(p) for p in caches], 'examples': examples})
        subprocess.run([str(args.legacy_python), str(Path(__file__).resolve()), '--embed-job', str(job_path)], check=True)
        features = SpanFeatures(directory, mode, device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if not checkpoint['configuration'].get('span_labels') or checkpoint['configuration']['mode'] != mode:
            raise ValueError('Expected trained BIO checkpoint for ' + mode)
        model = SpanDqn(mode).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        _, predictions = evaluate_spans(model, features, features.examples['test'])
        hashes[mode] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if mode == 'act':
            write_json(args.output / 'action_spans.json', predictions)
            examples = predicted_span_contexts({doc_id: record}, predictions)
        del model, features
    result = assemble_predictions({doc_id: record}, examples, predictions)
    write_json(args.output / 'predictions.json', result)
    write_json(args.output / 'receipt.json', {'status': 'completed', 'inference_uses_gold_labels': False,
                'input_sha256': hashlib.sha256(args.text_file.read_bytes()).hexdigest(), 'checkpoint_sha256': hashes,
                'token_count': len(tokens), 'device': str(device)})
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--embed-job':
        embed_job(Path(sys.argv[2]))
    else:
        main()
