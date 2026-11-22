"""Predict complete actions/arguments from plain text using the frozen holdout models."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--text-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--legacy-python', type=Path, required=True)
    p.add_argument('--embedding-cache', type=Path, default=Path('data/procedural/ricoreca/features_train_dev'))
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing prediction directory')
    from prepare_procedural_data import tokenize
    from risec_spans import embedding_key, predicted_span_contexts
    from evaluate_ceasdrl_spans import assemble_predictions
    from ceasdrl_spans import SpanFeatures, SpanDqn, evaluate_spans
    import torch
    receipt = json.loads((args.run/'final/receipt.json').read_text(encoding='utf-8'))
    if receipt['status'] != 'completed':
        raise ValueError('Use a completed, frozen experiment')
    with args.text_file.open(encoding='utf-8', newline='') as f:
        text = f.read()
    tokens = tokenize(text)
    if not 0 < len(tokens) <= 600:
        raise ValueError('Input must contain 1 to 600 tokens')
    doc_id = args.text_file.stem
    plain = {doc_id: {'text':text,'words':[t[0] for t in tokens], 'token_offsets':[[t[1],t[2]] for t in tokens]}}
    words = plain[doc_id]['words']
    examples = [{'id':doc_id,'doc_id':doc_id,'tokens':words,'embedding_key':embedding_key(words)}]
    args.output.mkdir(parents=True)
    def write(name, obj):
        path = args.output/name
        path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
        return path.resolve()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    hashes, devices = {}, {}
    for mode in ('act','arg'):
        path = args.run/mode/'best.pt'
        hashes[mode] = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[mode] != receipt['checkpoint_sha256'][mode]:
            raise ValueError('Frozen checkpoint hash mismatch')
        checkpoint = torch.load(path,map_location='cpu',weights_only=False)
        devices[mode] = checkpoint['configuration']['device']
        device = torch.device(devices[mode])
        output = (args.output/(mode+'_features')).resolve()
        caches = [args.embedding_cache.resolve(),(args.run/'final/act_features').resolve(),(args.run/'final/arg_features').resolve()]
        if mode == 'arg':
            caches.append((args.output/'act_features').resolve())
        job = write(mode+'_embedding_job.json', {'mode':mode,'output':str(output),'caches':[str(x) for x in caches],'examples':examples})
        subprocess.run([str(args.legacy_python),str(Path(__file__).with_name('predict_ceasdrl_text.py')),'--embed-job',str(job)],check=True)
        features = SpanFeatures(output,mode,device)
        model = SpanDqn(mode).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        _, predictions = evaluate_spans(model,features,features.examples['test'])
        if mode == 'act':
            write('action_spans.json',predictions)
            examples = predicted_span_contexts(plain,predictions)
        del features,model,checkpoint
    output = assemble_predictions(plain,examples,predictions)
    write('predictions.json',output)
    write('receipt.json',{'status':'completed','input_sha256':hashlib.sha256(args.text_file.read_bytes()).hexdigest(),
                         'checkpoint_sha256':hashes,'devices':devices,'inference_uses_gold_labels':False,
                         'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
