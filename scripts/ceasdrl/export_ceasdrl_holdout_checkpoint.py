"""Export frozen holdout checkpoints and CPU validation probes to original Keras layout."""
import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
from ceasdrl_torch import CnnDqn, Features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    args = p.parse_args()
    config = json.loads((args.run / 'config.json').read_text())
    mode = config['mode']
    model_class, feature_class = CnnDqn, Features
    span_labels = config.get('span_labels', False)
    if span_labels:
        from ceasdrl_spans import SpanDqn, SpanFeatures
        model_class, feature_class = SpanDqn, SpanFeatures
    root = Path(__file__).resolve().parents[2]
    source = root / 'data/ceasdrl/weights' / ('cooking_' + mode + '_elmo.h5')
    destination = args.run / 'best.keras.weights.h5'
    if destination.exists():
        raise ValueError('Refusing to overwrite exported weights')
    state = torch.load(args.run / 'best.pt', map_location='cpu', weights_only=False)['state_dict']
    model = model_class(mode)
    model.load_state_dict(state)
    values = []
    for conv in model.convs:
        values.extend([conv.weight.detach().numpy().transpose(2, 1, 0)[:, :, None, :], conv.bias.detach().numpy()])
    for layer in [model.dense, model.output]:
        values.extend([layer.weight.detach().numpy().T, layer.bias.detach().numpy()])
    shutil.copyfile(source, destination)
    with h5py.File(destination, 'r+') as f:
        index = 0
        for raw_name in f.attrs['layer_names']:
            name = raw_name.decode() if isinstance(raw_name, bytes) else raw_name
            group = f[name]
            for raw_weight in group.attrs.get('weight_names', []):
                key = raw_weight.decode() if isinstance(raw_weight, bytes) else raw_weight
                if group[key].shape != values[index].shape:
                    if not span_labels or index not in (10, 11):
                        raise ValueError('Weight shape mismatch')
                    # Only the documented BIO output head changes size, 2 -> 3.
                    del group[key]
                    group.create_dataset(key, data=values[index])
                else:
                    group[key][...] = values[index]
                index += 1
        assert index == len(values) == 12
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    features = feature_class(args.features, mode, torch.device('cpu'))
    examples = features.examples['dev'][:4]
    words, distance = features.batch(examples)
    tags = torch.zeros((4, features.num_words), device='cpu')
    for i in range(4):
        if span_labels:
            tags[i, :i*3] = torch.arange(i*3, device='cpu') % 3 + 1
        else:
            tags[i, :i*3] = 1
    model.cpu().eval()
    with torch.no_grad():
        expected = model(words, tags, distance).cpu().numpy()
    np.savez_compressed(args.run / 'trained_keras_probe.npz', words=words.cpu().numpy(),
                        distance=distance.cpu().numpy(), tags=tags.cpu().numpy(), q=expected)
    print(json.dumps({'exported': str(destination), 'parameters': index, 'mode': mode}))


if __name__ == '__main__':
    main()
