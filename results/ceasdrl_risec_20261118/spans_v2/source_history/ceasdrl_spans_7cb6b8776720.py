"""cEASDRL CNN/DQN with three BIO decisions for complete-name extraction."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from .ceasdrl_torch import CnnDqn, Features
    from .risec_spans import span_scores
except ImportError:
    from ceasdrl_torch import CnnDqn, Features
    from risec_spans import span_scores


class SpanDqn(CnnDqn):
    def __init__(self, mode, dim=868, filters=32, hidden=256):
        super().__init__(mode, dim, filters, hidden)
        self.output = nn.Linear(hidden, 3)
        nn.init.trunc_normal_(self.output.weight, std=.05, a=-.1, b=.1)
        nn.init.zeros_(self.output.bias)


def span_rewards(actions, labels, positive_count, num_words, terminal):
    # Same magnitudes, shaping and terminal bonus as binary cEASDRL.
    # Correct positive means the exact B or I decision, not merely non-O.
    base = np.where(labels != 0, 100., 50.)
    reward = np.where(actions == labels, base, -base)
    rate = positive_count / float(num_words)
    shaping = 250. * np.square(rate)
    reward += np.where((rate <= .1) & (reward > 0), shaping, -shaping)
    return (reward + 2. * terminal).astype(np.float32)


class SpanFeatures(Features):
    def __init__(self, path, mode, device):
        self.path, self.mode, self.device = Path(path), mode, device
        self.manifest = json.loads((self.path / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest['label_scheme'] != 'BIO':
            raise ValueError('Expected full-span BIO features')
        self.examples = self.manifest['examples'][mode]
        self.num_words = 600 if mode == 'act' else 100
        self.cache, self.all_examples = {}, []
        directories = [self.path] + [(self.path / p).resolve() for p in self.manifest.get('cache_directories', [])]
        for items in self.examples.values():
            for ex in items:
                ex['index'] = len(self.all_examples)
                self.all_examples.append(ex)
                key = ex['embedding_key']
                if key not in self.cache:
                    candidates = [p / (key + '.npy') for p in directories]
                    filename = next((p for p in candidates if p.is_file()), None)
                    if filename is None:
                        raise FileNotFoundError('Missing contextual embedding ' + key)
                    matrix = np.load(str(filename))
                    if matrix.shape != (len(ex['tokens']), 868) or not np.isfinite(matrix).all():
                        raise ValueError('Invalid contextual embedding')
                    self.cache[key] = matrix


@torch.no_grad()
def evaluate_spans(model, features, examples, batch_size=32):
    model.eval()
    predictions = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        words, distance = features.batch(chunk)
        parts = model.static_parts(words, distance)
        tags = torch.zeros((len(chunk), features.num_words), device=features.device)
        labels = [[] for _ in chunk]
        for i in range(max(len(x['tokens']) for x in chunk)):
            decisions = model.from_static(parts, tags).argmax(dim=1)
            tags[:, i] = decisions + 1
            for j, decision in enumerate(decisions.cpu().tolist()):
                if i < len(chunk[j]['tokens']):
                    labels[j].append(decision)
        predictions.extend(labels)
    # No gold action mask, gold span boundary, or gold name enters decoding.
    return span_scores(predictions, examples)
