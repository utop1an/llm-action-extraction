"""cEASDRL CNN/DQN with three BIO decisions for complete-name extraction."""
import json
import random
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


def span_starts(actions, previous_labels):
    """Count decoded mentions: B, or I without a preceding B/I."""
    return (actions == 1) | ((actions == 2) & (previous_labels == 0))


class BalancedSpanReplay:
    """Original 90% positive-reward quota, balanced over correct O/B/I.

    Rows carry the gold BIO decision at index 6, used ONLY while training.
    Sampling within each stratum remains uniform; empty strata fall back to
    available strata. This is an explicit adaptation of upstream sign replay.
    """
    def __init__(self, capacity=50000):
        self.capacity, self.cursor = capacity, 0
        self.rows, self.pools, self.positions = [], [[], [], [], []], []

    def add(self, row):
        if len(row) != 7 or row[6] not in (0, 1, 2):
            raise ValueError('Balanced BIO replay needs the gold training label')
        bucket = 1 + row[6] if row[3] > 0 else 0
        slot = self.cursor
        if slot < len(self.rows):
            old_bucket, old_position = self.positions[slot]
            pool = self.pools[old_bucket]
            last = pool[-1]
            pool[old_position] = last
            self.positions[last] = (old_bucket, old_position)
            pool.pop()
            self.rows[slot] = row
            self.positions[slot] = (bucket, len(self.pools[bucket]))
        else:
            self.rows.append(row)
            self.positions.append((bucket, len(self.pools[bucket])))
        self.pools[bucket].append(slot)
        self.cursor = (slot + 1) % self.capacity

    def sample(self, batch_size):
        positive = int(.9 * batch_size)
        available = [b for b in (1, 2, 3) if self.pools[b]]
        offset = random.randrange(len(available)) if available else 0
        result = []
        for i in range(batch_size):
            bucket = available[(i + offset) % len(available)] if i < positive and available else 0
            pool = self.pools[bucket]
            slot = random.choice(pool) if pool else random.randrange(len(self.rows))
            result.append(self.rows[slot])
        return result


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
