"""GPU implementation of the upstream cEASDRL CNN and binary-label DQN.

Repeated distance/tag channels are algebraically contracted, not replaced by
learned embeddings. Keras weight/forward/loss/gradient parity is checked separately.
"""
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CnnDqn(nn.Module):
    def __init__(self, mode, dim=868, filters=32, hidden=256):
        super().__init__()
        self.mode, self.dim = mode, dim
        width = dim * (2 if mode == 'act' else 3)
        self.convs = nn.ModuleList([nn.Conv1d(width, filters, k) for k in [2, 3, 4, 5]])
        self.dense = nn.Linear(filters * 4, hidden)
        self.output = nn.Linear(hidden, 2)
        for conv in self.convs:
            # Upstream is Conv2D(k, width), with one input channel. Keras includes
            # BOTH spatial dimensions in fan_out; Conv1D's default differs.
            spatial = conv.kernel_size[0] * width
            fan_in, fan_out = spatial, spatial * filters
            std = (2. / (fan_in + fan_out)) ** .5 / .87962566103423978
            nn.init.trunc_normal_(conv.weight, std=std, a=-2 * std, b=2 * std)
            nn.init.zeros_(conv.bias)
        for layer in [self.dense, self.output]:
            nn.init.trunc_normal_(layer.weight, std=.05, a=-.1, b=.1)
            nn.init.zeros_(layer.bias)

    def static_parts(self, words, distances=None):
        x = words.transpose(1, 2)
        result = []
        for conv in self.convs:
            value = F.conv1d(x, conv.weight[:, :self.dim], conv.bias)
            if self.mode == 'arg':
                if distances is None:
                    raise ValueError('Argument mode requires distances')
                value = value + F.conv1d(distances[:, None, :],
                                        conv.weight[:, self.dim:2*self.dim].sum(dim=1, keepdim=True))
            result.append(value)
        return result

    def from_static(self, parts, tags):
        pooled = []
        for conv, static in zip(self.convs, parts):
            dynamic = F.conv1d(tags[:, None, :], conv.weight[:, -self.dim:].sum(dim=1, keepdim=True))
            pooled.append(F.relu(static + dynamic).amax(dim=2))
        return self.output(F.relu(self.dense(torch.cat(pooled, dim=1))))

    def forward(self, words, tags, distances=None):
        return self.from_static(self.static_parts(words, distances), tags)

    def load_keras_h5(self, path):
        import h5py
        with h5py.File(path, 'r') as root:
            if 'model_weights' in root:
                root = root['model_weights']
            weighted = []
            for raw_name in root.attrs['layer_names']:
                name = raw_name.decode() if isinstance(raw_name, bytes) else raw_name
                group = root[name]
                values = []
                for raw_weight in group.attrs.get('weight_names', []):
                    key = raw_weight.decode() if isinstance(raw_weight, bytes) else raw_weight
                    values.append(np.array(group[key]))
                if values:
                    weighted.append(values)
        if len(weighted) != 6:
            raise ValueError('Unexpected Keras layer count')
        with torch.no_grad():
            for conv, (kernel, bias) in zip(self.convs, weighted[:4]):
                conv.weight.copy_(torch.from_numpy(kernel[:, :, 0, :].transpose(2, 1, 0)))
                conv.bias.copy_(torch.from_numpy(bias))
            for layer, (kernel, bias) in zip([self.dense, self.output], weighted[4:]):
                layer.weight.copy_(torch.from_numpy(kernel.T))
                layer.bias.copy_(torch.from_numpy(bias))


def binary_rewards(actions, positive, positive_count, num_words, terminal):
    """Upstream reward for label=1/2, including action-rate shaping and end bonus.

    No ES/OP/EX classes: positive means an annotated extraction target.
    """
    base = np.where(positive, 100., 50.)
    reward = np.where(actions == positive.astype(np.int64), base, -base)
    rate = positive_count / float(num_words)
    shaping = 250. * np.square(rate)
    reward += np.where((rate <= .1) & (reward > 0), shaping, -shaping)
    return (reward + 2. * terminal).astype(np.float32)


class Replay:
    """Bounded ring with O(1) reward-sign pools; no duplicated word embeddings."""
    def __init__(self, capacity=50000):
        self.capacity, self.cursor = capacity, 0
        self.rows, self.pools, self.positions = [], [[], []], []

    def add(self, row):
        sign = int(row[3] > 0)
        slot = self.cursor
        if slot < len(self.rows):
            old_sign, old_position = self.positions[slot]
            pool = self.pools[old_sign]
            last = pool[-1]
            pool[old_position] = last
            self.positions[last] = (old_sign, old_position)
            pool.pop()
            self.rows[slot] = row
            self.positions[slot] = (sign, len(self.pools[sign]))
        else:
            self.rows.append(row)
            self.positions.append((sign, len(self.pools[sign])))
        self.pools[sign].append(slot)
        self.cursor = (slot + 1) % self.capacity

    def sample(self, batch_size):
        positive = int(.9 * batch_size)
        selected = []
        for sign, count in [(1, positive), (0, batch_size - positive)]:
            pool = self.pools[sign]
            for _ in range(count):
                idx = random.choice(pool) if pool else random.randrange(len(self.rows))
                selected.append(self.rows[idx])
        return selected


def scores(predictions, examples):
    correct = predicted = gold = 0
    output = []
    for chosen, example in zip(predictions, examples):
        expected = set(example['positive_indices'])
        found = set(chosen)
        correct += len(found & expected)
        predicted += len(found)
        gold += len(expected)
        output.append({'id': example['id'], 'doc_id': example['doc_id'],
                       'predicted_indices': sorted(found), 'gold_indices': sorted(expected)})
    return {'precision': correct / predicted if predicted else 0.,
            'recall': correct / gold if gold else 0.,
            'f1': 2 * correct / (predicted + gold) if predicted + gold else 0.,
            'right': correct, 'predicted': predicted, 'gold': gold}, output


class Features:
    def __init__(self, path, mode, device):
        self.path, self.mode, self.device = Path(path), mode, device
        self.manifest = json.loads((self.path / 'manifest.json').read_text(encoding='utf-8'))
        self.examples = self.manifest['examples'][mode]
        self.num_words = 600 if mode == 'act' else 100
        self.cache = {}
        self.all_examples = []
        for split, items in self.examples.items():
            for ex in items:
                ex['index'] = len(self.all_examples)
                self.all_examples.append(ex)
                key = ex['embedding_key']
                if key not in self.cache:
                    a = np.load(str(self.path / (key + '.npy')))
                    if a.shape != (len(ex['tokens']), 868) or not np.isfinite(a).all():
                        raise ValueError('Embedding mismatch')
                    self.cache[key] = a

    def batch(self, examples):
        words = np.zeros((len(examples), self.num_words, 868), dtype=np.float32)
        distances = np.zeros((len(examples), self.num_words), dtype=np.float32)
        for i, ex in enumerate(examples):
            n = len(ex['tokens'])
            words[i, :n] = self.cache[ex['embedding_key']]
            if self.mode == 'arg':
                distances[i, :n] = np.abs(np.arange(n) - ex['act_ind'])
        return torch.from_numpy(words).to(self.device), torch.from_numpy(distances).to(self.device)


@torch.no_grad()
def evaluate(model, features, examples, batch_size=32):
    model.eval()
    predictions = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        words, distance = features.batch(chunk)
        parts = model.static_parts(words, distance)
        tags = torch.zeros((len(chunk), features.num_words), device=features.device)
        chosen = [[] for _ in chunk]
        for i in range(max(len(x['tokens']) for x in chunk)):
            decisions = model.from_static(parts, tags).argmax(dim=1)
            tags[:, i] = decisions + 1
            decisions = decisions.cpu().numpy()
            for j, ex in enumerate(chunk):
                if i < len(ex['tokens']) and decisions[j]:
                    if features.mode != 'arg' or i not in ex['act_inds']:
                        chosen[j].append(i)
        predictions.extend(chosen)
    return scores(predictions, examples)


def dqn_loss(model, target, words, before, after, distances, actions, rewards, terminal, gamma=.9):
    prediction = model(words, before, distances)
    with torch.no_grad():
        expected = prediction.detach().clone()
        maximum = target(words, after, distances).max(dim=1).values
        expected[torch.arange(len(actions), device=actions.device), actions] = rewards + gamma * maximum * (~terminal)
    return F.mse_loss(prediction, expected)
