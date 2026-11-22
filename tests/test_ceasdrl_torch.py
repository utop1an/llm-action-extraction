import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.ceasdrl.ceasdrl_torch import CnnDqn, Replay, binary_rewards, dqn_loss, evaluate
from scripts.ceasdrl.prepare_ceasdrl_pipeline import predicted_contexts


def test_contracted_channels_match_literal_upstream_forward_and_gradients():
    torch.set_num_threads(2)
    for mode in ['act', 'arg']:
        torch.manual_seed(1)
        model = CnnDqn(mode, dim=4, filters=3, hidden=7)
        literal = copy.deepcopy(model)
        words = torch.randn(3, 12, 4)
        tags = torch.randint(0, 3, (3, 12)).float()
        distances = torch.arange(12).repeat(3, 1).float()
        parts = [words]
        if mode == 'arg':
            parts.append(distances[:, :, None].repeat(1, 1, 4))
        parts.append(tags[:, :, None].repeat(1, 1, 4))
        expanded = torch.cat(parts, dim=2).transpose(1, 2)
        expected = literal.output(F.relu(literal.dense(torch.cat([F.relu(conv(expanded)).amax(dim=2) for conv in literal.convs], dim=1))))
        actual = model(words, tags, distances)
        assert torch.allclose(expected, actual, atol=1e-7, rtol=1e-5)
        expected.square().sum().backward()
        actual.square().sum().backward()
        for a, b in zip(model.parameters(), literal.parameters()):
            assert torch.allclose(a.grad, b.grad, atol=1e-7, rtol=1e-4)


def test_replay_wraparound_pools_remain_valid():
    replay = Replay(capacity=7)
    for i in range(30):
        replay.add((i, None, 1, 50. if i % 3 else -100., None, False))
        assert sorted(replay.pools[0] + replay.pools[1]) == list(range(len(replay.rows)))
        for sign, pool in enumerate(replay.pools):
            for position, slot in enumerate(pool):
                assert replay.positions[slot] == (sign, position)
                assert (replay.rows[slot][3] > 0) == bool(sign)
    assert min(row[0] for row in replay.sample(32)) >= 23


def test_original_reward_fixtures():
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / 'results/ceasdrl_risec_20261118/parity/reward_cases.json').read_text())
    for n, action, positive, count, terminal, expected in cases:
        actual = binary_rewards(np.array([action]), np.array([positive]), np.array([count]), n, np.array([terminal]))[0]
        assert abs(actual - expected) < 1e-4


def test_independent_target_receives_no_gradients():
    model = CnnDqn('act', dim=4, filters=3, hidden=7)
    target = copy.deepcopy(model)
    words, tags = torch.randn(2, 12, 4), torch.zeros(2, 12)
    loss = dqn_loss(model, target, words, tags, tags + 1, None,
                    torch.tensor([0, 1]), torch.tensor([50., 100.]), torch.tensor([False, True]))
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    assert all(p.grad is None for p in target.parameters())
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(model.parameters(), target.parameters()))


def test_inference_does_not_use_gold_labels_and_batches_match():
    class ToyFeatures:
        mode, device, num_words = 'act', torch.device('cpu'), 12

        def batch(self, examples):
            words = torch.zeros(len(examples), 12, 4)
            for i, ex in enumerate(examples):
                words[i, :len(ex['tokens'])] = ex['word_values']
            return words, None

    torch.manual_seed(5)
    model = CnnDqn('act', dim=4, filters=3, hidden=7)
    examples = [{'id': str(i), 'doc_id': str(i), 'tokens': ['x'] * n,
                 'positive_indices': [1], 'word_values': torch.randn(n, 4)} for i, n in enumerate([6, 10, 8])]
    _, first = evaluate(model, ToyFeatures(), examples, batch_size=1)
    for ex in examples:
        ex['positive_indices'] = [2, 3]
    _, second = evaluate(model, ToyFeatures(), examples, batch_size=3)
    assert [r['predicted_indices'] for r in first] == [r['predicted_indices'] for r in second]


def test_pipeline_contexts_ignore_gold_actions_and_arguments():
    records = {'recipe': {'words': ['w{}'.format(i) for i in range(160)],
                          'actions': [{'trigger_tokens': [70], 'explicit_arguments': [{'token_indices': [71]}]}]}}
    predictions = [{'doc_id': 'recipe', 'predicted_indices': [10, 120]}]
    first, contexts = predicted_contexts(records, predictions)
    records['recipe']['actions'] = []
    second, second_contexts = predicted_contexts(records, predictions)
    assert first == second and contexts == second_contexts
    assert [x['absolute_action_index'] for x in first] == [10, 120]
    assert [x['window_start'] for x in first] == [0, 60]
    assert all(x['positive_indices'] == [] for x in first)
    assert first[1]['act_inds'] == [60]
