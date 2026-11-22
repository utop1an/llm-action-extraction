import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.ceasdrl_spans import SpanDqn, evaluate_spans, span_rewards, span_starts, BalancedSpanReplay
from scripts.ceasdrl_torch import binary_rewards, dqn_loss
from scripts.risec_spans import encode_bio, decode_bio, make_span_examples, predicted_span_contexts
from scripts.evaluate_ceasdrl_spans import assemble_predictions, evaluate_predictions


def test_adjacent_multiword_spans_roundtrip_and_invalid_spans_fail():
    spans = [[0, 2], [2, 4], [5, 6]]
    assert encode_bio(7, spans) == [1, 2, 1, 2, 0, 1, 0]
    assert decode_bio(encode_bio(7, spans)) == spans
    assert decode_bio([2, 2, 0, 2]) == [[0, 2], [3, 4]]
    with pytest.raises(ValueError):
        encode_bio(5, [[0, 3], [2, 4]])
    with pytest.raises(ValueError):
        encode_bio(5, [[4, 6]])


def test_full_risec_annotations_roundtrip_without_dropping_any_span():
    root = Path(__file__).resolve().parents[1] / 'data/procedural/risec/converted'
    counts = {'act': 0, 'arg': 0}
    for split in ['train', 'dev', 'test']:
        records = [json.loads(x) for x in (root / (split + '.jsonl')).read_text(encoding='utf-8').splitlines()]
        for mode in counts:
            examples = make_span_examples(records, mode)
            for ex in examples:
                assert decode_bio(ex['labels']) == ex['gold_spans']
                counts[mode] += len(ex['gold_spans'])
    assert counts == {'act': 3061, 'arg': 5520}


def test_oracle_spans_roundtrip_names_and_window_offsets_for_entire_corpus():
    root = Path(__file__).resolve().parents[1] / 'data/procedural/risec/converted'
    for split in ['train', 'dev', 'test']:
        rows = [json.loads(x) for x in (root / (split + '.jsonl')).read_text(encoding='utf-8').splitlines()]
        records = {r['id']: r for r in rows}
        action_examples = make_span_examples(rows, 'act')
        action_predictions = [{'doc_id': e['doc_id'], 'predicted_spans': e['gold_spans']} for e in action_examples]
        contexts = predicted_span_contexts(records, action_predictions)
        gold_arguments = {(e['doc_id'], tuple(e['action_span'])): e for e in make_span_examples(rows, 'arg')}
        argument_predictions = [{'id': e['id'], 'doc_id': e['doc_id'],
                                 'predicted_spans': gold_arguments[(e['doc_id'], tuple(e['action_span']))]['gold_spans']}
                                for e in contexts]
        output = assemble_predictions(records, contexts, argument_predictions)
        metrics, errors = evaluate_predictions(records, output)
        assert metrics['action_name']['f1'] == 1.
        assert metrics['linked_argument_name']['f1'] == 1.
        assert all(not values for values in errors.values())


def test_bio_reward_reduces_to_binary_but_penalizes_wrong_boundary():
    actions = np.array([0, 1, 0, 1])
    labels = np.array([0, 1, 1, 0])
    counts = np.array([0, 1, 20, 50])
    terminal = np.array([False, True, False, True])
    assert np.array_equal(span_rewards(actions, labels, counts, 100, terminal),
                          binary_rewards(actions, labels.astype(bool), counts, 100, terminal))
    assert span_rewards(np.array([1]), np.array([2]), np.array([0]), 100, np.array([False]))[0] == -100


def test_mention_rate_counts_decoded_spans_not_each_inside_token():
    labels = [1, 2, 2, 0, 2, 2, 1, 0]
    starts = [bool(span_starts(np.array([label]), np.array([labels[i - 1] if i else 0]))[0])
              for i, label in enumerate(labels)]
    assert sum(starts) == len(decode_bio(labels)) == 3


def test_balanced_replay_rare_inside_labels_and_wraparound():
    replay = BalancedSpanReplay(capacity=23)
    for i in range(60):
        label = 1 if i % 11 == 0 else 2 if i % 13 == 0 else 0
        reward = -50. if i % 7 == 0 else 100. if label else 50.
        replay.add((i, None, label, reward, None, False, label))
        assert sorted(sum(replay.pools, [])) == list(range(len(replay.rows)))
        for b, pool in enumerate(replay.pools):
            for pos, slot in enumerate(pool):
                assert replay.positions[slot] == (b, pos)
                assert b == (1 + replay.rows[slot][6] if replay.rows[slot][3] > 0 else 0)
    batch = replay.sample(32)
    positive = [r for r in batch if r[3] > 0]
    counts = [sum(r[6] == label for r in positive) for label in range(3)]
    assert len(positive) == 28
    assert max(counts) - min(counts) <= 1
    assert min(r[0] for r in batch) >= 37


def test_pipeline_contexts_and_names_need_no_gold_annotations():
    row = {'words': ['mix', 'the', 'red', 'sauce', 'slowly'], 'text': 'mix the red sauce slowly',
           'token_offsets': [[0, 3], [4, 7], [8, 11], [12, 17], [18, 24]]}
    records = {'doc': row}
    actions = [{'doc_id': 'doc', 'predicted_spans': [[0, 1]]}]
    contexts = predicted_span_contexts(records, actions)
    assert 'labels' not in contexts[0] and 'gold_spans' not in contexts[0]
    args = [{'id': 'doc:pred0', 'doc_id': 'doc', 'predicted_spans': [[1, 4]]}]
    output = assemble_predictions(records, contexts, args)
    assert output[0]['actions'][0]['arguments'][0]['name'] == 'the red sauce'
    records['doc']['actions'] = ['deliberately invalid gold data']
    assert predicted_span_contexts(records, actions) == contexts
    assert assemble_predictions(records, contexts, args) == output


def test_overlong_predicted_action_is_retained_instead_of_silently_dropped():
    records = {'d': {'words': ['x'] * 160}}
    contexts = predicted_span_contexts(records, [{'doc_id': 'd', 'predicted_spans': [[0, 150]]}])
    assert len(contexts) == 1
    assert contexts[0]['action_span'] == [0, 150]
    assert len(contexts[0]['tokens']) == 100


def test_missing_actions_wrong_owners_and_incomplete_names_are_errors():
    records = {'d': {'actions': [
        {'text': 'mix', 'trigger_tokens': [0], 'explicit_arguments': [{'token_indices': [1, 2], 'target_text': 'red sauce'}]},
        {'text': 'serve', 'trigger_tokens': [4], 'explicit_arguments': [{'token_indices': [5], 'target_text': 'rice'}]}]}}
    pred = [{'doc_id': 'd', 'actions': [{'name': 'mix', 'token_span': [0, 1], 'arguments': [
        {'name': 'red', 'token_span': [1, 2]}, {'name': 'rice', 'token_span': [5, 6]}]}]}]
    metrics, _ = evaluate_predictions(records, pred)
    assert metrics['action_name']['f1'] == pytest.approx(2 / 3)
    assert metrics['linked_argument_name'] == {'precision': 0., 'recall': 0., 'f1': 0., 'right': 0, 'predicted': 2, 'gold': 2}
    pred[0]['actions'][0]['arguments'] = [{'name': 'RED  SAUCE', 'token_span': [1, 3]}]
    metrics, _ = evaluate_predictions(records, pred)
    assert metrics['linked_argument_name']['f1'] == pytest.approx(2 / 3)


def test_bio_inference_independent_of_gold_and_batch_size():
    class ToyFeatures:
        device, num_words = torch.device('cpu'), 12

        def batch(self, examples):
            words = torch.zeros(len(examples), 12, 4)
            for i, ex in enumerate(examples):
                words[i, :len(ex['tokens'])] = ex['word_values']
            return words, torch.arange(12).repeat(len(examples), 1).float()

    torch.set_num_threads(2)
    for mode in ['act', 'arg']:
        torch.manual_seed(5)
        model = SpanDqn(mode, dim=4, filters=3, hidden=7)
        examples = [{'id': str(i), 'doc_id': str(i), 'tokens': ['x'] * n,
                     'gold_spans': [[1, 2]], 'word_values': torch.randn(n, 4)} for i, n in enumerate([6, 10, 8])]
        _, first = evaluate_spans(model, ToyFeatures(), examples, batch_size=1)
        for ex in examples:
            ex['gold_spans'] = [[0, 4]]
            ex['labels'] = [2] * len(ex['tokens'])
        _, second = evaluate_spans(model, ToyFeatures(), examples, batch_size=3)
        assert first == second


def test_three_class_bellman_target_updates_i_class_without_target_gradients():
    torch.set_num_threads(2)
    model = SpanDqn('arg', dim=4, filters=3, hidden=7)
    target = copy.deepcopy(model)
    words = torch.randn(2, 12, 4)
    tags = torch.zeros(2, 12)
    distances = torch.arange(12).repeat(2, 1).float()
    actions, rewards, terminal = torch.tensor([2, 1]), torch.tensor([100., -100.]), torch.tensor([True, False])
    q = model(words, tags, distances)
    expected = q.detach().clone()
    with torch.no_grad():
        expected[0, 2] = 100.
        expected[1, 1] = -100. + .9 * target(words, tags + 1, distances)[1].max()
    loss = dqn_loss(model, target, words, tags, tags + 1, distances, actions, rewards, terminal)
    assert torch.allclose(loss, (q - expected).square().mean())
    loss.backward()
    assert model.output.bias.grad[2].abs() > 0
    assert all(p.grad is None for p in target.parameters())
