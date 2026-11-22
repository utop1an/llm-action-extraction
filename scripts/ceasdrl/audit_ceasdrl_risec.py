"""Independent raw-BRAT, cache, replay and label-free inference audit.

Preserves all existing data, weights and receipts. Does not fit any model.
The primary scorer below uses raw character offsets, not the project scorer.
"""
import argparse
import collections
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from .source_receipts import matches_source
except ImportError:
    from source_receipts import matches_source



def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text):
    return ' '.join(text.split()).casefold()


def raw_document(path):
    text = path.open(encoding='utf-8', newline='').read()
    entities, relations = {}, []
    for line in path.with_suffix('.ann').read_text(encoding='utf-8').splitlines():
        fields = line.split('\t')
        if fields[0].startswith('T'):
            kind, coordinates = fields[1].split(' ', 1)
            spans = [tuple(map(int, pair.split())) for pair in coordinates.split(';')]
            assert ' '.join(text[a:b] for a, b in spans) == fields[2]
            entities[fields[0]] = {'kind': kind, 'spans': spans, 'name': fields[2]}
        elif fields[0].startswith('R'):
            role, a, b = fields[1].split()
            relations.append((fields[0], role, a.split(':')[1], b.split(':')[1]))
    return {'text': text, 'entities': entities, 'relations': relations}


def key(doc_id, entity):
    assert len(entity['spans']) == 1
    return (doc_id,) + tuple(entity['spans'][0]) + (norm(entity['name']),)


def raw_sets(records, policy='legacy'):
    actions, links, link_roles = set(), set(), {}
    for doc_id, r in records.items():
        for e in r['entities'].values():
            if e['kind'] == 'AC':
                actions.add(key(doc_id, e))
        for _, role, a, b in r['relations']:
            source, target = r['entities'][a], r['entities'][b]
            if source['kind'] != 'AC' or target['kind'] == 'AC' or role.endswith('_X'):
                continue
            if policy != 'legacy' and (not role.startswith('Arg') or role.endswith('_Z')):
                continue
            if target['kind'] == 'CONDITION_CLAUSE' and policy != 'defined_with_conditions':
                continue
            pair = key(doc_id, source) + key(doc_id, target)[1:]
            links.add(pair)
            link_roles[pair] = role
    return actions, links, link_roles


def predicted_sets(records, predictions):
    assert len(predictions) == len(records)
    assert {p['doc_id'] for p in predictions} == set(records)
    actions, links = set(), set()
    duplicates = 0
    for row in predictions:
        text = records[row['doc_id']]['text']
        assert text == row['text']
        for a in row['actions']:
            start, stop = a['char_span']
            assert text[start:stop] == a['name']
            k = (row['doc_id'], start, stop, norm(a['name']))
            duplicates += k in actions
            actions.add(k)
            for arg in a['arguments']:
                x, y = arg['char_span']
                assert text[x:y] == arg['name']
                pair = k + (x, y, norm(arg['name']))
                duplicates += pair in links
                links.add(pair)
    assert not duplicates
    return actions, links


def score(predicted, gold):
    tp, fp, fn = len(predicted & gold), len(predicted - gold), len(gold - predicted)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'predicted': len(predicted), 'gold': len(gold),
            'precision': tp / (tp + fp) if tp + fp else 0.,
            'recall': tp / (tp + fn) if tp + fn else 0.,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def audit_replay(checkpoint, examples, num_words):
    import numpy as np
    replay = checkpoint['replay']
    assert sorted(sum(replay.pools, [])) == list(range(len(replay.rows)))
    maximum_error = 0.
    for slot, row in enumerate(replay.rows):
        index, before, decision, reward, after, terminal, label = row
        assert 0 <= index < len(examples), 'Non-training sample in replay'
        ex = examples[index]
        cursor = int(np.count_nonzero(before))
        assert 0 <= cursor < len(ex['tokens'])
        assert np.all(before[:cursor] > 0) and np.all(before[cursor:] == 0)
        assert int(np.count_nonzero(after)) == cursor + 1
        assert np.array_equal(before[:cursor], after[:cursor])
        assert after[cursor] == decision + 1 and np.all(after[cursor + 1:] == 0)
        assert label == ex['labels'][cursor]
        assert terminal == (cursor + 1 == len(ex['tokens']))
        labels = after[:cursor + 1].astype(np.int64) - 1
        previous = np.r_[0, labels[:-1]]
        count = int(np.sum((labels == 1) | ((labels == 2) & (previous == 0))))
        base = 100. if label else 50.
        expected = base if decision == label else -base
        rate = count / num_words
        expected += 250 * rate ** 2 * (1 if rate <= .1 and expected > 0 else -1)
        expected += 2 * terminal
        maximum_error = max(maximum_error, abs(expected - reward))
        bucket, position = replay.positions[slot]
        assert replay.pools[bucket][position] == slot
        assert bucket == (1 + label if reward > 0 else 0)
    assert maximum_error < 1e-4
    assert checkpoint['training_steps'] == sum(len(e['tokens']) for e in examples) * checkpoint['epoch']
    return {'rows_checked': len(replay.rows), 'all_samples_from_train': True,
            'state_transitions_valid': True, 'gold_labels_and_terminal_flags_match': True,
            'maximum_reward_error': maximum_error, 'replay_pool_counts': [len(p) for p in replay.pools],
            'positive_quota_for_batch32': 28, 'positive_fraction': 28 / 32}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Audit output already exists')
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    data = root / 'data/procedural/risec'
    run = root / 'results/ceasdrl_risec_20261118/spans_v2'
    report = {'status': 'running', 'no_training_performed': True,
              'started_at_utc': datetime.now(timezone.utc).isoformat()}
    bundle = read(data / 'source_bundle.json')
    report['dataset_source'] = {k: bundle[k] for k in ('repository', 'revision')}
    for item in bundle['files']:
        blob = item['content'].encode('utf-8')
        assert len(blob) == item['size']
        assert hashlib.sha1(b'blob ' + str(len(blob)).encode() + b'\0' + blob).hexdigest() == item['sha']
        assert (data / 'upstream' / item['path']).read_bytes() == blob
    report['source_files_verified'] = len(bundle['files'])
    raw, canonical, counts, taxonomy = {}, {}, {}, {}
    for split in ['train', 'dev', 'test']:
        folder = data / 'upstream/data' / split
        raw[split] = {p.stem: raw_document(p) for p in sorted(folder.glob('*.txt'))}
        canonical[split] = {r['id']: r for r in [json.loads(x) for x in (data / 'converted' / (split + '.jsonl')).read_text(encoding='utf-8').splitlines()]}
        assert set(raw[split]) == set(canonical[split])
        cnt = collections.Counter()
        for doc_id, r in raw[split].items():
            c = canonical[split][doc_id]
            assert r['text'] == c['text']
            # Independent tokenization verifies token boundaries are not supplied by labels.
            matches = list(re.finditer(r"\w+(?:['\u2019-]\w+)*|[^\w\s]", r['text']))
            assert [m.group() for m in matches] == c['words']
            assert [[m.start(), m.end()] for m in matches] == c['token_offsets']
            raw_action_ids = {tid for tid, e in r['entities'].items() if e['kind'] == 'AC'}
            assert raw_action_ids == {a['source_id'] for a in c['actions']}
            for a in c['actions']:
                e = r['entities'][a['source_id']]
                assert [c['token_offsets'][a['trigger_tokens'][0]][0], c['token_offsets'][a['trigger_tokens'][-1]][1]] == list(e['spans'][0])
                expected_ids = {rid for rid, role, s, t in r['relations'] if s == a['source_id']
                                and not role.endswith('_X') and r['entities'][t]['kind'] not in ('AC', 'CONDITION_CLAUSE')}
                assert expected_ids == {x['id'] for x in a['explicit_arguments']}
                for argument in a['explicit_arguments']:
                    target = r['entities'][argument['target']]
                    indices = argument['token_indices']
                    assert indices == list(range(indices[0], indices[-1] + 1))
                    coordinates = [c['token_offsets'][indices[0]][0], c['token_offsets'][indices[-1]][1]]
                    assert [coordinates] == [list(s) for s in target['spans']]
                    assert argument['target_text'] == target['name']
            for _, role, a, b in r['relations']:
                source, target = r['entities'][a], r['entities'][b]
                if source['kind'] != 'AC':
                    cnt['non_action_source'] += 1
                elif role.endswith(('_X', '_Z')):
                    cnt['suffix_X_or_Z'] += 1
                elif role in ('Next', 'Simultaneous'):
                    cnt['temporal_action_relation'] += 1
                elif target['kind'] == 'AC':
                    cnt['other_action_target'] += 1
                elif target['kind'] == 'CONDITION_CLAUSE':
                    cnt['condition_clause'] += 1
                elif role == 'TODO':
                    cnt['included_TODO'] += 1
                else:
                    assert role.startswith('Arg')
                    cnt['defined_explicit_noncondition'] += 1
        actions, links, _ = raw_sets(raw[split])
        counts[split] = {'documents': len(raw[split]), 'actions': len(actions), 'legacy_links': len(links),
                         'defined_noncondition_links': len(raw_sets(raw[split], 'defined_noncondition')[1])}
        taxonomy[split] = dict(cnt)
    report['split_counts'], report['relation_taxonomy'] = counts, taxonomy
    report['orphan_annotations'] = [str(p.relative_to(data / 'upstream')) for p in (data / 'upstream/data').rglob('*.ann') if not p.with_suffix('.txt').exists()]
    overlaps = {}
    for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        ids = set(raw[a]) & set(raw[b])
        ta = {norm(r['text']) for r in raw[a].values()}
        tb = {norm(r['text']) for r in raw[b].values()}
        overlaps[a + '_' + b] = {'document_ids': len(ids), 'normalized_full_texts': len(ta & tb)}
        assert not ids and not ta & tb
    report['split_overlap'] = overlaps
    predictions = read(run / 'pipeline_seed0/predictions.json')
    pa, pl = predicted_sets(raw['test'], predictions)
    metrics = {}
    for policy in ['legacy', 'defined_noncondition', 'defined_with_conditions']:
        ga, gl, roles = raw_sets(raw['test'], policy)
        metrics[policy] = {'action': score(pa, ga), 'linked_argument': score(pl, gl)}
    report['independent_character_span_metrics'] = metrics
    published = read(run / 'pipeline_seed0/receipt.json')['metrics']
    assert metrics['legacy']['action']['f1'] == published['action_name']['f1']
    assert metrics['legacy']['linked_argument']['f1'] == published['linked_argument_name']['f1']
    for independent, existing in [('action', 'action_name'), ('linked_argument', 'linked_argument_name')]:
        assert metrics['legacy'][independent]['tp'] == published[existing]['right']
        for field in ('predicted', 'gold', 'precision', 'recall'):
            assert metrics['legacy'][independent][field] == published[existing][field]
    report['current_scope_score_reproduced'] = True
    report['policy_sensitivity_note'] = 'Same frozen model/predictions, changed reference scope only; not a clean retraining or a replacement benchmark.'
    ga, gl, roles = raw_sets(raw['test'])
    todo = {pair for pair, role in roles.items() if role == 'TODO'}
    report['test_TODO_links'] = {'total': len(todo), 'predicted_correctly': len(todo & pl), 'examples': sorted(todo)}
    report['prediction_errors'] = {'self_arguments': sum(p[:4] == (p[0],) + p[4:] for p in pl),
                                   'action_fp': len(pa-ga), 'action_fn': len(ga-pa),
                                   'argument_fp': len(pl-gl), 'argument_fn': len(gl-pl)}
    report['artifact_sha256'] = {name: sha(run / name) for name in [
        'act_balanced_seed0/best.pt', 'arg_balanced_seed0/best.pt',
        'pipeline_seed0/predictions.json', 'pipeline_seed0/receipt.json', 'selection.json']}
    receipt = read(run / 'pipeline_seed0/receipt.json')
    for mode, name in [('act', 'act_balanced_seed0'), ('arg', 'arg_balanced_seed0')]:
        assert report['artifact_sha256'][name + '/best.pt'] == receipt['checkpoint_sha256'][mode]
    (args.output / 'data_and_scores.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Raw data, split membership and independent scoring checked', flush=True)

    import numpy as np
    import torch
    from ceasdrl_spans import SpanFeatures, SpanDqn, evaluate_spans
    from risec_spans import make_span_examples, predicted_span_contexts
    from evaluate_ceasdrl_spans import assemble_predictions
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
    feature_dir = data / 'features_spans_v2'
    manifest = read(feature_dir / 'manifest.json')
    cache_checks = {}
    for directory in [feature_dir, data / 'pipeline_spans_v2_seed0']:
        m = read(directory / 'manifest.json')
        dirs = [directory] + [(directory / p).resolve() for p in m.get('cache_directories', [])]
        for key, expected in m['feature_sha256'].items():
            path = next(p / (key + '.npy') for p in dirs if (p / (key + '.npy')).exists())
            assert sha(path) == expected
            arr = np.load(path)
            assert arr.shape[1] == 868 and np.isfinite(arr).all()
        cache_checks[str(directory.relative_to(data))] = len(m['feature_sha256'])
    report['embedding_cache_hashes_checked'] = cache_checks
    report['training'], report['inference'] = {}, {}
    for mode, name in [('act', 'act_balanced_seed0'), ('arg', 'arg_balanced_seed0')]:
        receipt = read(run / name / 'receipt.json')
        assert receipt['status'] == 'completed' and receipt['epochs_completed'] == 10
        assert receipt['dev']['f1'] == max(h['dev']['f1'] for h in receipt['history'])
        assert receipt['feature_manifest_sha256'] == sha(feature_dir / 'manifest.json')
        assert all(matches_source(f, h) for f, h in receipt['source_sha256'].items())
        for split in ['train', 'dev', 'test']:
            assert make_span_examples(list(canonical[split].values()), mode) == manifest['examples'][mode][split]
        last = torch.load(run / name / 'last.pt', map_location='cpu', weights_only=False)
        report['training'][mode] = audit_replay(last, manifest['examples'][mode]['train'], 600 if mode == 'act' else 100)
        report['training'][mode].update({'best_epoch': receipt['best_epoch'], 'training_steps': receipt['training_steps'],
                                         'optimizer_updates': receipt['optimizer_updates'], 'weights_changed': receipt['weights_changed']})
        assert all(a.data_ptr() != last['target_state'][k].data_ptr() for k, a in last['state_dict'].items())
        del last
    print('100,000 replay transitions and frozen embedding hashes checked', flush=True)
    # Re-run all 78 action inputs after stripping every annotation field.
    features = SpanFeatures(feature_dir, 'act', device)
    text_only = [{'id': e['id'], 'doc_id': e['doc_id'], 'tokens': e['tokens'], 'embedding_key': e['embedding_key']}
                 for e in features.examples['test']]
    model = SpanDqn('act').to(device)
    model.load_state_dict(torch.load(run / 'act_balanced_seed0/best.pt', map_location=device, weights_only=False)['state_dict'])
    _, actions = evaluate_spans(model, features, text_only, batch_size=7)
    assert actions == read(run / 'act_balanced_seed0/test_predictions.json')
    raw_inputs = {doc_id: {k: r[k] for k in ['words', 'text', 'token_offsets']} for doc_id, r in canonical['test'].items()}
    contexts = predicted_span_contexts(raw_inputs, actions)
    del model, features
    features = SpanFeatures(data / 'pipeline_spans_v2_seed0', 'arg', device)
    saved_contexts = [{k: v for k, v in e.items() if k != 'index'} for e in features.examples['test']]
    assert contexts == saved_contexts
    assert all('labels' not in e and 'gold_spans' not in e for e in contexts)
    model = SpanDqn('arg').to(device)
    model.load_state_dict(torch.load(run / 'arg_balanced_seed0/best.pt', map_location=device, weights_only=False)['state_dict'])
    _, arguments = evaluate_spans(model, features, contexts, batch_size=7)
    fresh = assemble_predictions(raw_inputs, contexts, arguments)
    assert fresh == predictions
    report['inference'] = {'documents': len(fresh), 'predicted_actions': len(contexts),
                           'batch_size': 7, 'original_batch_size': 32, 'annotations_stripped': True,
                           'fresh_all_document_predictions_identical': True}
    (args.output / 'fresh_predictions.json').write_text(json.dumps(fresh, indent=2, ensure_ascii=False), encoding='utf-8')
    report['status'] = 'passed_with_scope_findings'
    report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['audit_script_sha256'] = sha(Path(__file__))
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'metrics': metrics, 'taxonomy': taxonomy, 'inference': report['inference']}, indent=2))


if __name__ == '__main__':
    main()
