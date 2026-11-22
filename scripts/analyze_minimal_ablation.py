"""Paired, offline analysis of minimal versus full NL2P saved predictions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import pickle
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.result_paths import recorded_path, run_directory
import evaluation as ev
import src.evaluation_helpers as helpers

helpers.action_lemma_tokens = lru_cache(maxsize=None)(helpers.action_lemma_tokens)


def trace(item):
    words, acts, pred = item['words'], item['acts'], item['pred'] or []
    groups = defaultdict(list)
    for act in acts:
        if act.get('act_type') == ev.EXCLUSIVE_ACTION:
            groups[ev._exclusive_action_key(act)].append(act)
    processed, units = set(), []
    for order, act in enumerate(acts):
        kind = act.get('act_type')
        if kind == ev.EXCLUSIVE_ACTION:
            key = ev._exclusive_action_key(act)
            if key in processed:
                continue
            processed.add(key)
            units.append((order, kind, groups[key]))
        elif kind in {ev.ESSENTIAL_ACTION, ev.OPTIONAL_ACTION}:
            units.append((order, kind, [act]))
    used = [False] * len(pred)
    counts = np.zeros(6, dtype=np.int64)  # action TP/gold/pred; argument TP/gold/pred
    counts[2] = len(pred)
    matched, unit_rows = {}, []
    for order, kind, alternatives in sorted(units, key=lambda x: x[0]):
        act, idx = ev._strict_first_action_match(alternatives, pred, used, words)
        counts[1] += ev._gold_action_denominator_increment(kind, act is not None)
        row = {'unit_order': order, 'type': kind, 'alternatives': [helpers.action_record(a, words) for a in alternatives],
            'matched': act is not None, 'prediction_index': idx}
        if act is not None:
            _, right, gold, tagged, _ = helpers.match(act, pred[idx], words)
            counts[0] += 1
            counts[3:] += [right, gold, tagged]
            matched[order] = {'type': kind, 'gold': helpers.action_record(act, words), 'prediction': pred[idx],
                'argument_counts': [right, gold, tagged]}
            row['prediction'] = pred[idx]
        unit_rows.append(row)
    return {'counts': counts, 'matched': matched, 'units': unit_rows,
        'unmatched_predictions': [p for i, p in enumerate(pred) if not used[i]]}


def metrics(counts):
    values = []
    for i in [0, 3]:
        tp, gold, pred = counts[i:i+3]
        values.extend([float(tp / pred) if pred else 0., float(tp / gold) if gold else 0.,
            float(2 * tp / (gold + pred)) if gold + pred else 0.])
    return values


def f1(counts, offset):
    return np.divide(200 * counts[..., offset], counts[..., offset+1] + counts[..., offset+2],
        out=np.zeros(counts.shape[:-1], dtype=float), where=(counts[..., offset+1] + counts[..., offset+2]) != 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='gpt-5.4-mini')
    args = parser.parse_args()
    outdir = ROOT / 'results/rebuttal_20261027' / args.model
    previous = json.loads((outdir / 'comparison.json').read_text(encoding='utf-8'))
    expected = {(x['condition'], x['dataset']): x for x in previous['metrics']}
    for rel, key in [('evaluation.py', 'evaluator_sha256'), ('src/evaluation_helpers.py', 'helpers_sha256')]:
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == previous[key]
    for record in previous['inputs']:
        assert hashlib.sha256(recorded_path(record['path']).read_bytes()).hexdigest() == record['sha256']
    report = {'model': args.model, 'created_at': datetime.now(ZoneInfo('Australia/Sydney')).isoformat(timespec='seconds'),
        'bootstrap': {'replicates': 10000, 'seed': 20261027, 'unit': 'paired document',
            'interval': 'percentile 95%, no multiple-comparison adjustment',
            'scope': 'fixed saved outputs; excludes model-run and prompt-development variability'},
        'datasets': []}
    cases = []
    for di, ds in enumerate(['win2k', 'cooking', 'wikihow']):
        dataset = {}
        traced = {}
        for condition, solver in [('minimal', 'nl2p_minimal_rebuttal_20261027'), ('nl2p_full', 'nl2p_1')]:
            path = run_directory(ROOT / 'results', solver) / args.model / f'{ds}_{solver}_{args.model}.pkl'
            with path.open('rb') as f:
                dataset[condition] = pickle.load(f)
            traced[condition] = [trace(item) for item in dataset[condition]]
        arrays = {c: np.array([t['counts'] for t in traced[c]]) for c in traced}
        sums = {c: a.sum(axis=0) for c, a in arrays.items()}
        for c in arrays:
            assert np.allclose(metrics(sums[c]), [expected[c, ds][k] for k in
                ['action_precision', 'action_recall', 'action_f1', 'argument_precision', 'argument_recall', 'argument_f1']], atol=1e-12, rtol=0)
        rng = np.random.default_rng(20261027 + di)
        indices = rng.integers(0, len(arrays['minimal']), size=(10000, len(arrays['minimal'])))
        boot = {c: a[indices].sum(axis=1) for c, a in arrays.items()}
        info = {'dataset': ds, 'num_docs': len(indices[0]), 'counts': {c: x.tolist() for c, x in sums.items()},
            'f1_delta_full_minus_minimal_pp': {}, 'unit_match_comparison_by_type': {},
            'common_matched_argument_counts': {}, 'common_matched_argument_prf': {},
            'unmatched_predicted_verbs': {}, 'matched_predicted_verbs': {}, 'source_verb_presence': {}}
        for metric, offset in [('action', 0), ('argument', 3)]:
            delta = f1(boot['nl2p_full'], offset) - f1(boot['minimal'], offset)
            info['f1_delta_full_minus_minimal_pp'][metric] = {'delta': float(f1(sums['nl2p_full'], offset)-f1(sums['minimal'], offset)),
                'ci95': np.quantile(delta, [.025, .975]).tolist()}
        changes, common_args = Counter(), {c: np.zeros(3, dtype=np.int64) for c in arrays}
        unmatched_verbs = {c: Counter() for c in arrays}
        matched_verbs = {c: Counter() for c in arrays}
        source_presence = {c: Counter() for c in arrays}
        common_doc_args = {c: [] for c in arrays}
        for doc in range(len(arrays['minimal'])):
            mini, full = traced['minimal'][doc], traced['nl2p_full'][doc]
            assert [(u['unit_order'],u['type'],u['alternatives']) for u in mini['units']] == [(u['unit_order'],u['type'],u['alternatives']) for u in full['units']]
            for unit in mini['units']:
                key = unit['unit_order']
                outcome = ('both' if key in mini['matched'] and key in full['matched'] else
                    'minimal_only' if key in mini['matched'] else 'full_only' if key in full['matched'] else 'neither')
                changes[f"type{unit['type']}_{outcome}"] += 1
            common = set(mini['matched']) & set(full['matched'])
            for c, result in [('minimal', mini), ('nl2p_full', full)]:
                doc_args = np.zeros(3, dtype=np.int64)
                for key in common:
                    doc_args += result['matched'][key]['argument_counts']
                common_args[c] += doc_args
                common_doc_args[c].append(doc_args)
                source_tokens = set().union(*(helpers.action_lemma_tokens(w) for w in dataset[c][doc]['words']))
                for pred in result['unmatched_predictions']:
                    unmatched_verbs[c][helpers.lemma_text(pred['verb'])] += 1
                    tokens = helpers.action_lemma_tokens(pred['verb'])
                    source_presence[c]['all_lemma_tokens_present' if tokens.issubset(source_tokens) else 'some_lemma_tokens_absent'] += 1
                for match in result['matched'].values():
                    matched_verbs[c][helpers.lemma_text(match['prediction']['verb'])] += 1
            docrow = {'dataset': ds, 'doc_id': doc, 'sentences': [' '.join(s) for s in dataset['minimal'][doc]['sents']],
                'minimal_counts': mini['counts'].tolist(), 'full_counts': full['counts'].tolist(),
                'minimal_unmatched': mini['unmatched_predictions'], 'full_unmatched': full['unmatched_predictions'],
                'minimal_only_gold': [mini['matched'][i] for i in sorted(set(mini['matched'])-set(full['matched']))],
                'full_only_gold': [full['matched'][i] for i in sorted(set(full['matched'])-set(mini['matched']))],
                'common_matches': [{'unit_order': i, 'minimal': mini['matched'][i], 'full': full['matched'][i]} for i in sorted(common)],
                'gold_units': mini['units']}
            cases.append(docrow)
        info['unit_match_comparison_by_type'] = dict(changes)
        for c in arrays:
            info['common_matched_argument_counts'][c] = common_args[c].tolist()
            tp, gold, pred = common_args[c]
            info['common_matched_argument_prf'][c] = [float(100*tp/pred), float(100*tp/gold), float(200*tp/(gold+pred))]
            info['unmatched_predicted_verbs'][c] = unmatched_verbs[c].most_common(30)
            info['matched_predicted_verbs'][c] = matched_verbs[c].most_common(30)
            info['source_verb_presence'][c] = dict(source_presence[c])
        arg_boot = {c: np.array(common_doc_args[c])[indices].sum(axis=1) for c in arrays}
        delta = f1(arg_boot['nl2p_full'], 0)-f1(arg_boot['minimal'], 0)
        info['common_matched_argument_delta_ci95'] = np.quantile(delta, [.025, .975]).tolist()
        report['datasets'].append(info)
        print(json.dumps({k:v for k,v in info.items() if k not in ['unmatched_predicted_verbs','matched_predicted_verbs']}, indent=2), flush=True)
    (outdir / 'minimal_analysis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    (outdir / 'minimal_cases.json').write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Saved {len(cases)} document traces to {outdir}', flush=True)


if __name__ == '__main__':
    main()
