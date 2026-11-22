import json
from pathlib import Path

import pytest

from scripts.ceasdrl.prepare_procedural_data import audit, read_openpi_a, read_recifinegold, read_risec, span_tokens, tokenize
from scripts.ceasdrl.ceasdrl_local import make_examples


def test_brat_offsets_links_and_implicit_targets(tmp_path):
    p = tmp_path / 'example.txt'
    p.write_text('Stir in flour. Bake it.', encoding='utf-8')
    p.with_suffix('.ann').write_text(
        'T1\tAC 0 7\tStir in\nT2\tFOOD 8 13\tflour\nT3\tAC 15 19\tBake\n'
        'R1\tArg1_PPT Arg1:T1 Arg2:T2\nR2\tArg1_PPT_X Arg1:T3 Arg2:T1\n', encoding='utf-8')
    r = read_risec(p, 'train')
    assert r['actions'][0]['trigger_tokens'] == [0, 1]
    assert r['actions'][0]['explicit_arguments'][0]['token_indices'] == [2]
    assert not r['actions'][1]['explicit_arguments']
    assert len(r['actions'][1]['implicit_arguments']) == 1
    ex, skipped = make_examples([r], 'arg', 100)
    assert not skipped
    assert ex[0]['positive_indices'] == [2]
    assert ex[1]['positive_indices'] == []  # no fabricated argument for ellipsis


def test_offsets_fail_loudly():
    with pytest.raises(ValueError, match='boundary'):
        span_tokens(tokenize('mix flour'), [(1, 3)])


def test_recifine_no_invented_argument_links_and_duplicate_grouping(tmp_path):
    p = tmp_path / 'gold.jsonl'
    rows = [{'id': str(i), 'text': 'Mix flour {}.'.format(i), 'label': [
        {'start': 0, 'end': 3, 'tag': 'Action by chef'},
        {'start': 4, 'end': 9, 'tag': 'Food'}]} for i in range(20)]
    rows.append(dict(rows[0], id='duplicate'))
    p.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    out = read_recifinegold(p)
    assert len(out) == 21
    assert len({r['split'] for r in out if r['id'] in ('0', 'duplicate')}) == 1
    assert audit(out)['actions'] == 21
    assert all(r['actions'][0]['explicit_arguments'] is None for r in out)
    with pytest.raises(ValueError, match='No annotated'):
        make_examples(out, 'arg', 100)


def test_document_leakage_rejected():
    with pytest.raises(ValueError, match='leaks'):
        audit([{'id': 'a', 'text': 'mix flour', 'split': 'train'},
               {'id': 'b', 'text': 'mix flour', 'split': 'test'}])


def test_long_action_document_not_silently_truncated():
    row = {'id': 'long', 'words': ['a'] * 501, 'actions': []}
    with pytest.raises(ValueError, match='exceeds'):
        make_examples([row], 'act', 500)


def test_actual_risec_sample_integrity():
    root = Path(__file__).resolve().parents[1] / 'data/procedural/risec_smoke/data'
    rows = [read_risec(p, split) for split in ['train', 'dev', 'test'] for p in sorted((root / split).glob('*.txt'))]
    stats = audit(rows)
    assert stats['split_documents'] == {'train': 8, 'dev': 2, 'test': 2}
    assert stats['actions'] == 171
    assert stats['implicit_argument_links'] == 89


def test_unrepresentable_argument_context_reported():
    row = {'id': 'far', 'source': 'RISeC', 'words': ['x'] * 200,
           'argument_links_available': True, 'actions': [
               {'trigger_tokens': [10], 'explicit_arguments': [{'token_indices': [190]}],
                'implicit_arguments': []}]}
    examples, skipped = make_examples([row], 'arg', 100)
    assert not examples
    assert skipped[0]['reason'] == 'argument_outside_window'


def test_openpi_groups_source_documents_and_preserves_subword_trigger():
    path = Path(__file__).resolve().parents[1] / 'data/procedural/openpi_a/data/silver_standard.jsonl'
    records = read_openpi_a(path)
    stats = audit(records)
    assert stats['documents'] == 515
    assert stats['source_documents'] == 327
    assert stats['explicit_argument_links'] == 0
    row = next(r for r in records if r['id'] == 'doc314_step3')
    assert row['words'][row['actions'][0]['trigger_tokens'][0]] == 'fold'
    groups = {}
    for r in records:
        groups.setdefault(r['source_group'], set()).add(r['split'])
    assert all(len(s) == 1 for s in groups.values())
    assert all(not r['argument_links_available'] for r in records)
