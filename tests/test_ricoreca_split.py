"""Split integrity checks on synthetic records; not RiCoRecA experiment results."""
import json

import pytest

from scripts.ceasdrl.prepare_ricoreca_split import load_primary_records, make_split


def records():
    return [{'id': 'recipe_{}'.format(i), 'text': 'Synthetic recipe number {}.'.format(i)} for i in range(156)]


def test_document_holdout_is_complete_disjoint_and_order_independent():
    rows = records()
    split = make_split(rows)
    assert split['counts'] == {'train': 99, 'dev': 25, 'test': 32}
    assert split == make_split(list(reversed(rows)))
    groups = [set(v) for v in split['document_ids'].values()]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert set.union(*groups) == {r['id'] for r in rows}


def test_duplicate_ids_and_text_are_rejected_before_split():
    rows = records()
    with pytest.raises(ValueError, match='Duplicate document ID'):
        make_split(rows + [rows[0]])
    rows[-1]['text'] = '  SYNTHETIC   RECIPE NUMBER 0. '
    with pytest.raises(ValueError, match='Duplicate normalized text'):
        make_split(rows)


def test_missing_documents_are_not_silently_accepted():
    with pytest.raises(ValueError, match='no samples were silently skipped'):
        make_split(records()[:-1])


def test_loader_uses_primary_annotations_only_and_requires_both_sources(tmp_path):
    primary = tmp_path / 'Processed' / 'Annotator_P'
    secondary = tmp_path / 'Processed' / 'Annotator_H'
    primary.mkdir(parents=True)
    secondary.mkdir(parents=True)
    row = {'meta': {'id': 'example'}, 'text': 'Mix rice.', 'spans': [], 'relations': []}
    (primary / 'prodigy_dinner.json').write_text(json.dumps([row]), encoding='utf-8')
    (secondary / 'prodigy_dinner.json').write_text(json.dumps([row]), encoding='utf-8')
    with pytest.raises(ValueError, match='prodigy_lunch.json'):
        load_primary_records(tmp_path)
    (primary / 'prodigy_lunch.json').write_text('[]', encoding='utf-8')
    loaded, sources = load_primary_records(tmp_path)
    assert [r['id'] for r in loaded] == ['example']
    assert len(sources) == 2
