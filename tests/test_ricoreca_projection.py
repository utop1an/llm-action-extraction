import copy

import pytest

from scripts.prepare_ricoreca_data import convert_record, parse_published
from scripts.risec_spans import make_span_examples, decode_bio


def fixture():
    text = 'Mix cold rice and stir. Mix water until hot.'
    entries = [('Mix', 0, 'ACTION', 'Mix'), ('cold', 4, 'STT_INGR', 'Mix, stir'),
               ('rice', 9, 'INGR', 'Mix, stir'), ('stir', 18, 'ACTION', 'stir'),
               ('Mix', 24, 'ACTION', 'Mix'), ('water', 28, 'INGR', 'Mix'),
               ('until', 34, 'Until', 'until'), ('hot', 40, 'STT_INGR', 'until')]
    spans = [dict(start=start, end=start+len(word), token_end=i, label=label)
             for i, (word, start, label, _) in enumerate(entries)]
    edges = [(1, 2, 'Modifier'), (2, 0, 'Member'), (0, 3, 'Join'),
             (5, 4, 'Member'), (7, 6, 'Modifier'), (6, 4, 'Dependency')]
    relations = [dict(head=a, child=b, label=k) for a,b,k in edges]
    out, previous = '', 0
    for word, start, label, member in entries:
        out += text[previous:start] + '['+word+' | | '+member+' | '+label.upper()+' | ]'
        previous = start+len(word)
    out += text[previous:]
    return {'text': text, 'meta': {'id': 'example'}, 'spans': spans, 'relations': relations}, {'id':'example', 'recipe':text, 'out':out}


def test_repeat_occurrences_join_modifier_and_control_barrier():
    raw, released = fixture()
    row = convert_record(raw, released)
    assert [[a['target_text'] for a in act['explicit_arguments']] for act in row['actions']] == [['cold', 'rice'], ['cold', 'rice'], ['water']]
    assert row['excluded_control_members'][0]['text'] == 'hot'
    assert all('Dependency' not in [x['label'] for x in arg['relation_path']] for act in row['actions'] for arg in act['explicit_arguments'])
    for mode in ['act', 'arg']:
        for ex in make_span_examples([row], mode):
            assert decode_bio(ex['labels']) == ex['gold_spans']


def test_missing_relation_fails_instead_of_nearest_verb_assignment():
    raw, released = fixture()
    raw['relations'] = [r for r in raw['relations'] if r['head'] != 5]
    with pytest.raises(ValueError, match='Unresolved'):
        convert_record(raw, released)


def test_two_same_named_graph_targets_fail_instead_of_silent_guess():
    raw, released = fixture()
    raw['relations'].append(dict(head=0, child=4, label='Join'))
    with pytest.raises(ValueError, match='ambiguous'):
        convert_record(raw, released)


def test_published_text_must_reconstruct_exactly():
    _, released = fixture()
    released['recipe'] += ' changed'
    with pytest.raises(ValueError, match='exact source text'):
        parse_published(released)


def test_dangling_relation_is_recorded_but_cannot_supply_gold_link():
    raw, released = fixture()
    raw['relations'].append(dict(head=999, child=0, label='Member'))
    row = convert_record(raw, released)
    assert len(row['invalid_raw_relations']) == 1
    assert len(row['actions'][0]['explicit_arguments']) == 2
