from copy import deepcopy

import pytest

from scripts.ceasdrl.evaluate_ceasdrl_holdout import independent_char_scores
from scripts.ceasdrl.evaluate_ceasdrl_spans import assemble_predictions, evaluate_predictions
from scripts.ceasdrl.risec_spans import make_span_examples


def gold_and_predictions():
    row = {'id':'r', 'text':'Mix cold rice. Mix water.',
           'words':['Mix','cold','rice','.','Mix','water','.'],
           'token_offsets':[[0,3],[4,8],[9,13],[13,14],[15,18],[19,24],[24,25]],
           'argument_links_available':True,
           'actions':[{'text':'Mix','trigger_tokens':[0],'char_span':[0,3],
                       'explicit_arguments':[{'target_text':'cold rice','token_indices':[1,2],'char_span':[4,13]}]},
                      {'text':'Mix','trigger_tokens':[4],'char_span':[15,18],
                       'explicit_arguments':[{'target_text':'water','token_indices':[5],'char_span':[19,24]}]}]}
    contexts=make_span_examples([row],'arg')
    outputs=[{'id':e['id'],'doc_id':'r','predicted_spans':e['gold_spans']} for e in contexts]
    return {'r':row},assemble_predictions({'r':row},contexts,outputs)


def test_exact_gold_roundtrip_both_scorers():
    records,predictions=gold_and_predictions()
    metrics,_=evaluate_predictions(records,predictions)
    assert metrics==independent_char_scores(records,predictions)
    assert all(x['f1']==1 for x in metrics.values())


def test_missing_action_keeps_its_arguments_in_denominator():
    records,predictions=gold_and_predictions()
    predictions[0]['actions'].pop()
    metrics,_=evaluate_predictions(records,predictions)
    assert metrics==independent_char_scores(records,predictions)
    assert metrics['linked_argument_name']['gold']==2
    assert metrics['linked_argument_name']['right']==1


def test_wrong_action_link_and_partial_name_are_not_true_positives():
    records,predictions=gold_and_predictions()
    actions=predictions[0]['actions']
    actions[1]['arguments']=deepcopy(actions[0]['arguments'])
    actions[0]['arguments'][0].update(name='cold',token_span=[1,2],char_span=[4,8])
    metrics,_=evaluate_predictions(records,predictions)
    assert metrics==independent_char_scores(records,predictions)
    assert metrics['action_name']['f1']==1
    assert metrics['linked_argument_name']['right']==0


def test_independent_scorer_rejects_invented_surface_names():
    records,predictions=gold_and_predictions()
    predictions[0]['actions'][0]['arguments'][0]['name']='cooked rice'
    with pytest.raises(ValueError,match='source occurrence'):
        independent_char_scores(records,predictions)
