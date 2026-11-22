from scripts.ceasdrl.evaluate_risec_names import evaluate_names


def test_first_token_is_not_credited_as_complete_argument_name():
    gold = [{'id': 'r', 'actions': [{'trigger_tokens': [0], 'text': 'Beat',
                                   'explicit_arguments': [{'token_indices': [1, 2], 'target_text': 'the eggs'}]}]}]
    pred = [{'doc_id': 'r', 'action_index': 0, 'action_token': 'beat',
             'argument_indices': [1], 'argument_first_tokens': ['the']}]
    result = evaluate_names(gold, pred)
    assert result['action_name']['f1'] == 1
    assert result['linked_argument_name']['f1'] == 0
    pred[0]['argument_first_tokens'] = ['THE   eggs']
    assert evaluate_names(gold, pred)['linked_argument_name']['f1'] == 1


def test_wrong_owner_does_not_get_argument_credit_and_missed_actions_count():
    gold = [{'id': 'r', 'actions': [
        {'trigger_tokens': [0], 'text': 'cut', 'explicit_arguments': [{'token_indices': [1], 'target_text': 'onion'}]},
        {'trigger_tokens': [3], 'text': 'add', 'explicit_arguments': [{'token_indices': [4], 'target_text': 'salt'}]}]}]
    pred = [{'doc_id': 'r', 'action_index': 0, 'action_token': 'cut',
             'argument_indices': [4], 'argument_first_tokens': ['salt']}]
    result = evaluate_names(gold, pred)
    assert result['action_name']['gold'] == 2
    assert result['linked_argument_name']['right'] == 0
    assert result['linked_argument_name']['gold'] == 2


def test_identical_names_at_different_occurrences_are_not_collapsed():
    gold = [{'id': 'r', 'actions': [
        {'trigger_tokens': [0], 'text': 'cut', 'explicit_arguments': []},
        {'trigger_tokens': [4], 'text': 'cut', 'explicit_arguments': []}]}]
    pred = [{'doc_id': 'r', 'action_index': 0, 'action_token': 'cut',
             'argument_indices': [], 'argument_first_tokens': []}]
    assert evaluate_names(gold, pred)['action_name']['recall'] == .5
