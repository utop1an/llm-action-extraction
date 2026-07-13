import csv
import math

import evaluation as ev


WORDS = [
    "choose",
    "a",
    "square",
    "shadow",
    "box",
    "open",
    "the",
    "file",
    "save",
    "as",
    "PDF",
    "move",
    "to",
    "folder",
    "close",
]


def act(verb_idx, obj_idxs=None, act_type=1, related_acts=None, exclusive_obj_idxs=None):
    return {
        "act_idx": verb_idx,
        "obj_idxs": [obj_idxs or [], exclusive_obj_idxs or []],
        "act_type": act_type,
        "related_acts": related_acts or [],
    }


def sample(acts, pred, words=None, sents=None, **extra):
    data = {
        "words": words or WORDS,
        "acts": acts,
        "pred": pred,
        "sents": sents or [["choose", "a", "square", "shadow", "box"]],
    }
    data.update(extra)
    return data


def assert_close_tuple(actual, expected):
    if len(expected) == 6 and len(actual) >= 9:
        expected = (*expected, *expected[3:6])
        actual = actual[:9]
    elif len(actual) > len(expected):
        actual = actual[:len(expected)]
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-9)


def test_evaluation_reports_raw_event_counts():
    data = [
        sample(
            [act(5, [7])],
            [{"verb": "open", "arguments": ["file"]}],
            sents=[["open", "the", "file"]],
        ),
        sample(
            [act(8, [10])],
            [{"verb": "save", "arguments": []}],
            sents=[["save", "as", "PDF"]],
        ),
    ]

    metrics = ev.evaluation(data)

    assert len(metrics) == 12
    assert metrics[9:] == (
        1,  # perfect_action_argument_matches
        1,  # argument_mismatch_actions
        2,  # matched_action_events
    )


def test_parse_result_filename_handles_solver_and_model_underscores():
    assert ev.parse_result_filename("cooking_nl2p_1_gpt-5-mini.pkl") == (
        "cooking",
        "nl2p_1",
        "gpt-5-mini",
    )
    assert ev.parse_result_filename("cooking_nl2p_1_ablation_gpt-5-mini.pkl") == (
        "cooking",
        "nl2p_1_ablation",
        "gpt-5-mini",
    )
    assert ev.parse_result_filename("cooking_nl2p_1_coref_gpt-5-mini.pkl") == (
        "cooking",
        "nl2p_1_coref",
        "gpt-5-mini",
    )
    assert ev.parse_result_filename("cooking_nl2p_1_ablation_coref_gpt-5-mini.pkl") == (
        "cooking",
        "nl2p_1_ablation_coref",
        "gpt-5-mini",
    )
    assert ev.parse_result_filename("wikihow_gpt3_to_plan_gpt-4.1.pkl") == (
        "wikihow",
        "gpt3_to_plan",
        "gpt-4.1",
    )
    assert ev.parse_result_filename("wikihow_gpt3_to_plan_reparsed_gpt-5.4.pkl") == (
        "wikihow",
        "gpt3_to_plan_reparsed",
        "gpt-5.4",
    )
    assert ev.parse_result_filename("win2k_verb_args_gemma3_12b.pkl") == (
        "win2k",
        "verb_args",
        "gemma3_12b",
    )


def test_match_obj_handles_lemma_phrase_and_empty_input():
    assert ev.match_obj("boxes", "box")
    assert ev.match_obj("box", "square shadow box")
    assert not ev.match_obj("", "box")
    assert not ev.match_obj("box", "")


def test_match_obj_rejects_shared_modifier_or_substring_false_positives():
    assert not ev.match_obj("red button", "blue button")
    assert not ev.match_obj("cream cheese", "cream sauce")
    assert not ev.match_obj("file", "profile")


def test_argument_match_score_uses_one_way_gold_token_containment():
    assert ev.argument_match_type("box", "square shadow box") == "token_containment"
    assert ev.argument_match_type("square shadow box", "box") == ""
    assert ev.argument_match_type("red button", "blue button") == ""
    assert ev.argument_match_type("cream cheese", "cream sauce") == ""


def test_argument_lemmas_are_noun_biased_for_plural_objects():
    assert ev.normalized_argument_text("leaves") == "leaf"
    assert ev.argument_head_lemma("bay leaves") == "leaf"
    assert ev.content_lemmas("bay leaves") == {"bay", "leaf"}
    assert ev.match_obj("leaves", "leaf")
    assert ev.match_obj("bay leaves", "bay leaf")
    assert ev.lemma_text("leaves") == "leave"


def test_match_obj_accepts_complete_gold_tokens_inside_predicted_phrases():
    assert ev.match_obj("box", "square shadow box")
    assert not ev.match_obj("square shadow box", "box")
    assert ev.match_obj("square shadow box", "shadow square box")
    assert ev.match_obj("file", "open file")
    assert ev.match_obj("file", "file name")
    assert not ev.match_obj("source file", "target file")
    assert ev.match_obj("mushroom soup", "cream mushroom soup")
    assert not ev.match_obj("file", "profile")


def test_match_objs_scores_partial_extra_and_empty_predictions():
    obj_right, obj_true, obj_tagged, obj_f1 = ev.match_objs([["box"], []], ["square shadow box"])
    assert (obj_right, obj_true, obj_tagged) == (1, 1, 1)
    assert math.isclose(obj_f1, 1.0)

    obj_right, obj_true, obj_tagged, obj_f1 = ev.match_objs([["box"], []], ["square", "shadow", "box"])
    assert (obj_right, obj_true, obj_tagged) == (1, 1, 3)
    assert math.isclose(obj_f1, 0.5)

    assert ev.match_objs([["box"], []], []) == (0, 1, 0, 0)


def test_match_objs_treats_string_arguments_as_single_argument():
    assert ev.match_objs([["file"], []], "file") == (1, 1, 1, 1.0)


def test_match_objs_treats_exclusive_arguments_as_alternatives_not_extras():
    assert ev.match_objs([["switch"], ["button"]], ["button"]) == (1, 1, 1, 1.0)
    assert ev.match_objs([["switch"], ["button"]], ["switch", "button"]) == (1, 1, 1, 1.0)

    obj_right, obj_true, obj_tagged, obj_f1 = ev.match_objs(
        [["switch"], ["button"]],
        ["switch", "button", "panel"],
    )

    assert (obj_right, obj_true, obj_tagged) == (1, 1, 2)
    assert math.isclose(obj_f1, 2 / 3)


def test_arg_diff_uses_strict_one_to_one_matching():
    missing, extra = ev.arg_diff(["file", "target folder"], ["folder", "file"])

    assert missing == ["target folder"]
    assert extra == ["folder"]


def test_arg_diff_does_not_match_head_mismatch_or_modifier_conflict():
    missing, extra = ev.arg_diff(["file", "source file"], ["file name", "target file"])

    assert missing == ["source file"]
    assert extra == ["file name"]


def test_arg_diff_keeps_modifier_conflict_as_missing_and_extra():
    missing, extra = ev.arg_diff(["red button"], ["blue button"])

    assert missing == ["red button"]
    assert extra == ["blue button"]


def test_match_accepts_verb_phrase_and_rejects_missing_or_wrong_verb():
    matched, obj_right, obj_true, obj_tagged, obj_f1 = ev.match(
        act(0, [4]),
        {"verb": "choose carefully", "arguments": ["square shadow box"]},
        WORDS,
    )
    assert matched
    assert (obj_right, obj_true, obj_tagged) == (1, 1, 1)
    assert math.isclose(obj_f1, 1.0)

    matched, obj_right, obj_true, obj_tagged, obj_f1 = ev.match(
        {"act_idx": 0, "obj_idxs": [[2], []]},
        {"verb": "add in", "arguments": ["hash"]},
        ["add", "in", "hash"],
    )
    assert matched
    assert (obj_right, obj_true, obj_tagged) == (1, 1, 1)
    assert math.isclose(obj_f1, 1.0)

    assert ev.match(act(0, [4]), {"arguments": ["box"]}, WORDS) == (False, 0, 0, 0, 0)
    assert ev.match(act(0, [4]), {"verb": "open", "arguments": ["box"]}, WORDS) == (False, 0, 0, 0, 0)


def test_action_matching_uses_complete_lemma_tokens_not_character_substrings():
    words = ["press", "button"]

    assert ev.match_action(act(0, [1]), {"verb": "press down"}, words)
    assert not ev.match_action(act(0, [1]), {"verb": "surpress"}, words)
    assert not ev.match_action(act(0, [1]), {"verb": "compress"}, words)


def test_match_handles_bad_indices_missing_obj_idxs_and_string_arguments():
    assert ev.match({"act_idx": 999, "obj_idxs": [[4], []]}, {"verb": "choose", "arguments": ["box"]}, WORDS) == (
        False,
        0,
        0,
        0,
        0,
    )
    assert ev.match({"act_idx": 0}, {"verb": "choose", "arguments": []}, WORDS) == (True, 0, 0, 0, 0)
    assert ev.match({"act_idx": 0, "obj_idxs": [[4, 999], []]}, {"verb": "choose", "arguments": "box"}, WORDS) == (
        True,
        1,
        1,
        1,
        1.0,
    )


def test_action_record_ignores_empty_marker_and_bad_indices():
    assert ev.action_record(act(0, [-1]), WORDS)["arguments"] == []
    assert ev.action_record(act(99, [4, 99]), WORDS) == {
        "verb": "",
        "arguments": ["box"],
        "act_idx": 99,
        "obj_idxs": [[4, 99], []],
        "act_type": 1,
        "related_acts": [],
    }


def test_classify_argument_mismatch_missing_extra_wrong_and_subtypes():
    missing = ev.classify_argument_mismatch(["box"], [])
    assert missing["candidate_dataset_issue"] == ""
    assert missing["candidate_llm_issue"] == "missing_arguments"

    extra = ev.classify_argument_mismatch([], ["box"])
    assert extra["candidate_dataset_issue"] == ""
    assert extra["candidate_llm_issue"] == "extra_arguments"

    wrong = ev.classify_argument_mismatch(["file"], ["folder"])
    assert wrong["candidate_dataset_issue"] == ""
    assert "wrong_arguments" in wrong["candidate_llm_issue"]

    dataset_missing = ev.classify_argument_mismatch([], ["box"], source_text="choose the box")
    assert dataset_missing["candidate_dataset_issue"] == ""
    assert dataset_missing["candidate_llm_issue"] == "extra_arguments"

    preposition = ev.classify_argument_mismatch([], ["to folder"])
    assert "extra_arguments:preposition_argument" in preposition["candidate_llm_issue"]

    split = ev.classify_argument_mismatch(["square shadow box"], ["square", "shadow", "box"])
    assert split["candidate_llm_issue"] == "missing_arguments|extra_arguments|wrong_arguments"


def test_best_verb_candidate_prefers_overlap():
    candidate, score = ev.best_verb_candidate(
        {"verb": "open", "arguments": ["file"]},
        [
            (0, {"verb": "close", "arguments": ["window"]}),
            (1, {"verb": "open", "arguments": ["document file"]}),
        ],
    )
    assert candidate == (1, {"verb": "open", "arguments": ["document file"]})
    assert score > 0

    assert ev.best_verb_candidate({"verb": "open", "arguments": []}, [(0, {"verb": "save", "arguments": []})]) == (
        None,
        0,
    )


def test_evaluation_exact_match_metrics_have_no_diagnostics():
    data = [
        sample(
            [act(0, [4])],
            [{"verb": "choose", "arguments": ["square shadow box"]}],
            doc_id=7,
            docId="cooking:7",
        )
    ]

    metrics, diagnostics = ev.evaluation(data, names=("cooking", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (1, 1, 1, 1, 1, 1))
    assert diagnostics == []


def test_evaluation_argument_mismatch_diagnoses_extra_llm_arguments():
    data = [
        sample(
            [act(0, [4])],
            [{"verb": "choose", "arguments": ["square", "shadow", "box"]}],
            doc_id=118,
        )
    ]

    metrics, diagnostics = ev.evaluation(data, names=("cooking", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (1, 1, 1, 1 / 3, 1, 0.5))
    assert len(diagnostics) == 1
    row = diagnostics[0]
    assert row["docId"] == "cooking:118"
    assert row["mismatch_type"] == "argument_mismatch"
    assert row["candidate_llm_issue"] == "extra_arguments"
    assert row["candidate_dataset_issue"] == ""
    assert row["gold_verb"] == "choose"
    assert row["pred_verb"] == "choose"


def test_evaluation_argument_mismatch_diagnoses_dataset_split_head_words():
    data = [
        sample(
            [act(0, [2, 3, 4])],
            [{"verb": "choose", "arguments": ["square shadow box"]}],
            doc_id=119,
        )
    ]

    metrics, diagnostics = ev.evaluation(data, names=("cooking", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (1, 1, 1, 1, 1 / 3, 0.5, 1, 1, 1))
    assert len(diagnostics) == 1
    row = diagnostics[0]
    assert row["mismatch_type"] == "argument_mismatch"
    assert row["candidate_dataset_issue"] == "extra_arguments|extra_arguments:unnecessary_head_or_modifier_split"
    assert row["strong_dataset_issue"] == "extra_arguments|extra_arguments:unnecessary_head_or_modifier_split"
    assert row["dataset_issue_confidence"] == "strong"
    assert row["candidate_llm_issue"] == ""
    assert row["gold_arguments"] == '["square", "shadow", "box"]'
    assert row["pred_arguments"] == '["square shadow box"]'
    assert row["missing_gold_args"] == '["square", "shadow"]'
    assert row["extra_pred_args"] == "[]"


def test_evaluation_argument_mismatch_diagnoses_dataset_split_when_llm_outputs_head_only():
    data = [
        sample(
            [act(0, [2, 3, 4])],
            [{"verb": "choose", "arguments": ["box"]}],
            doc_id=120,
        )
    ]

    metrics, diagnostics = ev.evaluation(data, names=("cooking", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (1, 1, 1, 1, 1 / 3, 0.5, 1, 1, 1))
    assert len(diagnostics) == 1
    row = diagnostics[0]
    assert row["mismatch_type"] == "argument_mismatch"
    assert row["candidate_dataset_issue"] == "extra_arguments|extra_arguments:unnecessary_head_or_modifier_split"
    assert row["candidate_llm_issue"] == ""
    assert row["gold_arguments"] == '["square", "shadow", "box"]'
    assert row["pred_arguments"] == '["box"]'


def test_classify_argument_mismatch_flags_gt_split_with_pred_head_only():
    info = ev.classify_argument_mismatch(["square", "shadow", "box"], ["box"])

    assert info["missing_from_pred"] == ["square", "shadow"]
    assert info["extra_in_pred"] == []
    assert "extra_arguments:unnecessary_head_or_modifier_split" in info["candidate_dataset_issue"]
    assert info["candidate_llm_issue"] == ""


def test_classify_argument_mismatch_does_not_call_single_gold_arg_a_split():
    info = ev.classify_argument_mismatch(["lemon"], ["lemon juice"])

    assert info["missing_from_pred"] == []
    assert info["extra_in_pred"] == []
    assert info["candidate_dataset_issue"] == ""
    assert "extra_arguments:unnecessary_head_or_modifier_split" not in info["candidate_llm_issue"]
    assert info["candidate_llm_issue"] == ""


def test_classify_argument_mismatch_flags_gt_split_with_other_arguments_present():
    phrase = ev.classify_argument_mismatch(["lemon", "juice", "bowl"], ["lemon juice", "bowl"])
    assert phrase["missing_from_pred"] == ["lemon"]
    assert phrase["extra_in_pred"] == []
    assert "extra_arguments:unnecessary_head_or_modifier_split" in phrase["candidate_dataset_issue"]
    assert phrase["candidate_llm_issue"] == ""

    head_only = ev.classify_argument_mismatch(["lemon", "juice", "bowl"], ["juice", "bowl"])
    assert head_only["missing_from_pred"] == ["lemon"]
    assert head_only["extra_in_pred"] == []
    assert "extra_arguments:unnecessary_head_or_modifier_split" in head_only["candidate_dataset_issue"]
    assert head_only["candidate_llm_issue"] == ""


def test_classify_argument_mismatch_does_not_infer_gt_split_from_unrelated_head():
    info = ev.classify_argument_mismatch(["lemon", "bowl"], ["juice"])

    assert info["candidate_dataset_issue"] == ""
    assert "extra_arguments:unnecessary_head_or_modifier_split" not in info["candidate_llm_issue"]
    assert "wrong_arguments" in info["candidate_llm_issue"]


def test_classify_argument_mismatch_flags_gold_pronoun_or_generic_reference():
    pronoun = ev.classify_argument_mismatch(["it"], ["batter"])

    assert pronoun["candidate_dataset_issue"] == (
        "wrong_arguments|wrong_arguments:gold_pronoun_or_generic_reference"
    )
    assert pronoun["candidate_llm_issue"] == ""

    generic = ev.classify_argument_mismatch(
        ["everything"],
        ["everything", "salt"],
    )

    assert generic["candidate_dataset_issue"] == (
        "wrong_arguments|wrong_arguments:gold_pronoun_or_generic_reference"
    )
    assert generic["candidate_llm_issue"] == ""


def test_classify_argument_mismatch_flags_unmatched_gold_generic_reference_without_extra_pred():
    info = ev.classify_argument_mismatch(["it", "salt"], ["salt"])

    assert info["missing_from_pred"] == ["it"]
    assert info["extra_in_pred"] == []
    assert info["candidate_dataset_issue"] == (
        "wrong_arguments|wrong_arguments:gold_pronoun_or_generic_reference"
    )
    assert info["candidate_llm_issue"] == ""


def test_classify_argument_mismatch_keeps_generic_missing_as_llm_when_no_concrete_prediction():
    info = ev.classify_argument_mismatch(["it"], [])

    assert info["candidate_dataset_issue"] == ""
    assert info["candidate_llm_issue"] == "missing_arguments"


def test_classify_argument_mismatch_keeps_preposition_object_priority_over_generic_reference():
    info = ev.classify_argument_mismatch(
        ["it"],
        ["salt"],
        source_text="Season it with salt.",
    )

    assert "missing_arguments:preposition_object" in info["candidate_dataset_issue"]
    assert "gold_pronoun_or_generic_reference" not in info["candidate_dataset_issue"]
    assert info["strong_dataset_issue"] == info["candidate_dataset_issue"]
    assert info["dataset_issue_confidence"] == "strong"
    assert info["candidate_llm_issue"] == ""


def test_classify_argument_mismatch_does_not_keep_weak_gold_missing_valid_argument():
    info = ev.classify_argument_mismatch([], ["batter"], source_text="Bake the batter for 45 minutes")

    assert info["candidate_dataset_issue"] == ""
    assert info["strong_dataset_issue"] == ""
    assert info["dataset_issue_confidence"] == ""
    assert info["candidate_llm_issue"] == "extra_arguments"


def test_classify_argument_mismatch_does_not_flag_non_entity_missing_valid_argument():
    info = ev.classify_argument_mismatch([], ["aside"], source_text="Set aside for 10 minutes")

    assert info["candidate_dataset_issue"] == ""
    assert info["candidate_llm_issue"] == "extra_arguments"


def test_classify_argument_mismatch_does_not_flag_adjectival_or_verbal_missing_valid_argument():
    adjective = ev.classify_argument_mismatch([], ["golden brown"], source_text="Cook until golden brown.")
    verbal = ev.classify_argument_mismatch([], ["running"], source_text="Keep running.")

    assert adjective["candidate_dataset_issue"] == ""
    assert adjective["candidate_llm_issue"] == "extra_arguments"
    assert verbal["candidate_dataset_issue"] == ""
    assert verbal["candidate_llm_issue"] == "extra_arguments"


def test_classify_argument_mismatch_requires_missing_valid_argument_to_appear_in_text():
    info = ev.classify_argument_mismatch([], ["batter"], source_text="Bake for 45 minutes.")

    assert info["candidate_dataset_issue"] == ""
    assert info["candidate_llm_issue"] == "extra_arguments"


def test_evaluation_argument_mismatch_diagnoses_llm_missing_argument():
    data = [sample([act(5, [7])], [{"verb": "open", "arguments": []}], sents=[["open", "the", "file"]])]

    metrics, diagnostics = ev.evaluation(data, names=("win2k", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (1, 1, 1, 0, 0, 0))
    assert len(diagnostics) == 1
    row = diagnostics[0]
    assert row["mismatch_type"] == "argument_mismatch"
    assert row["candidate_dataset_issue"] == ""
    assert row["candidate_llm_issue"] == "missing_arguments"
    assert row["reason"] == "gold has arguments not matched by prediction"


def test_evaluation_argument_mismatch_keeps_weak_missing_argument_as_llm_extra():
    data = [sample([act(5, [])], [{"verb": "open", "arguments": ["file"]}], sents=[["open", "the", "file"]])]

    metrics, diagnostics = ev.evaluation(data, names=("win2k", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (1, 1, 1, 0, 0, 0))
    assert len(diagnostics) == 1
    row = diagnostics[0]
    assert row["mismatch_type"] == "argument_mismatch"
    assert row["candidate_dataset_issue"] == ""
    assert row["strong_dataset_issue"] == ""
    assert row["candidate_llm_issue"] == "extra_arguments"


def test_evaluation_argument_mismatch_diagnoses_dataset_missing_preposition_object():
    words = ["put", "food"]
    data = [
        sample(
            [act(0, [1])],
            [{"verb": "put", "arguments": ["food", "bowl"]}],
            words=words,
            sents=[["put", "food", "into", "the", "red", "bowl"]],
        )
    ]

    metrics, diagnostics = ev.evaluation(data, names=("cooking", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (1, 1, 1, 0.5, 1, 2 / 3, 1, 1, 1))
    assert len(diagnostics) == 1
    row = diagnostics[0]
    assert row["mismatch_type"] == "argument_mismatch"
    assert row["candidate_dataset_issue"] == "missing_arguments|missing_arguments:preposition_object"
    assert row["strong_dataset_issue"] == "missing_arguments|missing_arguments:preposition_object"
    assert row["dataset_issue_confidence"] == "strong"
    assert row["candidate_llm_issue"] == ""
    assert row["missing_gold_args"] == "[]"
    assert row["extra_pred_args"] == '["bowl"]'


def test_adjusted_metric_only_filters_preposition_extra_pred_args():
    words = ["put", "food"]
    data = [
        sample(
            [act(0, [1])],
            [{"verb": "put", "arguments": ["food", "bowl", "timer"]}],
            words=words,
            sents=[["put", "food", "into", "the", "red", "bowl"]],
        )
    ]

    metrics, diagnostics = ev.evaluation(data, names=("cooking", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (1, 1, 1, 1 / 3, 1, 0.5, 0.5, 1, 2 / 3))
    assert len(diagnostics) == 1
    assert diagnostics[0]["extra_pred_args"] == '["bowl", "timer"]'


def test_evaluation_argument_mismatch_diagnoses_dataset_missing_with_object():
    words = ["make", "sauce"]
    data = [
        sample(
            [act(0, [1])],
            [{"verb": "make", "arguments": ["sauce", "butter"]}],
            words=words,
            sents=[["make", "sauce", "with", "butter"]],
        )
    ]

    _, diagnostics = ev.evaluation(data, names=("cooking", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert len(diagnostics) == 1
    row = diagnostics[0]
    assert row["candidate_dataset_issue"] == "missing_arguments|missing_arguments:preposition_object"
    assert row["strong_dataset_issue"] == "missing_arguments|missing_arguments:preposition_object"
    assert row["candidate_llm_issue"] == ""


def test_evaluation_argument_mismatch_diagnoses_dataset_extra_preposition_object():
    words = ["put", "food", "bowl"]
    data = [
        sample(
            [act(0, [1, 2])],
            [{"verb": "put", "arguments": ["food"]}],
            words=words,
            sents=[["put", "food", "into", "the", "red", "bowl"]],
        )
    ]

    metrics, diagnostics = ev.evaluation(data, names=("cooking", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (1, 1, 1, 1, 0.5, 2 / 3, 1, 1, 1))
    assert len(diagnostics) == 1
    row = diagnostics[0]
    assert row["mismatch_type"] == "argument_mismatch"
    assert row["candidate_dataset_issue"] == "extra_arguments|extra_arguments:preposition_object"
    assert row["strong_dataset_issue"] == "extra_arguments|extra_arguments:preposition_object"
    assert row["candidate_llm_issue"] == ""
    assert row["reason"] == (
        "gold has arguments not matched by prediction; "
        "gold unmatched argument is a preposition object in source text"
    )
    assert row["missing_gold_args"] == '["bowl"]'
    assert row["extra_pred_args"] == "[]"


def test_adjusted_metric_only_filters_preposition_missing_gold_args():
    words = ["put", "food", "bowl", "timer"]
    data = [
        sample(
            [act(0, [1, 2, 3])],
            [{"verb": "put", "arguments": ["food"]}],
            words=words,
            sents=[["put", "food", "into", "the", "red", "bowl"]],
        )
    ]

    metrics, diagnostics = ev.evaluation(data, names=("cooking", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (1, 1, 1, 1, 1 / 3, 0.5, 1, 0.5, 2 / 3))
    assert len(diagnostics) == 1
    assert diagnostics[0]["missing_gold_args"] == '["bowl", "timer"]'


def test_evaluation_argument_mismatch_diagnoses_dataset_extra_with_object():
    words = ["make", "sauce", "butter"]
    data = [
        sample(
            [act(0, [1, 2])],
            [{"verb": "make", "arguments": ["sauce"]}],
            words=words,
            sents=[["make", "sauce", "with", "butter"]],
        )
    ]

    _, diagnostics = ev.evaluation(data, names=("cooking", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert len(diagnostics) == 1
    row = diagnostics[0]
    assert row["candidate_dataset_issue"] == "extra_arguments|extra_arguments:preposition_object"
    assert row["strong_dataset_issue"] == "extra_arguments|extra_arguments:preposition_object"
    assert row["candidate_llm_issue"] == ""


def test_evaluation_argument_mismatch_diagnoses_wrong_arguments():
    data = [sample([act(5, [7])], [{"verb": "open", "arguments": ["folder"]}], sents=[["open", "the", "file"]])]

    metrics, diagnostics = ev.evaluation(data, names=("win2k", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (1, 1, 1, 0, 0, 0))
    assert len(diagnostics) == 1
    row = diagnostics[0]
    assert row["mismatch_type"] == "argument_mismatch"
    assert row["candidate_dataset_issue"] == ""
    assert "wrong_arguments" in row["candidate_llm_issue"]
    assert "gold has arguments not matched by prediction" in row["reason"]
    assert "prediction has arguments not matched by gold" in row["reason"]


def test_evaluation_preposition_object_uses_action_local_sentence_for_missing_gold_arg():
    data = [
        sample(
            [act(0, [1])],
            [{"verb": "preheat", "arguments": []}],
            words=["preheat", "oven"],
            sents=[["preheat", "oven", "to", "325oF"], ["remove", "cake", "out", "of", "oven"]],
        )
    ]

    _, diagnostics = ev.evaluation(data, names=("cooking", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert len(diagnostics) == 1
    row = diagnostics[0]
    assert row["candidate_dataset_issue"] == ""
    assert row["strong_dataset_issue"] == ""
    assert row["candidate_llm_issue"] == "missing_arguments"


def test_evaluation_preposition_object_uses_action_local_sentence_for_extra_pred_arg():
    data = [
        sample(
            [act(0, [])],
            [{"verb": "start", "arguments": ["blender"]}],
            words=["start", "blender"],
            sents=[["start", "the", "blender", "again"], ["pour", "fruit", "in", "the", "blender"]],
        )
    ]

    _, diagnostics = ev.evaluation(data, names=("cooking", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert len(diagnostics) == 1
    row = diagnostics[0]
    assert row["candidate_dataset_issue"] == ""
    assert row["strong_dataset_issue"] == ""
    assert row["candidate_llm_issue"] == "extra_arguments"


def test_classify_preposition_object_rejects_leading_pp_fragment_false_positive():
    info = ev.classify_argument_mismatch(
        ["options"],
        [],
        source_text="In control panel double-click internet options.",
        action_verb="double-click",
    )

    assert info["candidate_dataset_issue"] == ""
    assert info["strong_dataset_issue"] == ""
    assert info["candidate_llm_issue"] == "missing_arguments"


def test_classify_argument_mismatch_does_not_call_overlong_prep_span_a_dataset_issue():
    info = ev.classify_argument_mismatch(
        ["fruit"],
        ["fruit chunks in the blender"],
        source_text="Pulse fruit chunks in the blender.",
    )

    assert info["candidate_dataset_issue"] == ""
    assert info["strong_dataset_issue"] == ""
    assert info["candidate_llm_issue"] == ""


def test_classify_argument_mismatch_flags_dataset_preposition_argument():
    info = ev.classify_argument_mismatch(["to folder"], [])

    assert info["candidate_dataset_issue"] == ""
    assert info["candidate_llm_issue"] == "missing_arguments"


def test_evaluation_empty_predictions_count_against_recall_and_diagnose_missing_action():
    data = [sample([act(5, [7])], [], doc_id=3, sents=[["open", "the", "file"]])]

    metrics, diagnostics = ev.evaluation(data, names=("win2k", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (0, 0, 0, 0, 0, 0))
    assert len(diagnostics) == 1
    assert diagnostics[0]["mismatch_type"] == "unmatched_gold_action"
    assert diagnostics[0]["candidate_dataset_issue"] == ""
    assert diagnostics[0]["candidate_llm_issue"] == "missing_actions"


def test_evaluation_unused_prediction_diagnoses_possible_missing_dataset_action():
    data = [
        sample(
            [act(5, [7])],
            [
                {"verb": "open", "arguments": ["file"]},
                {"verb": "save", "arguments": ["file"]},
            ],
            sents=[["open", "the", "file"], ["save", "the", "file"]],
            original_text="open the file. save the file.",
        )
    ]

    metrics, diagnostics = ev.evaluation(data, names=("win2k", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (0.5, 1, 2 / 3, 1, 1, 1))
    assert len(diagnostics) == 1
    assert diagnostics[0]["mismatch_type"] == "unmatched_prediction"
    assert diagnostics[0]["candidate_dataset_issue"] == "missing_actions"
    assert diagnostics[0]["candidate_llm_issue"] == ""


def test_evaluation_unused_prediction_not_in_original_is_llm_extra_only():
    data = [
        sample(
            [act(5, [7])],
            [
                {"verb": "open", "arguments": ["file"]},
                {"verb": "delete", "arguments": ["file"]},
            ],
            sents=[["open", "the", "file"]],
            original_text="open the file.",
        )
    ]

    metrics, diagnostics = ev.evaluation(data, names=("win2k", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (0.5, 1, 2 / 3, 1, 1, 1))
    assert len(diagnostics) == 1
    assert diagnostics[0]["mismatch_type"] == "unmatched_prediction"
    assert diagnostics[0]["candidate_dataset_issue"] == ""
    assert diagnostics[0]["candidate_llm_issue"] == "extra_actions"
    assert diagnostics[0]["reason"] == "unused prediction did not match any gold action"


def test_evaluation_prediction_with_no_gold_actions_is_diagnosed_as_unused_prediction():
    data = [sample([], [{"verb": "save", "arguments": ["file"]}], sents=[["save", "the", "file"]])]

    metrics, diagnostics = ev.evaluation(data, names=("win2k", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (0, 0, 0, 0, 0, 0))
    assert len(diagnostics) == 1
    assert diagnostics[0]["mismatch_type"] == "unmatched_prediction"
    assert diagnostics[0]["candidate_dataset_issue"] == "missing_actions"
    assert diagnostics[0]["candidate_llm_issue"] == ""


def test_evaluation_single_argument_overlap_is_not_enough_for_wrong_action():
    data = [
        sample(
            [act(5, [7])],
            [{"verb": "load", "arguments": ["file"]}],
            sents=[["open", "the", "file"]],
            original_text="open the file.",
        )
    ]

    metrics, diagnostics = ev.evaluation(data, names=("win2k", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (0, 0, 0, 0, 0, 0))
    assert len(diagnostics) == 2
    assert diagnostics[0]["mismatch_type"] == "unmatched_gold_action"
    assert diagnostics[0]["candidate_dataset_issue"] == ""
    assert diagnostics[0]["candidate_llm_issue"] == "missing_actions"
    assert diagnostics[1]["mismatch_type"] == "unmatched_prediction"
    assert diagnostics[1]["candidate_llm_issue"] == "extra_actions"


def test_evaluation_stronger_overlap_diagnoses_wrong_action():
    words = ["open", "source", "file"]
    data = [
        sample(
            [act(0, [1, 2])],
            [{"verb": "load", "arguments": ["source file"]}],
            words=words,
            sents=[["open", "source", "file"]],
            original_text="open source file.",
        )
    ]

    metrics, diagnostics = ev.evaluation(data, names=("win2k", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (0, 0, 0, 0, 0, 0))
    assert len(diagnostics) == 2
    assert diagnostics[0]["mismatch_type"] == "wrong_action"
    assert diagnostics[0]["candidate_llm_issue"] == "wrong_actions"
    assert diagnostics[1]["mismatch_type"] == "unmatched_prediction"


def test_unmatched_gold_diagnostic_does_not_reuse_future_matched_prediction():
    words = ["give", "caesar", "salad", "transfer", "dressing"]
    data = [
        sample(
            [
                act(0, [2]),
                act(3, [4]),
            ],
            [{"verb": "transfer", "arguments": ["dressing"]}],
            words=words,
            sents=[["give", "caesar", "salad"], ["transfer", "dressing"]],
            original_text="give caesar salad. transfer dressing.",
        )
    ]

    metrics, diagnostics = ev.evaluation(data, names=("cooking", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (1, 0.5, 2 / 3, 1, 1, 1))
    assert len(diagnostics) == 1
    assert diagnostics[0]["mismatch_type"] == "unmatched_gold_action"
    assert diagnostics[0]["gold_verb"] == "give"
    assert diagnostics[0]["pred_verb"] == ""


def test_repeated_verbs_use_strict_first_match_order():
    words = ["add", "flavor", "add", "bread", "add", "salt", "pepper"]
    data = [
        sample(
            [
                act(0, [1]),
                act(2, [3]),
                act(4, [5, 6]),
            ],
            [
                {"verb": "Add", "arguments": ["bread"]},
                {"verb": "Add", "arguments": ["salt", "pepper"]},
            ],
            words=words,
            sents=[["add", "flavor"], ["add", "bread"], ["add", "salt", "pepper"]],
            original_text="add flavor. add bread. add salt and pepper.",
        )
    ]

    metrics, diagnostics = ev.evaluation(data, names=("cooking", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (1, 2 / 3, 0.8, 0, 0, 0))
    assert len(diagnostics) == 3
    assert diagnostics[0]["mismatch_type"] == "argument_mismatch"
    assert diagnostics[0]["gold_arguments"] == '["flavor"]'
    assert diagnostics[0]["pred_arguments"] == '["bread"]'
    assert diagnostics[1]["mismatch_type"] == "argument_mismatch"
    assert diagnostics[1]["gold_arguments"] == '["bread"]'
    assert diagnostics[1]["pred_arguments"] == '["salt", "pepper"]'
    assert diagnostics[2]["mismatch_type"] == "unmatched_gold_action"
    assert diagnostics[2]["gold_arguments"] == '["salt", "pepper"]'


def test_evaluation_exclusive_actions_count_truth_once():
    words = ["turn", "switch", "press", "button"]
    data = [
        sample(
            [
                act(0, [1], act_type=3, related_acts=[2]),
                act(2, [3], act_type=3, related_acts=[0]),
            ],
            [{"verb": "turn", "arguments": ["switch"]}],
            words=words,
            sents=[["turn", "switch"], ["or", "press", "button"]],
        )
    ]

    metrics, diagnostics = ev.evaluation(data, names=("win2k", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (1, 1, 1, 1, 1, 1))
    assert diagnostics == []


def test_evaluation_exclusive_actions_count_additional_group_predictions_as_fp():
    words = ["turn", "switch", "press", "button"]
    data = [
        sample(
            [
                act(0, [1], act_type=3, related_acts=[2]),
                act(2, [3], act_type=3, related_acts=[0]),
            ],
            [
                {"verb": "turn", "arguments": ["switch"]},
                {"verb": "press", "arguments": ["button"]},
            ],
            words=words,
            sents=[["turn", "switch"], ["or", "press", "button"]],
        )
    ]

    metrics, diagnostics = ev.evaluation(data, names=("win2k", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (0.5, 1, 2 / 3, 1, 1, 1))
    assert len(diagnostics) == 1
    assert diagnostics[0]["mismatch_type"] == "unmatched_prediction"


def test_evaluation_exclusive_first_match_does_not_rank_by_object_f1():
    words = ["turn", "switch", "press", "button"]
    data = [
        sample(
            [
                act(0, [1], act_type=3, related_acts=[2]),
                act(2, [3], act_type=3, related_acts=[0]),
            ],
            [
                {"verb": "turn", "arguments": ["wrong object"]},
                {"verb": "press", "arguments": ["button"]},
            ],
            words=words,
            sents=[["turn", "switch"], ["or", "press", "button"]],
        )
    ]

    metrics, diagnostics = ev.evaluation(
        data,
        names=("win2k", "nl2p_1", "gpt-5-mini"),
        collect_diagnostics=True,
    )

    assert_close_tuple(metrics, (0.5, 1, 2 / 3, 0, 0, 0))
    assert [row["mismatch_type"] for row in diagnostics] == [
        "argument_mismatch",
        "unmatched_prediction",
    ]


def test_evaluation_optional_action_can_be_omitted_without_diagnostic_or_fn():
    words = ["open", "file", "preview", "document"]
    data = [
        sample(
            [
                act(0, [1], act_type=1),
                act(2, [3], act_type=2),
            ],
            [{"verb": "open", "arguments": ["file"]}],
            words=words,
            sents=[["open", "file"], ["preview", "document"]],
        )
    ]

    metrics, diagnostics = ev.evaluation(data, names=("win2k", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (1, 1, 1, 1, 1, 1))
    assert diagnostics == []


def test_evaluation_optional_action_matched_adds_truth_denominator():
    words = ["open", "file", "preview", "document"]
    data = [
        sample(
            [
                act(0, [1], act_type=1),
                act(2, [3], act_type=2),
            ],
            [
                {"verb": "open", "arguments": ["file"]},
                {"verb": "preview", "arguments": ["document"]},
            ],
            words=words,
            sents=[["open", "file"], ["preview", "document"]],
        )
    ]

    metrics = ev.evaluation(data)

    assert_close_tuple(metrics, (1, 1, 1, 1, 1, 1))


def test_evaluation_optional_action_scores_arguments_only_when_action_is_extracted():
    words = ["preview", "document"]
    omitted = [sample([act(0, [1], act_type=2)], [], words=words)]
    extracted_with_wrong_object = [
        sample(
            [act(0, [1], act_type=2)],
            [{"verb": "preview", "arguments": ["file"]}],
            words=words,
        )
    ]

    omitted_metrics, omitted_diagnostics = ev.evaluation(
        omitted,
        names=("win2k", "nl2p_1", "gpt-5-mini"),
        collect_diagnostics=True,
    )
    extracted_metrics, extracted_diagnostics = ev.evaluation(
        extracted_with_wrong_object,
        names=("win2k", "nl2p_1", "gpt-5-mini"),
        collect_diagnostics=True,
    )

    # EASDRL has optional actions, but no optional-object annotation type.
    # Omitted optional actions contribute neither action nor object truth.
    assert_close_tuple(omitted_metrics, (0, 0, 0, 0, 0, 0))
    assert omitted_diagnostics == []

    # Once the optional action is extracted, it is a normal matched action and
    # its essential/exclusive arguments are evaluated normally.
    assert_close_tuple(extracted_metrics, (1, 1, 1, 0, 0, 0))
    assert len(extracted_diagnostics) == 1
    assert extracted_diagnostics[0]["mismatch_type"] == "argument_mismatch"


def test_evaluation_nonmatching_prediction_is_not_excused_by_optional_gold_action():
    words = ["preview", "document"]
    data = [
        sample(
            [act(0, [1], act_type=2)],
            [{"verb": "print", "arguments": ["document"]}],
            words=words,
        )
    ]

    metrics, diagnostics = ev.evaluation(
        data,
        names=("win2k", "nl2p_1", "gpt-5-mini"),
        collect_diagnostics=True,
    )

    assert_close_tuple(metrics, (0, 0, 0, 0, 0, 0))
    assert len(diagnostics) == 1
    assert diagnostics[0]["mismatch_type"] == "unmatched_prediction"


def test_evaluation_handles_none_predictions_like_empty_predictions():
    data = [sample([act(5, [7])], None, doc_id=4, sents=[["open", "the", "file"]])]

    metrics, diagnostics = ev.evaluation(data, names=("win2k", "nl2p_1", "gpt-5-mini"), collect_diagnostics=True)

    assert_close_tuple(metrics, (0, 0, 0, 0, 0, 0))
    assert len(diagnostics) == 1
    assert diagnostics[0]["candidate_llm_issue"] == "missing_actions"


def test_run_evaluation_collects_metrics_and_diagnostics():
    data = [sample([act(5, [7])], [], doc_id=5, sents=[["open", "the", "file"]])]

    results, diagnostics = ev.run_evaluation({("win2k", "nl2p_1", "gpt-5-mini"): data}, collect_diagnostics=True)

    assert_close_tuple(results[("win2k", "nl2p_1", "gpt-5-mini")], (0, 0, 0, 0, 0, 0))
    assert len(diagnostics) == 1
    assert diagnostics[0]["docId"] == "win2k:5"


def test_diagnostic_row_falls_back_to_words_when_text_metadata_missing():
    row = ev.diagnostic_row(
        ("win2k", "nl2p_1", "gpt-5-mini"),
        {"words": ["open", "file"], "doc_id": 9},
        9,
        "unmatched_gold_action",
        gold={"verb": "open", "arguments": ["file"]},
    )

    assert row["docId"] == "win2k:9"
    assert row["original_text"] == "open file"
    assert row["gold_arguments"] == '["file"]'


def test_diagnostic_row_preserves_existing_docid_original_text_and_source_file():
    row = ev.diagnostic_row(
        ("cooking", "nl2p_1", "gpt-5-mini"),
        {
            "words": ["choose", "box"],
            "doc_id": 10,
            "docId": "custom-doc",
            "original_text": "choose the box.",
            "source_file": "data/easdrl/cooking_labeled_text_data.pkl",
        },
        10,
        "argument_mismatch",
        pred={"verb": "choose", "arguments": ["box"]},
    )

    assert row["docId"] == "custom-doc"
    assert row["original_text"] == "choose the box."
    assert row["source_file"] == "data/easdrl/cooking_labeled_text_data.pkl"
    assert row["pred_arguments"] == '["box"]'


def test_write_diagnostics_empty_input_does_not_create_csv(tmp_path):
    ev.write_diagnostics([], str(tmp_path))

    assert not (tmp_path / "evaluation_mismatch_diagnostics.csv").exists()


def test_write_diagnostics_outputs_expected_csv_columns(tmp_path):
    rows = [
        ev.diagnostic_row(
            ("cooking", "nl2p_1", "gpt-5-mini"),
            sample([act(0, [4])], [], doc_id=1, source_file="data/easdrl/cooking.pkl"),
            1,
            "unmatched_gold_action",
            gold={"verb": "choose", "arguments": ["box"]},
            dataset_issue="extra_actions",
            llm_issue="missing_actions",
            reason="test reason",
        )
    ]

    ev.write_diagnostics(rows, str(tmp_path))

    out = tmp_path / "evaluation_mismatch_diagnostics.csv"
    assert out.exists()
    with out.open(encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))
    assert len(records) == 1
    assert records[0]["docId"] == "cooking:1"
    assert records[0]["candidate_llm_issue"] == "missing_actions"
    assert "strong_dataset_issue" in records[0]
    assert "dataset_issue_confidence" in records[0]
    assert "original_text" in records[0]
    assert "missing_gold_args" in records[0]
    assert "extra_pred_args" in records[0]
