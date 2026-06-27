"""Shared helpers for action-extraction evaluation.

This module is intentionally importable from both the command-line evaluator
(`evaluation.py`) and tests.  The helpers cover three related concerns:

* parsing experiment result filenames into dataset/solver/model names;
* matching predicted action arguments against EASDRL gold arguments; and
* writing heuristic mismatch diagnostics for later notebook review.

The diagnostic labels are candidates, not ground truth.  They are designed to
separate likely annotation issues from likely model issues so that reviewers can
prioritize rows in `evaluation_mismatch_diagnostics.csv`.
"""

import csv
import json
import os

import spacy


nlp = spacy.load("en_core_web_sm")

DATASETS = ("cooking", "wikihow", "win2k")
SOLVERS = ("gpt3_to_plan", "nl2p_1_ablation", "nl2p_1", "nl2p_2", "nl2p_3", "verb_args")

PREPOSITIONS = {
    "about", "above", "across", "after", "against", "along", "among", "around",
    "at", "before", "behind", "below", "beneath", "beside", "between", "by",
    "down", "during", "for", "from", "in", "inside", "into", "near", "of",
    "off", "on", "onto", "out", "over", "through", "to", "under", "up", "with",
    "within", "without",
}

GENERIC_REFERENCE_LEMMAS = {
    "anything",
    "everything",
    "it",
    "itself",
    "one",
    "ones",
    "something",
    "that",
    "them",
    "these",
    "they",
    "this",
    "those",
}

ARGUMENT_MATCH_RANK = {
    "exact": 4,
    "lemma_exact": 3,
    "modifier_exact": 2,
    "head_expansion": 1,
}


def parse_result_filename(file):
    """Return `(dataset, solver, model)` parsed from a result pickle filename.

    Solver names such as `gpt3_to_plan` and `nl2p_1` contain underscores, so a
    simple `stem.split("_")` would misclassify the model name.  Known dataset
    and solver prefixes are tried first; the final split-based path is a
    defensive fallback for older or ad hoc result files.
    """
    stem = os.path.splitext(file)[0]
    for ds_name in DATASETS:
        prefix = ds_name + "_"
        if not stem.startswith(prefix):
            continue
        rest = stem[len(prefix):]
        for solver_name in sorted(SOLVERS, key=len, reverse=True):
            if rest == solver_name:
                return ds_name, solver_name, "none"
            solver_prefix = solver_name + "_"
            if rest.startswith(solver_prefix):
                return ds_name, solver_name, rest[len(solver_prefix):] or "none"
    parts = stem.split("_")
    ds_name = parts[0]
    solver_name = "_".join(parts[1:-1]) if len(parts) > 2 else (parts[1] if len(parts) > 1 else "unknown")
    model_name = parts[-1] if len(parts) > 2 else "none"
    return ds_name, solver_name, model_name


def write_diagnostics(diagnostics: list, dir: str):
    """Write mismatch diagnostic rows to `evaluation_mismatch_diagnostics.csv`.

    Rows are normalized to a fixed field list so callers can pass partial
    dictionaries without breaking downstream CSV readers.  Empty diagnostics are
    treated as a no-op because exact-match runs should not create an empty file.
    """
    if not diagnostics:
        print("No mismatch diagnostics to write.")
        return
    if not os.path.exists(dir):
        os.makedirs(dir)
    outpath = os.path.join(dir, "evaluation_mismatch_diagnostics.csv")
    fieldnames = [
        "dataset", "solver", "model", "doc_id", "docId", "source_file",
        "mismatch_type", "candidate_dataset_issue", "candidate_llm_issue",
        "strong_dataset_issue", "dataset_issue_confidence", "reason",
        "original_text", "gold_verb", "gold_arguments", "pred_verb",
        "pred_arguments", "gold_action", "pred_action",
    ]
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in diagnostics:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print("Mismatch diagnostics written to %s" % outpath)


def normalized_argument_text(text):
    """Normalize an argument phrase to lowercase lemmas without spaces/punctuation."""
    return " ".join(
        token.lemma_.lower()
        for token in nlp(str(text))
        if not token.is_space and not token.is_punct
    )


def argument_head_lemma(text):
    """Return the best available head lemma for an argument phrase.

    The preference order is noun chunk root, rightmost noun/proper noun/pronoun,
    rightmost non-stop token, then the final token.  This keeps short UI labels
    and parser-unfriendly fragments usable while still rejecting unrelated heads.
    """
    doc = nlp(str(text))
    if not doc:
        return ""
    try:
        chunks = list(doc.noun_chunks)
    except ValueError:
        chunks = []
    if chunks:
        return chunks[-1].root.lemma_.lower()
    for token in reversed(doc):
        if token.pos_ in {"NOUN", "PROPN", "PRON"}:
            return token.lemma_.lower()
    for token in reversed(doc):
        if not token.is_space and not token.is_punct and not token.is_stop:
            return token.lemma_.lower()
    return doc[-1].lemma_.lower()


def argument_match_type(left, right):
    """Classify how two argument strings match under the strict object rules.

    Returns one of:

    * `exact`: normalized lemma strings are identical;
    * `lemma_exact`: content lemma sets are identical, ignoring order;
    * `modifier_exact`: phrases have the same head and same modifiers;
    * `head_expansion`: one phrase is a head-only version of the other; or
    * `""`: no accepted match.

    Shared modifiers or substrings alone are not enough to match.  For example,
    `red button` and `blue button` share a head but conflict on modifiers, while
    `file` and `profile` only share a substring.
    """
    left_norm = normalized_argument_text(left)
    right_norm = normalized_argument_text(right)
    if not left_norm or not right_norm:
        return ""
    if left_norm == right_norm:
        return "exact"

    left_lemmas = content_lemmas(left)
    right_lemmas = content_lemmas(right)
    if not left_lemmas or not right_lemmas:
        return ""
    if left_lemmas == right_lemmas:
        return "lemma_exact"

    left_head = argument_head_lemma(left)
    right_head = argument_head_lemma(right)
    if not left_head or left_head != right_head:
        return ""

    left_mods = left_lemmas - {left_head}
    right_mods = right_lemmas - {right_head}
    if not left_mods or not right_mods:
        return "head_expansion"
    if left_mods == right_mods:
        return "modifier_exact"
    return ""


def argument_match_score(left, right):
    """Return a numeric confidence score for `argument_match_type`.

    The score is used only for ranking and reporting; boolean object matching
    still depends on whether `argument_match_type` returns a non-empty label.
    """
    match_type = argument_match_type(left, right)
    if match_type == "exact":
        return 1
    if match_type == "lemma_exact":
        return 0.95
    if match_type == "modifier_exact":
        return 0.9
    if match_type == "head_expansion":
        return 0.8
    return 0


def match_obj(gt_name, pred_name):
    """Return whether a predicted object is an accepted match for a gold object."""
    return bool(argument_match_type(gt_name, pred_name))


def lemma_text(text):
    """Return lowercase lemmas for all non-space tokens in `text`."""
    return " ".join(token.lemma_.lower() for token in nlp(str(text)) if not token.is_space)


def content_lemmas(text):
    """Return lowercase lemmas excluding whitespace and punctuation tokens."""
    return {
        token.lemma_.lower()
        for token in nlp(str(text))
        if not token.is_space and not token.is_punct
    }


def normalize_args(args):
    """Normalize predicted argument payloads to a list of strings.

    Some solvers return a single string instead of a list; treating that string
    as one argument avoids accidentally iterating over its characters.
    """
    if args is None:
        return []
    if isinstance(args, str):
        return [args]
    return [str(arg) for arg in args if arg is not None]


def action_text(act, words):
    """Resolve a gold action's verb token from its `act_idx`."""
    act_idx = act.get("act_idx")
    if isinstance(act_idx, int) and 0 <= act_idx < len(words):
        return words[act_idx]
    return ""


def action_arguments(act, words):
    """Resolve gold action object tokens from `obj_idxs[0]`.

    EASDRL uses `-1` as an empty-object marker in some records.  Invalid indices
    are ignored so malformed records do not abort a full evaluation run.
    """
    obj_idxs = act.get("obj_idxs", [[], []])
    if len(obj_idxs) == 0:
        return []
    if len(obj_idxs[0]) == 1 and obj_idxs[0][0] == -1:
        return []
    return [words[idx] for idx in obj_idxs[0] if isinstance(idx, int) and 0 <= idx < len(words)]


def action_record(act, words):
    """Convert an EASDRL action annotation into the diagnostic row shape."""
    return {
        "verb": action_text(act, words),
        "arguments": action_arguments(act, words),
        "act_idx": act.get("act_idx"),
        "obj_idxs": act.get("obj_idxs"),
        "act_type": act.get("act_type"),
        "related_acts": act.get("related_acts", []),
    }


def original_text(item):
    """Return the best available source text for an evaluated dataset item."""
    if item.get("original_text"):
        return item["original_text"]
    if item.get("sents"):
        return ". ".join(" ".join(sent) for sent in item["sents"]) + "."
    return " ".join(item.get("words", []))


def action_source_text(item, gold_action):
    """Return the sentence most likely to contain a gold action."""
    sentences = item.get("sents") or []
    verb = gold_action.get("verb", "")
    verb_lemmas = content_lemmas(verb)
    gold_arg_lemmas = content_lemmas(" ".join(gold_action.get("arguments", [])))

    if sentences and verb_lemmas:
        best_sentence = ""
        best_score = 0
        for sent in sentences:
            sent_text = " ".join(sent) if isinstance(sent, list) else str(sent)
            sent_lemmas = content_lemmas(sent_text)
            if not (verb_lemmas & sent_lemmas):
                continue
            score = 10 * len(verb_lemmas & sent_lemmas) + len(gold_arg_lemmas & sent_lemmas)
            if score > best_score:
                best_sentence = sent_text
                best_score = score
        if best_sentence:
            return best_sentence

    return original_text(item)


def doc_id(item, fallback):
    """Return a stable document id, falling back to the row index."""
    return item.get("doc_id", fallback)


def doc_key(item, dataset, fallback):
    """Return a globally useful document key for diagnostics."""
    if item.get("docId"):
        return item["docId"]
    return f"{dataset}:{item.get('doc_id', fallback)}"


def is_preposition_argument(arg):
    """Return whether an argument phrase starts with a preposition."""
    doc = nlp(str(arg))
    return bool(doc) and doc[0].lemma_.lower() in PREPOSITIONS


def is_preposition_object_in_text(arg, source_text):
    """Return whether `arg` appears as the object of a preposition in text.

    This is intentionally narrower than checking whether the argument text
    appears anywhere in the source.  It targets inconsistent annotations around
    phrases such as `put food into the red bowl`, where the object of the
    preposition may or may not be annotated.
    """
    if not arg or not source_text:
        return False

    arg_lemmas = content_lemmas(arg)
    if not arg_lemmas:
        return False

    doc = nlp(str(source_text))
    try:
        chunks = list(doc.noun_chunks)
    except ValueError:
        chunks = []

    for chunk in chunks:
        chunk_lemmas = content_lemmas(chunk.text)
        if arg_lemmas.issubset(chunk_lemmas):
            root = chunk.root
            if root.dep_ in {"pobj", "pcomp"} and root.head.lemma_.lower() in PREPOSITIONS:
                return True

    if len(arg_lemmas) != 1:
        return False

    for token in doc:
        if token.lemma_.lower() not in arg_lemmas:
            continue
        if token.dep_ in {"pobj", "pcomp"} and token.head.lemma_.lower() in PREPOSITIONS:
            return True

    return False


def is_generic_reference_argument(arg):
    """Return whether a gold argument is only a pronoun or generic placeholder."""
    lemmas = content_lemmas(arg)
    return bool(lemmas) and lemmas.issubset(GENERIC_REFERENCE_LEMMAS)


def has_gold_generic_reference_with_concrete_prediction(gold_args, pred_args, extra_in_pred):
    """Return whether generic gold arguments are refined by concrete predictions."""
    if not any(is_generic_reference_argument(arg) for arg in gold_args):
        return False
    candidates = extra_in_pred or pred_args
    return any(arg and not is_generic_reference_argument(arg) for arg in candidates)


def has_missing_gold_generic_reference(missing_from_pred, pred_args):
    """Return whether an unmatched generic gold argument co-occurs with concrete predictions."""
    if not any(is_generic_reference_argument(arg) for arg in missing_from_pred):
        return False
    return any(arg and not is_generic_reference_argument(arg) for arg in pred_args)


def is_entity_like_argument_in_text(arg, source_text):
    """Return whether `arg` appears as a noun phrase in the original text."""
    if not arg or not source_text:
        return False

    arg_lemmas = content_lemmas(arg)
    if not arg_lemmas or arg_lemmas.issubset(GENERIC_REFERENCE_LEMMAS):
        return False

    doc = nlp(str(source_text))
    try:
        chunks = list(doc.noun_chunks)
    except ValueError:
        chunks = []

    for chunk in chunks:
        if chunk.root.pos_ not in {"NOUN", "PROPN"}:
            continue
        if chunk.root.dep_ not in {"dobj", "obj", "attr", "nsubj", "nsubjpass"}:
            continue
        chunk_lemmas = content_lemmas(chunk.text)
        if arg_lemmas.issubset(chunk_lemmas):
            return True
    return False


def has_missing_valid_argument_evidence(gold_args, extra_in_pred, source_text):
    """Return whether empty gold arguments likely omitted a concrete text argument."""
    if gold_args or not extra_in_pred:
        return False
    return any(is_entity_like_argument_in_text(arg, source_text) for arg in extra_in_pred)


def is_split_modifier_case(extra_args, other_args, *, allow_head_only=False):
    """Detect split noun phrase arguments.

    This flags cases like `choose(square, shadow, box)` when another side has
    `square shadow box`.  With `allow_head_only=True`, it also catches a gold
    split such as `["square", "shadow", "box"]` against a predicted head-only
    argument like `["box"]`, which is usually an annotation-side split.
    """
    other_lemmas = content_lemmas(" ".join(other_args))
    if not other_lemmas:
        return False
    extra_lemmas = [content_lemmas(arg) for arg in extra_args]
    if any(lemmas and lemmas.issubset(other_lemmas) for lemmas in extra_lemmas):
        return True
    if not allow_head_only:
        return False
    if len(extra_args) < 2:
        return False

    combined_head = argument_head_lemma(" ".join([*extra_args, *other_args]))
    if not combined_head:
        return False
    for other_arg in other_args:
        if argument_head_lemma(other_arg) == combined_head:
            return True
    return False


def is_gold_split_modifier_case(gold_args, pred_args, missing_from_pred):
    """Return whether gold annotations appear to split one predicted phrase.

    A dataset-side head/modifier split needs evidence that multiple gold
    argument entries describe one predicted argument.  The evidence is checked
    per predicted argument; unrelated predicted arguments are not pooled into a
    single lemma set.  This avoids treating `pred=[split1, y]` as if it were the
    phrase `split1 y`.
    """
    if len(gold_args) < 2 or not missing_from_pred or not pred_args:
        return False

    for pred_arg in pred_args:
        pred_lemmas = content_lemmas(pred_arg)
        if not pred_lemmas:
            continue

        covered_gold = []
        covered_missing = []
        for gold_arg in gold_args:
            gold_lemmas = content_lemmas(gold_arg)
            if gold_lemmas and gold_lemmas.issubset(pred_lemmas):
                covered_gold.append(gold_arg)
                if gold_arg in missing_from_pred:
                    covered_missing.append(gold_arg)
        if len(covered_gold) >= 2 and covered_missing:
            return True

        pred_head = argument_head_lemma(pred_arg)
        if not pred_head:
            continue
        has_matched_gold_head = any(
            gold_arg not in missing_from_pred and argument_match_type(gold_arg, pred_arg)
            for gold_arg in gold_args
        )
        if not has_matched_gold_head:
            continue
        for missing_arg in missing_from_pred:
            combined_head = argument_head_lemma(" ".join([missing_arg, pred_arg]))
            if combined_head == pred_head:
                return True

    return False


def arg_diff(gold_args, pred_args):
    """Return unmatched gold and predicted arguments after one-to-one matching.

    Candidate pairs are ranked by strict match type before greedily assigning
    matches.  This prevents one predicted argument from satisfying multiple gold
    arguments and keeps extra/missing argument counts meaningful.
    """
    ranked_pairs = []
    for gi, gold_arg in enumerate(gold_args):
        for pi, pred_arg in enumerate(pred_args):
            match_type = argument_match_type(gold_arg, pred_arg)
            if match_type:
                ranked_pairs.append((ARGUMENT_MATCH_RANK[match_type], gi, pi))
    ranked_pairs.sort(reverse=True)

    matched_gold = set()
    matched_pred = set()
    for _, gi, pi in ranked_pairs:
        if gi in matched_gold or pi in matched_pred:
            continue
        matched_gold.add(gi)
        matched_pred.add(pi)

    missing_from_pred = [gold_args[i] for i in range(len(gold_args)) if i not in matched_gold]
    extra_in_pred = [pred_args[i] for i in range(len(pred_args)) if i not in matched_pred]
    return missing_from_pred, extra_in_pred


def _append_unique(items, values):
    for value in values:
        if value and value not in items:
            items.append(value)


def _any_preposition_object(args, source_text):
    return any(is_preposition_object_in_text(arg, source_text) for arg in args)


def classify_argument_mismatch(gold_args, pred_args, source_text=""):
    """Classify a matched action's argument-level mismatch.

    The return value contains the unmatched argument lists plus mutually
    exclusive candidate issue fields:

    * `candidate_dataset_issue` is populated when the gold annotation looks
      incomplete or over-split.
    * `candidate_llm_issue` is populated when the prediction appears to have
      missed, added, or confused arguments.

    When clear dataset-side evidence exists, LLM-side labels are suppressed so
    downstream analysis does not double-count the same mismatch as both causes.
    The `reason` string records the heuristic evidence used for that assignment.
    """
    missing_from_pred, extra_in_pred = arg_diff(gold_args, pred_args)
    notes = []
    dataset_issues = []
    llm_issues = []

    if missing_from_pred:
        notes.append("gold has arguments not matched by prediction")
        gold_specific_issues = []
        if _any_preposition_object(missing_from_pred, source_text):
            gold_specific_issues.extend(["extra_arguments", "extra_arguments:preposition_object"])
            notes.append("gold unmatched argument is a preposition object in source text")
        elif is_gold_split_modifier_case(gold_args, pred_args, missing_from_pred):
            gold_specific_issues.extend(["extra_arguments", "extra_arguments:unnecessary_head_or_modifier_split"])
            notes.append("gold unmatched argument looks like a split modifier/head word")
        elif not extra_in_pred and has_missing_gold_generic_reference(missing_from_pred, pred_args):
            gold_specific_issues.extend(["wrong_arguments", "wrong_arguments:gold_pronoun_or_generic_reference"])
            notes.append("gold unmatched argument is a pronoun or generic reference")
        if gold_specific_issues:
            _append_unique(dataset_issues, gold_specific_issues)
        else:
            _append_unique(llm_issues, ["missing_arguments"])

    if extra_in_pred:
        notes.append("prediction has arguments not matched by gold")
        if _any_preposition_object(extra_in_pred, source_text):
            _append_unique(dataset_issues, ["missing_arguments", "missing_arguments:preposition_object"])
            notes.append("pred unmatched argument is a preposition object in source text; annotation may have omitted it")
        elif has_gold_generic_reference_with_concrete_prediction(gold_args, pred_args, extra_in_pred):
            _append_unique(dataset_issues, ["wrong_arguments", "wrong_arguments:gold_pronoun_or_generic_reference"])
            notes.append("gold argument is a pronoun or generic reference while prediction provides a concrete argument")
        elif not dataset_issues:
            pred_specific_issues = ["extra_arguments"]
            if any(is_preposition_argument(arg) for arg in extra_in_pred):
                pred_specific_issues.append("extra_arguments:preposition_argument")
                notes.append("pred unmatched argument starts with a preposition")
            _append_unique(llm_issues, pred_specific_issues)

    if missing_from_pred and extra_in_pred:
        if dataset_issues:
            _append_unique(dataset_issues, ["wrong_arguments"])
            llm_issues = []
        else:
            _append_unique(llm_issues, ["wrong_arguments"])

    return {
        "missing_from_pred": missing_from_pred,
        "extra_in_pred": extra_in_pred,
        "candidate_dataset_issue": "|".join(dict.fromkeys(dataset_issues)),
        "candidate_llm_issue": "|".join(dict.fromkeys(llm_issues)),
        "strong_dataset_issue": "|".join(dict.fromkeys(dataset_issues)),
        "dataset_issue_confidence": "strong" if dataset_issues else "",
        "reason": "; ".join(notes),
    }


def best_verb_candidate(gold_action, unused_preds):
    """Find the unused prediction with the most lexical overlap to a gold action."""
    gold_verb = gold_action.get("verb", "")
    best = None
    best_score = 0
    for pred_idx, pred in unused_preds:
        pred_verb = pred.get("verb", "")
        score = len(content_lemmas(gold_verb) & content_lemmas(pred_verb))
        score += len(
            content_lemmas(" ".join(gold_action.get("arguments", [])))
            & content_lemmas(" ".join(normalize_args(pred.get("arguments", []))))
        )
        if score > best_score:
            best = (pred_idx, pred)
            best_score = score
    return best, best_score


def diagnostic_row(
    names,
    item,
    item_idx,
    mismatch_type,
    gold=None,
    pred=None,
    dataset_issue="",
    llm_issue="",
    reason="",
    strong_dataset_issue="",
    dataset_issue_confidence="",
):
    """Build one normalized mismatch diagnostic row."""
    ds_name, solver_name, model_name = names
    gold = gold or {}
    pred = pred or {}
    return {
        "dataset": ds_name,
        "solver": solver_name,
        "model": model_name,
        "doc_id": doc_id(item, item_idx),
        "docId": doc_key(item, ds_name, item_idx),
        "source_file": item.get("source_file", ""),
        "mismatch_type": mismatch_type,
        "candidate_dataset_issue": dataset_issue,
        "candidate_llm_issue": llm_issue,
        "strong_dataset_issue": strong_dataset_issue,
        "dataset_issue_confidence": dataset_issue_confidence,
        "reason": reason,
        "original_text": original_text(item),
        "gold_verb": gold.get("verb", ""),
        "gold_arguments": json.dumps(gold.get("arguments", []), ensure_ascii=False),
        "pred_verb": pred.get("verb", ""),
        "pred_arguments": json.dumps(normalize_args(pred.get("arguments", [])), ensure_ascii=False),
        "gold_action": json.dumps(gold, ensure_ascii=False),
        "pred_action": json.dumps(pred, ensure_ascii=False),
    }


def match_objs(act_obj_names, pred_obj_names):
    """Score predicted arguments against gold essential arguments.

    `act_obj_names` is `[essential_args, exclusive_args]` from the EASDRL action
    record.  Exclusive arguments can satisfy a prediction for matching purposes,
    but only essential arguments contribute to the gold denominator.  The return
    tuple is `(obj_right, obj_true, obj_tagged, obj_f1)`.
    """
    pred_obj_names = normalize_args(pred_obj_names)
    es_obj_names = act_obj_names[0] if len(act_obj_names) > 0 else []
    ex_obj_names = act_obj_names[1] if len(act_obj_names) > 1 else []

    obj_true = len(es_obj_names)
    obj_tagged = len(pred_obj_names)
    if obj_tagged == 0:
        return 0, obj_true, obj_tagged, 0

    ranked_pairs = []
    for gold_idx, gold_arg in enumerate(es_obj_names):
        for pred_idx, pred_arg in enumerate(pred_obj_names):
            match_type = argument_match_type(gold_arg, pred_arg)
            rank = ARGUMENT_MATCH_RANK.get(match_type, 0)
            for ex_arg in ex_obj_names:
                ex_match_type = argument_match_type(ex_arg, pred_arg)
                rank = max(rank, ARGUMENT_MATCH_RANK.get(ex_match_type, 0))
            if rank:
                ranked_pairs.append((rank, gold_idx, pred_idx))
    ranked_pairs.sort(reverse=True)

    matched_gold = set()
    matched_pred = set()
    for _, gold_idx, pred_idx in ranked_pairs:
        if gold_idx in matched_gold or pred_idx in matched_pred:
            continue
        matched_gold.add(gold_idx)
        matched_pred.add(pred_idx)

    obj_right = len(matched_gold)

    obj_precision = obj_right / obj_tagged if obj_tagged > 0 else 0
    obj_recall = obj_right / obj_true if obj_true > 0 else 0
    obj_f1 = 2 * obj_precision * obj_recall / (obj_precision + obj_recall) if (obj_precision + obj_recall) > 0 else 0

    return obj_right, obj_true, obj_tagged, obj_f1


def match(act, pred, words):
    """Match one gold action record to one predicted action.

    A match requires the gold verb lemma to appear in the predicted verb phrase.
    Object scoring is delegated to `match_objs`, so argument precision/recall can
    still expose partial or noisy object predictions after the verb matches.
    """
    act_idx = act.get("act_idx")
    if not isinstance(act_idx, int) or not 0 <= act_idx < len(words):
        return False, 0, 0, 0, 0
    act_name = words[act_idx]

    pred_act_name = pred.get("verb", None)
    if pred_act_name is None:
        return False, 0, 0, 0, 0

    act_lemma = nlp(act_name)[0].lemma_.lower()
    doc2 = nlp(pred_act_name)
    pred_act_lemma = " ".join([token.lemma_.lower() for token in doc2])
    if not act_lemma in pred_act_lemma:
        return False, 0, 0, 0, 0

    obj_idxs = act.get("obj_idxs", [[], []])
    es_obj_idxs = obj_idxs[0] if len(obj_idxs) > 0 else []
    ex_obj_idxs = obj_idxs[1] if len(obj_idxs) > 1 else []
    act_obj_names = [
        [words[ind] for ind in es_obj_idxs if isinstance(ind, int) and 0 <= ind < len(words)],
        [words[ind] for ind in ex_obj_idxs if isinstance(ind, int) and 0 <= ind < len(words)],
    ]
    pred_obj_names = normalize_args(pred.get("arguments", []))

    obj_right, obj_true, obj_tagged, obj_f1 = match_objs(act_obj_names, pred_obj_names)
    return True, obj_right, obj_true, obj_tagged, obj_f1
