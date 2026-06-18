import csv
import json
import os

import spacy


nlp = spacy.load("en_core_web_sm")

DATASETS = ("cooking", "wikihow", "win2k")
SOLVERS = ("gpt3_to_plan", "nl2p_1", "nl2p_2", "nl2p_3", "verb_args")

PREPOSITIONS = {
    "about", "above", "across", "after", "against", "along", "among", "around",
    "at", "before", "behind", "below", "beneath", "beside", "between", "by",
    "down", "during", "for", "from", "in", "inside", "into", "near", "of",
    "off", "on", "onto", "out", "over", "through", "to", "under", "up", "with",
    "within", "without",
}

ARGUMENT_MATCH_RANK = {
    "exact": 4,
    "lemma_exact": 3,
    "modifier_exact": 2,
    "head_expansion": 1,
}


def parse_result_filename(file):
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
    if not diagnostics:
        print("No mismatch diagnostics to write.")
        return
    if not os.path.exists(dir):
        os.makedirs(dir)
    outpath = os.path.join(dir, "evaluation_mismatch_diagnostics.csv")
    fieldnames = [
        "dataset", "solver", "model", "doc_id", "docId", "source_file",
        "mismatch_type", "candidate_dataset_issue", "candidate_llm_issue",
        "reason", "original_text", "gold_verb", "gold_arguments",
        "pred_verb", "pred_arguments", "gold_action", "pred_action",
    ]
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in diagnostics:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print("Mismatch diagnostics written to %s" % outpath)


def normalized_argument_text(text):
    return " ".join(
        token.lemma_.lower()
        for token in nlp(str(text))
        if not token.is_space and not token.is_punct
    )


def argument_head_lemma(text):
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
    return bool(argument_match_type(gt_name, pred_name))


def lemma_text(text):
    return " ".join(token.lemma_.lower() for token in nlp(str(text)) if not token.is_space)


def content_lemmas(text):
    return {
        token.lemma_.lower()
        for token in nlp(str(text))
        if not token.is_space and not token.is_punct
    }


def normalize_args(args):
    if args is None:
        return []
    if isinstance(args, str):
        return [args]
    return [str(arg) for arg in args if arg is not None]


def action_text(act, words):
    act_idx = act.get("act_idx")
    if isinstance(act_idx, int) and 0 <= act_idx < len(words):
        return words[act_idx]
    return ""


def action_arguments(act, words):
    obj_idxs = act.get("obj_idxs", [[], []])
    if len(obj_idxs) == 0:
        return []
    if len(obj_idxs[0]) == 1 and obj_idxs[0][0] == -1:
        return []
    return [words[idx] for idx in obj_idxs[0] if isinstance(idx, int) and 0 <= idx < len(words)]


def action_record(act, words):
    return {
        "verb": action_text(act, words),
        "arguments": action_arguments(act, words),
        "act_idx": act.get("act_idx"),
        "obj_idxs": act.get("obj_idxs"),
        "act_type": act.get("act_type"),
        "related_acts": act.get("related_acts", []),
    }


def original_text(item):
    if item.get("original_text"):
        return item["original_text"]
    if item.get("sents"):
        return ". ".join(" ".join(sent) for sent in item["sents"]) + "."
    return " ".join(item.get("words", []))


def doc_id(item, fallback):
    return item.get("doc_id", fallback)


def doc_key(item, dataset, fallback):
    if item.get("docId"):
        return item["docId"]
    return f"{dataset}:{item.get('doc_id', fallback)}"


def is_preposition_argument(arg):
    doc = nlp(str(arg))
    return bool(doc) and doc[0].lemma_.lower() in PREPOSITIONS


def is_split_modifier_case(extra_args, other_args, *, allow_head_only=False):
    """Detect cases like choose(square, shadow, box) for one noun phrase."""
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


def arg_diff(gold_args, pred_args):
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


def _args_appear_in_text(args, text):
    if not text:
        return False
    normalized_text = normalized_argument_text(text)
    for arg in args:
        normalized_arg = normalized_argument_text(arg)
        if normalized_arg and normalized_arg in normalized_text:
            return True
    return False


def classify_argument_mismatch(gold_args, pred_args, source_text=""):
    missing_from_pred, extra_in_pred = arg_diff(gold_args, pred_args)
    notes = []
    dataset_issues = []
    llm_issues = []

    if missing_from_pred:
        notes.append("gold has arguments not matched by prediction")
        gold_specific_issues = []
        if any(is_preposition_argument(arg) for arg in missing_from_pred):
            gold_specific_issues.extend(["extra_arguments", "extra_arguments:preposition_argument"])
            notes.append("gold unmatched argument starts with a preposition")
        if is_split_modifier_case(missing_from_pred, pred_args, allow_head_only=True):
            gold_specific_issues.extend(["extra_arguments", "extra_arguments:unnecessary_head_or_modifier_split"])
            notes.append("gold unmatched argument looks like a split modifier/head word")
        if gold_specific_issues:
            _append_unique(dataset_issues, gold_specific_issues)
        else:
            _append_unique(llm_issues, ["missing_arguments"])

    if extra_in_pred:
        notes.append("prediction has arguments not matched by gold")
        has_full_phrase_extra = any(len(content_lemmas(arg)) > 1 for arg in extra_in_pred)
        if (not gold_args or has_full_phrase_extra) and _args_appear_in_text(extra_in_pred, source_text):
            _append_unique(dataset_issues, ["missing_arguments"])
            notes.append("pred unmatched argument appears in original text; annotation may have missed it")
        elif not dataset_issues:
            pred_specific_issues = ["extra_arguments"]
            if any(is_preposition_argument(arg) for arg in extra_in_pred):
                pred_specific_issues.append("extra_arguments:preposition_argument")
                notes.append("pred unmatched argument starts with a preposition")
            if is_split_modifier_case(extra_in_pred, gold_args):
                pred_specific_issues.append("extra_arguments:unnecessary_head_or_modifier_split")
                notes.append("pred unmatched argument looks like a split modifier/head word")
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
        "reason": "; ".join(notes),
    }


def best_verb_candidate(gold_action, unused_preds):
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


def diagnostic_row(names, item, item_idx, mismatch_type, gold=None, pred=None, dataset_issue="", llm_issue="", reason=""):
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
