from collections import defaultdict
import csv
import json
import os
import sys

from tqdm import tqdm

from src.utils import load_pkl
from src.evaluation_helpers import (
    DATASETS,
    SOLVERS,
    action_record,
    action_source_text,
    arg_diff,
    argument_head_lemma,
    argument_match_score,
    argument_match_type,
    best_verb_candidate,
    classify_argument_mismatch,
    content_lemmas,
    diagnostic_row,
    doc_id,
    doc_key,
    is_preposition_argument,
    is_split_modifier_case,
    lemma_text,
    match,
    match_action,
    match_obj,
    match_objs,
    normalize_args,
    normalized_argument_text,
    original_text,
    parse_result_filename,
    write_diagnostics,
)

DEBUG = False


def read_from_refined_dataset(filename, limit=None):
    """Read a refined EASDRL dataset file."""
    path = os.path.join("./data/easdrl", filename + ".pkl")
    dataset = load_pkl(path)[-1]
    if limit is not None:
        dataset = dataset[:limit]
    return dataset


def read_from_labeled_dataset(filename, limit=None):
    """Read a labeled EASDRL dataset file."""
    path = os.path.join("./data/easdrl", filename + ".pkl")
    dataset = load_pkl(path)
    if limit is not None:
        dataset = dataset[:limit]
    return dataset


def read_from_predicted_dataset(dir):
    res_dict = defaultdict(list)
    if not os.path.exists(dir):
        raise FileNotFoundError(f"The results dir {dir} does not exist.")
    files = sorted(os.listdir(dir))
    pkl_files = [f for f in files if f.endswith(".pkl")]
    for file in pkl_files:
        print(f"Loading {file}")
        path = os.path.join(dir, file)
        ds_name, solver_name, model_name = parse_result_filename(file)
        res_dict[(ds_name, solver_name, model_name)] = load_pkl(path)
    return res_dict


def write_results(results: dict, dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)
    outpath = os.path.join(dir, "evaluation_result.csv")
    count_columns = [
        "perfect_action_argument_matches",
        "argument_mismatch_actions",
        "matched_action_events",
    ]
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "solver",
            "model",
            "Precision",
            "Recall",
            "F1",
            "Object Precision",
            "Object Recall",
            "Object F1",
            "adjusted_precision",
            "adjusted_recall",
            "adjusted_f1",
            *count_columns,
        ])
        for k in sorted(results):
            v = results[k]
            ds_name, solver, model_name = k
            if len(v) == 6:
                precision, recall, f1, obj_precision, obj_recall, obj_f1 = v
                adjusted_precision, adjusted_recall, adjusted_f1 = obj_precision, obj_recall, obj_f1
                counts = [""] * len(count_columns)
            elif len(v) == 9:
                precision, recall, f1, obj_precision, obj_recall, obj_f1, adjusted_precision, adjusted_recall, adjusted_f1 = v
                counts = [""] * len(count_columns)
            else:
                precision, recall, f1, obj_precision, obj_recall, obj_f1, adjusted_precision, adjusted_recall, adjusted_f1 = v[:9]
                counts = list(v[9:])
            writer.writerow([
                ds_name,
                solver,
                model_name,
                precision,
                recall,
                f1,
                obj_precision,
                obj_recall,
                obj_f1,
                adjusted_precision,
                adjusted_recall,
                adjusted_f1,
                *counts,
            ])
    print("Results written to %s" % outpath)


def _exclusive_action_key(act):
    return frozenset([act.get("act_idx"), *act.get("related_acts", [])])


ESSENTIAL_ACTION = 1
OPTIONAL_ACTION = 2
EXCLUSIVE_ACTION = 3


def _gold_action_denominator_increment(act_type, matched):
    """Return the EASDRL contribution of one action unit to TotalTruth.

    EASDRL defines optionality for actions, not for arguments.  An essential
    action or one exclusive-action group is always a gold unit.  An optional
    action becomes a gold unit only when it is extracted; omitting it is a
    valid output and must therefore be neutral rather than a false negative.
    Arguments are scored separately, and only after their parent action has
    matched.
    """
    if act_type == OPTIONAL_ACTION:
        return int(matched)
    return 1


def _strict_first_action_match(acts, pred, used, words):
    """Return the first unused prediction matching one gold action alternative.

    Matching is action-only.  Arguments are intentionally not inspected until
    the gold/predicted action pair has been fixed.
    """
    for pred_idx, pred_act in enumerate(pred):
        if used[pred_idx]:
            continue
        for act in acts:
            if match_action(act, pred_act, words):
                used[pred_idx] = True
                return act, pred_idx
    return None, None


def _classify_matched_argument_mismatch(item, gold, pred_act):
    return classify_argument_mismatch(
        gold["arguments"],
        normalize_args(pred_act.get("arguments", [])),
        source_text=action_source_text(item, gold),
        action_verb=gold.get("verb", ""),
    )


def _diagnose_matched_argument_mismatch(names, item, item_idx, gold, pred_act, arg_info):
    row = diagnostic_row(
        names,
        item,
        item_idx,
        "argument_mismatch",
        gold=gold,
        pred=pred_act,
        dataset_issue=arg_info["candidate_dataset_issue"],
        llm_issue=arg_info["candidate_llm_issue"],
        strong_dataset_issue=arg_info["strong_dataset_issue"],
        dataset_issue_confidence=arg_info["dataset_issue_confidence"],
        reason=arg_info["reason"],
    )
    row["missing_gold_args"] = json_dumps(arg_info["missing_from_pred"])
    row["extra_pred_args"] = json_dumps(arg_info["extra_in_pred"])
    return row


def json_dumps(value):
    return json.dumps(value, ensure_ascii=False)


def _adjusted_object_deductions(arg_info):
    issues = set(filter(None, arg_info.get("strong_dataset_issue", "").split("|")))
    gold_deduction = 0
    pred_deduction = 0
    if "extra_arguments:unnecessary_head_or_modifier_split" in issues:
        gold_deduction += len(arg_info.get("split_missing_from_pred", []))
    if "extra_arguments:preposition_object" in issues:
        gold_deduction += len(arg_info.get("missing_preposition_objects", []))
    if "missing_arguments:preposition_object" in issues:
        pred_deduction += len(arg_info.get("extra_preposition_objects", []))
    return gold_deduction, pred_deduction


def _diagnose_unmatched_gold(names, item, item_idx, gold, pred, used, diagnostic_used=None):
    if diagnostic_used is None:
        diagnostic_used = set()
    unused_preds = [
        (idx, pred_act)
        for idx, pred_act in enumerate(pred)
        if not used[idx] and idx not in diagnostic_used
    ]
    candidate, score = best_verb_candidate(gold, unused_preds)
    if candidate and score > 1:
        pred_idx, pred_act = candidate
        diagnostic_used.add(pred_idx)
        row = diagnostic_row(
            names,
            item,
            item_idx,
            "wrong_action",
            gold=gold,
            pred=pred_act,
            llm_issue="wrong_actions",
            reason="unmatched gold action has lexical/argument overlap with an unused prediction",
        )
        return row
    row = diagnostic_row(
        names,
        item,
        item_idx,
        "unmatched_gold_action",
        gold=gold,
        llm_issue="missing_actions",
        reason="gold action was not matched by any prediction",
    )
    return row


def _diagnose_unused_predictions(names, item, item_idx, pred, used):
    rows = []
    for pred_idx, pred_act in enumerate(pred):
        if used[pred_idx]:
            continue
        pred_verb = pred_act.get("verb", "")
        if pred_verb and lemma_text(pred_verb) in lemma_text(original_text(item)):
            dataset_issue = "missing_actions"
            llm_issue = ""
            reason = "unused prediction verb appears in original text; annotation may have missed this action"
        else:
            dataset_issue = ""
            llm_issue = "extra_actions"
            reason = "unused prediction did not match any gold action"
        rows.append(
            diagnostic_row(
                names,
                item,
                item_idx,
                "unmatched_prediction",
                pred=pred_act,
                dataset_issue=dataset_issue,
                llm_issue=llm_issue,
                reason=reason,
            )
        )
    return rows


def evaluation(preds, names=("", "", ""), collect_diagnostics=False):
    total_right = total_truth = total_tagged = 0
    obj_total_right = obj_total_truth = obj_total_tagged = 0
    adjusted_obj_total_truth = adjusted_obj_total_tagged = 0
    # matched action, and arguments are also perfectly matched
    perfect_action_argument_matches = 0
    # matched action, but arguments have missing/extra mismatch
    argument_mismatch_actions = 0
    diagnostics = []

    for item_idx, item in enumerate(tqdm(preds, desc="Processing", unit="item")):
        words = item["words"]
        acts = item["acts"]
        pred = item["pred"] or []

        if not pred:
            print(f"No predictions found for item {item_idx}.")

        used = [False] * len(pred)
        pending_unmatched_gold = []
        exclusive_groups = defaultdict(list)
        for act in acts:
            if act.get("act_type") == EXCLUSIVE_ACTION:
                exclusive_groups[_exclusive_action_key(act)].append(act)

        processed_exclusive_groups = set()
        action_units = []
        for action_order, act in enumerate(acts):
            act_type = act.get("act_type")
            if act_type == EXCLUSIVE_ACTION:
                group_key = _exclusive_action_key(act)
                if group_key in processed_exclusive_groups:
                    continue
                processed_exclusive_groups.add(group_key)
                action_units.append((action_order, act_type, exclusive_groups[group_key]))
            elif act_type in {ESSENTIAL_ACTION, OPTIONAL_ACTION}:
                action_units.append((action_order, act_type, [act]))

        for _, act_type, alternatives in sorted(action_units, key=lambda unit: unit[0]):
            matched_act, pred_idx = _strict_first_action_match(alternatives, pred, used, words)
            total_truth += _gold_action_denominator_increment(act_type, matched_act is not None)
            if matched_act is None:
                if collect_diagnostics and act_type != OPTIONAL_ACTION:
                    pending_unmatched_gold.append(action_record(alternatives[0], words))
                continue

            total_right += 1
            _, obj_right, obj_true, obj_tagged, _ = match(matched_act, pred[pred_idx], words)
            obj_total_tagged += obj_tagged
            obj_total_truth += obj_true
            obj_total_right += obj_right
            adjusted_obj_total_tagged += obj_tagged
            adjusted_obj_total_truth += obj_true
            if obj_right == obj_true and obj_right == obj_tagged:
                perfect_action_argument_matches += 1
            else:
                argument_mismatch_actions += 1
            if obj_right < obj_true or obj_right < obj_tagged:
                gold = action_record(matched_act, words)
                arg_info = _classify_matched_argument_mismatch(item, gold, pred[pred_idx])
                gold_deduction, pred_deduction = _adjusted_object_deductions(arg_info)
                adjusted_obj_total_truth -= gold_deduction
                adjusted_obj_total_tagged -= pred_deduction
                if collect_diagnostics:
                    diagnostics.append(_diagnose_matched_argument_mismatch(names, item, item_idx, gold, pred[pred_idx], arg_info))

        total_tagged += len(pred)

        if collect_diagnostics:
            diagnostic_used = set()
            for gold in pending_unmatched_gold:
                diagnostics.append(_diagnose_unmatched_gold(names, item, item_idx, gold, pred, used, diagnostic_used))
            diagnostics.extend(_diagnose_unused_predictions(names, item, item_idx, pred, used))

    precision = total_right / total_tagged if total_tagged > 0 else 0
    recall = total_right / total_truth if total_truth > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    obj_precision = obj_total_right / obj_total_tagged if obj_total_tagged > 0 else 0
    obj_recall = obj_total_right / obj_total_truth if obj_total_truth > 0 else 0
    obj_f1 = 2 * obj_precision * obj_recall / (obj_precision + obj_recall) if (obj_precision + obj_recall) > 0 else 0
    adjusted_obj_total_truth = max(adjusted_obj_total_truth, obj_total_right)
    adjusted_obj_total_tagged = max(adjusted_obj_total_tagged, obj_total_right)
    adjusted_precision = obj_total_right / adjusted_obj_total_tagged if adjusted_obj_total_tagged > 0 else 0
    adjusted_recall = obj_total_right / adjusted_obj_total_truth if adjusted_obj_total_truth > 0 else 0
    adjusted_f1 = (
        2 * adjusted_precision * adjusted_recall / (adjusted_precision + adjusted_recall)
        if (adjusted_precision + adjusted_recall) > 0
        else 0
    )

    if precision == 0 or recall == 0:
        print("warning: zero precision or recall")

    matched_action_events = perfect_action_argument_matches + argument_mismatch_actions

    metrics = (
        precision,
        recall,
        f1,
        obj_precision,
        obj_recall,
        obj_f1,
        adjusted_precision,
        adjusted_recall,
        adjusted_f1,
        perfect_action_argument_matches,
        argument_mismatch_actions,
        matched_action_events,
    )
    if collect_diagnostics:
        return metrics, diagnostics
    return metrics


def run_evaluation(predicates, collect_diagnostics=False):
    results = {}
    all_diagnostics = []
    for names, raw_res in predicates.items():
        ds_name, solver_name, model_name = names
        print(f"Evaluating {ds_name} with solver {solver_name} and model {model_name}")
        if collect_diagnostics:
            metrics, diagnostics = evaluation(raw_res, names=names, collect_diagnostics=True)
            all_diagnostics.extend(diagnostics)
        else:
            metrics = evaluation(raw_res)
        results[(ds_name, solver_name, model_name)] = metrics
    if collect_diagnostics:
        return results, all_diagnostics
    return results


def main(args):
    if args.debug:
        global DEBUG
        DEBUG = True
        print("Debug mode is on!")

    dir = args.d
    if not os.path.exists(dir):
        print(f"The results dir {dir} does not exist.")
        sys.exit(1)

    predicates = read_from_predicted_dataset(dir)
    if args.diagnostics:
        results, diagnostics = run_evaluation(predicates, collect_diagnostics=True)
    else:
        results = run_evaluation(predicates)
        diagnostics = []
    print("Evaluation done!")
    write_results(results, dir)
    if args.diagnostics:
        write_diagnostics(diagnostics, dir)

def debug():
    # Evaluation Parameters
    solver_name = "nl2p_1"
    dataset_name = "cooking"
    model_name = "gpt-5.4"
    doc_id = 38

    # Read dataset
    input_dir = f"./results/{solver_name}/{model_name}"
    output_dir = f"./results/{solver_name}/{model_name}/debug"
    os.makedirs(output_dir, exist_ok=True)
    predicts = read_from_predicted_dataset(input_dir)
    target_ds = predicts.get((dataset_name, solver_name, model_name), [])
    if not target_ds:
        raise ValueError(f"No predictions found for dataset {dataset_name}, solver {solver_name}, model {model_name}.")    
    
    # Create dummy input for evaluation
    target_doc = [doc for doc in target_ds if doc.get("doc_id") == doc_id]
    target_input = {
        (dataset_name, solver_name, model_name): target_doc
    }

    # Run evaluation and output
    results, diagnostics = run_evaluation(target_input, collect_diagnostics=True)
    print("Evaluation done!")
    write_results(results, output_dir)
    write_diagnostics(diagnostics, output_dir)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        debug()
        sys.exit(0)
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, default="./results", help="results directory")
    parser.add_argument("--diagnostics", action="store_true", help="write mismatch diagnostics for annotation/LLM error analysis")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()
    main(args)
