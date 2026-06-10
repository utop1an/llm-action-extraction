from collections import defaultdict
import csv
import os
import sys

from tqdm import tqdm

from src.utils import load_pkl
from src.evaluation_helpers import (
    action_record,
    best_verb_candidate,
    classify_argument_mismatch,
    diagnostic_row,
    lemma_text,
    match,
    normalize_args,
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
    files = os.listdir(dir)
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
        ])
        for k, v in results.items():
            ds_name, solver, model_name = k
            precision, recall, f1, obj_precision, obj_recall, obj_f1 = v
            writer.writerow([ds_name, solver, model_name, precision, recall, f1, obj_precision, obj_recall, obj_f1])
    print("Results written to %s" % outpath)


def _count_truth(act, counted_exclusive_acts):
    act_type = act["act_type"]
    if act_type == 1:
        return 1
    if act_type == 3:
        act_idx = act["act_idx"]
        related_act_indices = act["related_acts"]
        all_indices = set(related_act_indices + [act_idx])
        if all_indices.isdisjoint(counted_exclusive_acts):
            counted_exclusive_acts.update(all_indices)
            return 1
    return 0


def _best_prediction_for_act(act, pred, used, words):
    best = (None, 0, 0, 0, 0)
    for pred_idx, pred_act in enumerate(pred):
        if used[pred_idx]:
            continue
        matched, obj_right, obj_true, obj_tagged, obj_f1 = match(act, pred_act, words)
        if matched and best[4] < obj_f1:
            best = (pred_idx, obj_right, obj_true, obj_tagged, obj_f1)
    return best


def _diagnose_matched_argument_mismatch(names, item, item_idx, gold, pred_act):
    arg_info = classify_argument_mismatch(gold["arguments"], normalize_args(pred_act.get("arguments", [])))
    return diagnostic_row(
        names,
        item,
        item_idx,
        "argument_mismatch",
        gold=gold,
        pred=pred_act,
        dataset_issue=arg_info["candidate_dataset_issue"],
        llm_issue=arg_info["candidate_llm_issue"],
        reason=arg_info["reason"],
    )


def _diagnose_unmatched_gold(names, item, item_idx, gold, pred, used):
    unused_preds = [(idx, pred_act) for idx, pred_act in enumerate(pred) if not used[idx]]
    candidate, score = best_verb_candidate(gold, unused_preds)
    if candidate and score > 0:
        _, pred_act = candidate
        return diagnostic_row(
            names,
            item,
            item_idx,
            "wrong_action",
            gold=gold,
            pred=pred_act,
            dataset_issue="wrong_actions",
            llm_issue="wrong_actions",
            reason="unmatched gold action has lexical/argument overlap with an unused prediction",
        )
    return diagnostic_row(
        names,
        item,
        item_idx,
        "unmatched_gold_action",
        gold=gold,
        dataset_issue="extra_actions",
        llm_issue="missing_actions",
        reason="gold action was not matched by any prediction",
    )


def _diagnose_unused_predictions(names, item, item_idx, pred, used):
    rows = []
    for pred_idx, pred_act in enumerate(pred):
        if used[pred_idx]:
            continue
        pred_verb = pred_act.get("verb", "")
        if pred_verb and lemma_text(pred_verb) in lemma_text(original_text(item)):
            dataset_issue = "missing_actions"
            reason = "unused prediction verb appears in original text; annotation may have missed this action"
        else:
            dataset_issue = ""
            reason = "unused prediction did not match any gold action"
        rows.append(
            diagnostic_row(
                names,
                item,
                item_idx,
                "unmatched_prediction",
                pred=pred_act,
                dataset_issue=dataset_issue,
                llm_issue="extra_actions",
                reason=reason,
            )
        )
    return rows


def evaluation(preds, names=("", "", ""), collect_diagnostics=False):
    total_right = total_truth = total_tagged = 0
    obj_total_right = obj_total_truth = obj_total_tagged = 0
    diagnostics = []

    for item_idx, item in enumerate(tqdm(preds, desc="Processing", unit="item")):
        words = item["words"]
        acts = item["acts"]
        pred = item["pred"] or []
        counted_exclusive_acts = set()

        if not pred:
            print(f"No predictions found for item {item_idx}.")

        used = [False] * len(pred)
        for act in acts:
            total_truth += _count_truth(act, counted_exclusive_acts)
            best = _best_prediction_for_act(act, pred, used, words)

            if best[0] is None:
                if collect_diagnostics:
                    gold = action_record(act, words)
                    diagnostics.append(_diagnose_unmatched_gold(names, item, item_idx, gold, pred, used))
                continue

            if act["act_type"] == 2:
                total_truth += 1
            total_right += 1
            obj_total_tagged += best[3]
            obj_total_truth += best[2]
            obj_total_right += best[1]

            used[best[0]] = True
            if collect_diagnostics and (best[1] < best[2] or best[1] < best[3]):
                gold = action_record(act, words)
                diagnostics.append(_diagnose_matched_argument_mismatch(names, item, item_idx, gold, pred[best[0]]))

        total_tagged += len(pred)

        if collect_diagnostics:
            diagnostics.extend(_diagnose_unused_predictions(names, item, item_idx, pred, used))

    precision = total_right / total_tagged if total_tagged > 0 else 0
    recall = total_right / total_truth if total_truth > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    obj_precision = obj_total_right / obj_total_tagged if obj_total_tagged > 0 else 0
    obj_recall = obj_total_right / obj_total_truth if obj_total_truth > 0 else 0
    obj_f1 = 2 * obj_precision * obj_recall / (obj_precision + obj_recall) if (obj_precision + obj_recall) > 0 else 0

    if precision == 0 or recall == 0:
        print("warning: zero precision or recall")

    metrics = (precision, recall, f1, obj_precision, obj_recall, obj_f1)
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, default="./results", help="results directory")
    parser.add_argument("--diagnostics", action="store_true", help="write mismatch diagnostics for annotation/LLM error analysis")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()
    main(args)
