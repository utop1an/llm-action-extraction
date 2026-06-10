
from collections import defaultdict
import csv
import os, sys
import json

from tqdm import tqdm
from src.utils import load_pkl

import spacy

nlp = spacy.load("en_core_web_sm")

DEBUG = False

DATASETS = ("cooking", "wikihow", "win2k")
SOLVERS = ("gpt3_to_plan", "nl2p_1", "nl2p_2", "nl2p_3", "verb_args")

PREPOSITIONS = {
    "about", "above", "across", "after", "against", "along", "among", "around",
    "at", "before", "behind", "below", "beneath", "beside", "between", "by",
    "down", "during", "for", "from", "in", "inside", "into", "near", "of",
    "off", "on", "onto", "out", "over", "through", "to", "under", "up", "with",
    "within", "without",
}

ARGUMENT_MATCH_THRESHOLD = 0.65

def read_from_refined_dataset(filename, limit=None):
    """
    Read data from a dataset file
    """
    path = os.path.join('./data/easdrl', filename + '.pkl')
    dataset = load_pkl(path)[-1]
    if limit is not None:
        dataset = dataset[:limit]
    return dataset

def read_from_labeled_dataset(filename, limit=None):
    """
    Read data from a dataset file
    """
    path = os.path.join('./data/easdrl', filename + '.pkl')
    dataset = load_pkl(path)
    if limit is not None:
        dataset = dataset[:limit]
    return dataset

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

def read_from_predicted_dataset(dir):
    res_dict = defaultdict(list)
    if not os.path.exists(dir):
        raise FileNotFoundError(f"The results dir {dir} does not exist.")
    files = os.listdir(dir)
    pkl_files = [f for f in files if f.endswith('.pkl')]
    for file in pkl_files:
        print(f"Loading {file}")
        path = os.path.join(dir, file)
        ds_name, solver_name, model_name = parse_result_filename(file)
        res_dict[(ds_name,solver_name,model_name)] = load_pkl(path)
    return res_dict

def write_results(results: dict, dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)
    outpath = os.path.join(dir, 'evaluation_result.csv')
    with open(outpath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'solver', 'model', 'Precision', 'Recall', 'F1', 'Object Precision', 'Object Recall', 'Object F1'])
        for k,v in results.items():
            ds_name, solver, model_name = k
            precision, recall, f1, obj_precision, obj_recall, obj_f1 = v
            writer.writerow([ds_name, solver, model_name, precision, recall, f1, obj_precision, obj_recall, obj_f1])
    print('Results written to %s' % outpath)

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

def argument_match_score(left, right):
    left_norm = normalized_argument_text(left)
    right_norm = normalized_argument_text(right)
    if not left_norm or not right_norm:
        return 0
    if left_norm == right_norm:
        return 1

    left_lemmas = content_lemmas(left)
    right_lemmas = content_lemmas(right)
    if not left_lemmas or not right_lemmas:
        return 0

    left_head = argument_head_lemma(left)
    right_head = argument_head_lemma(right)
    intersection = left_lemmas & right_lemmas
    if not intersection:
        return 0

    smaller = min(len(left_lemmas), len(right_lemmas))
    larger = max(len(left_lemmas), len(right_lemmas))
    overlap_small = len(intersection) / smaller
    overlap_large = len(intersection) / larger

    if left_lemmas == right_lemmas:
        return 0.95
    if left_head and left_head == right_head:
        if left_lemmas.issubset(right_lemmas) or right_lemmas.issubset(left_lemmas):
            return 0.88
        if overlap_small == 1 and overlap_large >= 0.75:
            return 0.78
        return 0.55
    if left_lemmas.issubset(right_lemmas) or right_lemmas.issubset(left_lemmas):
        return 0.72
    if overlap_large >= 0.8:
        return 0.7
    return 0

def match_obj(gt_name, pred_name):
    return argument_match_score(gt_name, pred_name) >= ARGUMENT_MATCH_THRESHOLD

def lemma_text(text):
    return " ".join(token.lemma_.lower() for token in nlp(str(text)) if not token.is_space)

def content_lemmas(text):
    return {
        token.lemma_.lower()
        for token in nlp(str(text))
        if not token.is_space and not token.is_punct and not token.is_stop
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

def is_split_modifier_case(extra_args, other_args):
    """Detect cases like choose(square, shadow, box) for one noun phrase."""
    other_lemmas = content_lemmas(" ".join(other_args))
    if not other_lemmas:
        return False
    extra_lemmas = [content_lemmas(arg) for arg in extra_args]
    return any(lemmas and lemmas.issubset(other_lemmas) for lemmas in extra_lemmas)

def arg_diff(gold_args, pred_args):
    scored_pairs = []
    for gi, gold_arg in enumerate(gold_args):
        for pi, pred_arg in enumerate(pred_args):
            score = argument_match_score(gold_arg, pred_arg)
            if score >= ARGUMENT_MATCH_THRESHOLD:
                scored_pairs.append((score, gi, pi))
    scored_pairs.sort(reverse=True)

    matched_gold = set()
    matched_pred = set()
    for _, gi, pi in scored_pairs:
        if gi in matched_gold or pi in matched_pred:
            continue
        matched_gold.add(gi)
        matched_pred.add(pi)

    missing_from_pred = [gold_args[i] for i in range(len(gold_args)) if i not in matched_gold]
    extra_in_pred = [pred_args[i] for i in range(len(pred_args)) if i not in matched_pred]
    return missing_from_pred, extra_in_pred

def classify_argument_mismatch(gold_args, pred_args):
    missing_from_pred, extra_in_pred = arg_diff(gold_args, pred_args)
    notes = []
    dataset_issues = []
    llm_issues = []

    if missing_from_pred:
        dataset_issues.append("extra_arguments")
        llm_issues.append("missing_arguments")
        notes.append("gold has arguments not matched by prediction")
        if any(is_preposition_argument(arg) for arg in missing_from_pred):
            dataset_issues.append("extra_arguments:preposition_argument")
            notes.append("gold unmatched argument starts with a preposition")
        if is_split_modifier_case(missing_from_pred, pred_args):
            dataset_issues.append("extra_arguments:unnecessary_head_or_modifier_split")
            notes.append("gold unmatched argument looks like a split modifier/head word")

    if extra_in_pred:
        dataset_issues.append("missing_arguments")
        llm_issues.append("extra_arguments")
        notes.append("prediction has arguments not matched by gold")
        if any(is_preposition_argument(arg) for arg in extra_in_pred):
            llm_issues.append("extra_arguments:preposition_argument")
            notes.append("pred unmatched argument starts with a preposition")
        if is_split_modifier_case(extra_in_pred, gold_args):
            llm_issues.append("extra_arguments:unnecessary_head_or_modifier_split")
            notes.append("pred unmatched argument looks like a split modifier/head word")

    if missing_from_pred and extra_in_pred:
        dataset_issues.append("wrong_arguments")
        llm_issues.append("wrong_arguments")

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
        score += len(content_lemmas(" ".join(gold_action.get("arguments", []))) & content_lemmas(" ".join(normalize_args(pred.get("arguments", [])))))
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
    es_obj_names = act_obj_names[0]
    ex_obj_names = act_obj_names[1]

    obj_true = len(es_obj_names)
    obj_tagged = len(pred_obj_names)
    if obj_tagged == 0:
        return 0, obj_true, obj_tagged, 0

    scored_pairs = []
    for gold_idx, gold_arg in enumerate(es_obj_names):
        for pred_idx, pred_arg in enumerate(pred_obj_names):
            score = argument_match_score(gold_arg, pred_arg)
            for ex_arg in ex_obj_names:
                score = max(score, argument_match_score(ex_arg, pred_arg))
            if score >= ARGUMENT_MATCH_THRESHOLD:
                scored_pairs.append((score, gold_idx, pred_idx))
    scored_pairs.sort(reverse=True)

    matched_gold = set()
    matched_pred = set()
    for _, gold_idx, pred_idx in scored_pairs:
        if gold_idx in matched_gold or pred_idx in matched_pred:
            continue
        matched_gold.add(gold_idx)
        matched_pred.add(pred_idx)

    obj_right = len(matched_gold)

    obj_precision = obj_right / obj_tagged if obj_tagged > 0 else 0
    obj_recall = obj_right / obj_true if obj_true > 0 else 0
    obj_f1 = 2 * obj_precision * obj_recall / (obj_precision + obj_recall) if (obj_precision + obj_recall) > 0 else 0

    return obj_right, obj_true, obj_tagged,obj_f1

def match(act, pred, words):
    act_idx = act['act_idx']
    act_name = words[act_idx]
    
    pred_act_name = pred.get('verb', None)
    if pred_act_name is None:
        return False, 0, 0, 0, 0
    
    act_lemma = nlp(act_name)[0].lemma_.lower()
    doc2 = nlp(pred_act_name)
    pred_act_lemma = " ".join([token.lemma_.lower() for token in doc2])
    if not act_lemma in pred_act_lemma:
        return False, 0, 0, 0, 0
    
    
    act_obj_names = [[words[ind] for ind in act['obj_idxs'][0]], [words[ind] for ind in act['obj_idxs'][1]]]
    pred_obj_names = pred.get('arguments', [])

    obj_right, obj_true, obj_tagged, obj_f1 = match_objs(act_obj_names, pred_obj_names)
    return True, obj_right, obj_true, obj_tagged, obj_f1



def evaluation(preds, names=("", "", ""), collect_diagnostics=False):
    

    total_right = total_truth = total_tagged = 0
    obj_total_right = obj_total_truth = obj_total_tagged = 0
    diagnostics = []

    for item_idx, item in enumerate(tqdm(preds, desc="Processing", unit="item")):
        
        words = item['words']
        acts = item['acts']
        pred = item['pred']

        
        counted_exclusive_acts = set()

        if not pred:
            print(f"No predictions found for item {item_idx}.")
            pred = []
        used = [False] * len(pred)
        for act in acts:
            matched = False

            act_type = act['act_type']
            if act_type == 1:
                total_truth += 1
            elif act_type == 3:
                act_idx = act['act_idx']
                related_act_indices = act['related_acts']
                all_indices = set(related_act_indices + [act_idx])
                if all_indices.isdisjoint(counted_exclusive_acts):
                    total_truth += 1
                    counted_exclusive_acts.update(all_indices)
            
            best = (None, 0,0,0,0)
            for pred_idx, pred_act in enumerate(pred):
                if used[pred_idx]:
                    continue
                matched, obj_right, obj_true, obj_tagged, obj_f1 = match(act, pred_act, words)

                if matched:
                    if best[4] < obj_f1:
                        best = (pred_idx, obj_right, obj_true, obj_tagged, obj_f1)
            
            # matched the best prediction
            if best[0] is not None:
                if act_type == 2:
                        total_truth += 1
                total_right += 1
                    
                    
                obj_total_tagged += best[3]
                obj_total_truth += best[2]
                obj_total_right += best[1]
                    
                used[best[0]] = True
                if collect_diagnostics and (best[1] < best[2] or best[1] < best[3]):
                    gold = action_record(act, words)
                    pred_act = pred[best[0]]
                    arg_info = classify_argument_mismatch(gold["arguments"], normalize_args(pred_act.get("arguments", [])))
                    diagnostics.append(
                        diagnostic_row(
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
                    )
            elif collect_diagnostics:
                gold = action_record(act, words)
                unused_preds = [(idx, pred_act) for idx, pred_act in enumerate(pred) if not used[idx]]
                candidate, score = best_verb_candidate(gold, unused_preds)
                if candidate and score > 0:
                    pred_idx, pred_act = candidate
                    diagnostics.append(
                        diagnostic_row(
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
                    )
                else:
                    diagnostics.append(
                        diagnostic_row(
                            names,
                            item,
                            item_idx,
                            "unmatched_gold_action",
                            gold=gold,
                            dataset_issue="extra_actions",
                            llm_issue="missing_actions",
                            reason="gold action was not matched by any prediction",
                        )
                    )

        total_tagged += len(pred)

        if collect_diagnostics:
            for pred_idx, pred_act in enumerate(pred):
                if used[pred_idx]:
                    continue
                pred_verb = pred_act.get("verb", "")
                if pred_verb and lemma_text(pred_verb) in lemma_text(original_text(item)):
                    dataset_issue = "missing_actions"
                    llm_issue = "extra_actions"
                    reason = "unused prediction verb appears in original text; annotation may have missed this action"
                else:
                    dataset_issue = ""
                    llm_issue = "extra_actions"
                    reason = "unused prediction did not match any gold action"
                diagnostics.append(
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

    precision = total_right / total_tagged if total_tagged > 0 else 0
    recall = total_right / total_truth if total_truth > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    obj_precision = obj_total_right / obj_total_tagged if obj_total_tagged > 0 else 0
    obj_recall = obj_total_right / obj_total_truth if obj_total_truth > 0 else 0
    obj_f1 = 2 * obj_precision * obj_recall / (obj_precision + obj_recall) if (obj_precision + obj_recall) > 0 else 0
    

    if (precision == 0 or recall == 0):
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
            precision, recall, f1, obj_precision, obj_recall, obj_f1 = metrics
        else:
            precision, recall, f1, obj_precision, obj_recall, obj_f1 = evaluation(raw_res)
        results[(ds_name, solver_name, model_name)] = (precision, recall, f1, obj_precision, obj_recall, obj_f1)
    if collect_diagnostics:
        return results, all_diagnostics
    return results



def main(args):
    # Debug mode
    if args.debug:
        global DEBUG
        DEBUG = True
        print('Debug mode is on!')

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
    print('Evaluation done!')
    write_results(results, dir)
    if args.diagnostics:
        write_diagnostics(diagnostics, dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='./results', help='results directory')
    parser.add_argument('--diagnostics', action='store_true', help='write mismatch diagnostics for annotation/LLM error analysis')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()
    main(args)
