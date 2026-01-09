
from collections import defaultdict
import csv
import os, sys

from tqdm import tqdm
from src.utils import load_pkl

import spacy

nlp = spacy.load("en_core_web_sm")

DEBUG = False

def read_from_refined_dataset(filename, limit=None):
    """
    Read data from a dataset file
    """
    path = os.path.join('./data/easdrl', filename + '.pkl')
    dataset = load_pkl(path)[-1]
    if limit is not None:
        dataset = dataset[:(max(limit, len(dataset)))]
    return dataset

def read_from_labeled_dataset(filename, limit=None):
    """
    Read data from a dataset file
    """
    path = os.path.join('./data/easdrl', filename + '.pkl')
    dataset = load_pkl(path)
    if limit is not None:
        dataset = dataset[:(max(limit, len(dataset)))]
    return dataset

def read_from_predicted_dataset(dir):
    res_dict = defaultdict(list)
    if not os.path.exists(dir):
        raise FileNotFoundError(f"The results dir {dir} does not exist.")
    files = os.listdir(dir)
    pkl_files = [f for f in files if f.endswith('.pkl')]
    for file in pkl_files:
        print(f"Loading {file}")
        path = os.path.join(dir, file)
        full_filename = file.split('.')[0].split('_')
        ds_name = full_filename[0]
        solver_name = full_filename[1]
        model_name = full_filename[2] if len(full_filename) > 2 else 'none'
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

def match_obj(gt_name, pred_name):
    gt_lemma = nlp(gt_name)[0].lemma_.lower()
    doc2 = nlp(pred_name)
    pred_lemma = " ".join([token.lemma_.lower() for token in doc2])
    return gt_lemma in pred_lemma

def match_objs(act_obj_names, pred_obj_names):
    common = 0
    gt_pointer = 0
    pred_pointer = 0
    while gt_pointer < len(act_obj_names):
        matched = False
        while pred_pointer < len(pred_obj_names):
            matched = match_obj(act_obj_names[gt_pointer], pred_obj_names[pred_pointer])
            if matched:
                common += 1
                gt_pointer += 1
                pred_pointer += 1
                break
            else:
                pred_pointer += 1
        if not matched:
            gt_pointer += 1
    tp = common
    fp = len(pred_obj_names) - common
    fn = len(act_obj_names) - common

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def match(act, pred, words):
    act_idx = act['act_idx']
    act_name = words[act_idx]
    
    pred_act_name = pred['verb']
    
    act_lemma = nlp(act_name)[0].lemma_.lower()
    doc2 = nlp(pred_act_name)
    pred_act_lemma = " ".join([token.lemma_.lower() for token in doc2])
    if not act_lemma in pred_act_lemma:
        return False, None, None, None
    
    
    act_obj_names = [words[ind] for ind in act['obj_idxs'][0]]
    pred_obj_names = pred['arguments']

    obj_precision, obj_recall, obj_f1 = match_objs(act_obj_names, pred_obj_names)
    return True, obj_precision, obj_recall, obj_f1

def get_total_truth(acts):
    total_truth = 0
    counted = set()
    for act in acts:
        if (act['act_type'] == 1) or (act['act_type'] == 2):
            total_truth += 1
        else:
            act_idx = act['act_idx']
            related_act_indices = act['related_acts']
            all_indices = set(related_act_indices + [act_idx])
            if all_indices.isdisjoint(counted):
                total_truth += 1
                counted.update(all_indices)
    return total_truth

def evaluation(preds):
    precisions, recalls, f1s = [],[],[]
    obj_precisions, obj_recalls, obj_f1s = [],[],[]
    for i, item in enumerate(tqdm(preds, desc="Processing", unit="item")):
        
        words = item['words']
        acts = item['acts']
        pred = item['pred']

        total_right = 0
        total_truth = 0
        counted_exclusive_acts = set()

        if not pred:
            print(f"No predictions found for item {i}.")
            continue
        pred_pointer = 0
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
            
            
            for pred_act_idx in range(pred_pointer, len(pred)):
                matched, obj_precision, obj_recall, obj_f1 = match(act, pred[pred_act_idx], words)


                if matched:
                    if act_type == 2:
                        total_truth += 1
                    obj_precisions.append(obj_precision)
                    obj_recalls.append(obj_recall)
                    obj_f1s.append(obj_f1)
                    total_right += 1
                    pred_pointer = pred_act_idx + 1
                    break

        total_tagged = len(pred)

        precision = total_right / total_tagged if total_tagged > 0 else 0
        precisions.append(precision)
        recall = total_right / total_truth if total_truth > 0 else 0
        recalls.append(recall)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1s.append(f1)
    
    avg_precision = sum(precisions) / len(precisions) if len(precisions) > 0 else 0
    avg_recall = sum(recalls) / len(recalls) if len(recalls) > 0 else 0
    avg_f1 = sum(f1s) / len(f1s) if len(f1s) > 0 else 0

    avg_obj_precision = sum(obj_precisions) / len(obj_precisions) if len(obj_precisions) > 0 else 0
    avg_obj_recall = sum(obj_recalls) / len(obj_recalls) if len(obj_recalls) > 0 else 0
    avg_obj_f1 = sum(obj_f1s) / len(obj_f1s) if len(obj_f1s) > 0 else 0

    if (precision == 0 or recall == 0):
        print("warning: zero precision or recall")

    return avg_precision, avg_recall, avg_f1, avg_obj_precision, avg_obj_recall, avg_obj_f1

def run_evaluation(predicates):
    results = {}
    for names, raw_res in predicates.items():
        ds_name, solver_name, model_name = names
        print(f"Evaluating {ds_name} with solver {solver_name} and model {model_name}")
        tp, fp, fn, precision, recall, f1 = evaluation(raw_res)
        results[(ds_name, solver_name, model_name)] = (tp, fp, fn, precision, recall, f1)
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
    results = run_evaluation(predicates)
    print('Evaluation done!')
    write_results(results, dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='./results', help='results directory')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()
    main(args)