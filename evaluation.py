
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
    obj_right  = 0
    gt_pointer = 0
    pred_pointer = 0

    es_obj_names = act_obj_names[0]
    ex_obj_names = act_obj_names[1]

    obj_true = len(es_obj_names)
    obj_tagged = len(pred_obj_names)
    if obj_tagged == 0:
        return 0, obj_true, obj_tagged

    while gt_pointer < len(es_obj_names):
        matched = False
        while pred_pointer < len(pred_obj_names):
            matched = match_obj(es_obj_names[gt_pointer], pred_obj_names[pred_pointer])
            if matched:
                obj_right += 1
                gt_pointer += 1
                pred_pointer += 1
                break
            else:
                if (len(ex_obj_names)>0):
                    ex_matched = False
                    for ex_obj in ex_obj_names:
                        ex_matched = match_obj(ex_obj, pred_obj_names[pred_pointer])
                        if ex_matched:
                            obj_right += 1
                            gt_pointer += 1
                            pred_pointer += 1
                            break
                    if ex_matched:
                        break
                    else:
                        pred_pointer += 1
                else:
                    pred_pointer += 1
        if not matched:
            gt_pointer += 1


    return obj_right, obj_true, obj_tagged

def match(act, pred, words):
    act_idx = act['act_idx']
    act_name = words[act_idx]
    
    pred_act_name = pred.get('verb', None)
    if pred_act_name is None:
        return False, 0, 0, 0
    
    act_lemma = nlp(act_name)[0].lemma_.lower()
    doc2 = nlp(pred_act_name)
    pred_act_lemma = " ".join([token.lemma_.lower() for token in doc2])
    if not act_lemma in pred_act_lemma:
        return False, 0, 0, 0
    
    
    act_obj_names = [[words[ind] for ind in act['obj_idxs'][0]], [words[ind] for ind in act['obj_idxs'][1]]]
    pred_obj_names = pred.get('arguments', [])

    obj_right, obj_true, obj_tagged = match_objs(act_obj_names, pred_obj_names)
    return True, obj_right, obj_true, obj_tagged



def evaluation(preds):
    

    total_right = total_truth = total_tagged = 0
    obj_total_right = obj_total_truth = obj_total_tagged = 0

    for i, item in enumerate(tqdm(preds, desc="Processing", unit="item")):
        
        words = item['words']
        acts = item['acts']
        pred = item['pred']

        
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
                matched, obj_right, obj_true, obj_tagged = match(act, pred[pred_act_idx], words)


                if matched:
                    if act_type == 2:
                        total_truth += 1
                    total_right += 1
                    pred_pointer = pred_act_idx + 1
                    
                    obj_total_tagged += obj_tagged
                    obj_total_truth += obj_true
                    obj_total_right += obj_right
                    
                    break

        total_tagged += len(pred)

    precision = total_right / total_tagged if total_tagged > 0 else 0
    recall = total_right / total_truth if total_truth > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    obj_precision = obj_total_right / obj_total_tagged if obj_total_tagged > 0 else 0
    obj_recall = obj_total_right / obj_total_truth if obj_total_truth > 0 else 0
    obj_f1 = 2 * obj_precision * obj_recall / (obj_precision + obj_recall) if (obj_precision + obj_recall) > 0 else 0
    

    if (precision == 0 or recall == 0):
        print("warning: zero precision or recall")

    return precision, recall, f1, obj_precision, obj_recall, obj_f1

def run_evaluation(predicates):
    results = {}
    for names, raw_res in predicates.items():
        ds_name, solver_name, model_name = names
        print(f"Evaluating {ds_name} with solver {solver_name} and model {model_name}")
        precision, recall, f1, obj_precision, obj_recall, obj_f1 = evaluation(raw_res)
        results[(ds_name, solver_name, model_name)] = (precision, recall, f1, obj_precision, obj_recall, obj_f1)
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