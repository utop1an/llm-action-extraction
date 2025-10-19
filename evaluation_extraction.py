
from collections import defaultdict
import csv
import json
import os, sys
from utils import load_pkl
from solvers.gpt3_to_plan import GPT3ToPlan
from solvers.nl2p import NL2P
from solvers.ceasdrl import cEASDRL
from solvers.naruto import Naruto

DEBUG = False

# class Dataset:
#     def __init__(self, filename):
#         self.name = filename
#         self.data = read_from_dataset(filename + '.pkl')

#     def get_paragraph(self, idx):
#         paragraph = []
#         for j in range(len(self.data[idx])):
#             item = self.data[idx][j]
#             paragraph += item['this_sent']
#         return paragraph

#     def get_sent(self, i, j):
#         item = self.data[i][j]
#         return item['this_sent']

#     def get_sent_item(self, i, j):
#         return self.data[i][j]
    
#     def get_acts(self, i, j):
#         item = self.data[i][j]
#         acts = {}
#         for k in range(len(item['acts'])):
#             act_idx = item['acts'][k]['act_idx']
#             act_name = item['this_sent'][act_idx]
            
#             obj_inds = item['acts'][k]['obj_idxs']
#             obj_names = [item['this_sent'][ind] for ind in obj_inds[0]]
#             acts[act_name] = obj_names
#         return acts




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

def read_from_predicted_dataset():
    dir = './results'
    res_dict = defaultdict(list)
    if not os.path.exists(dir):
        raise FileNotFoundError(f"The results dir {dir} does not exist.")
    files = os.listdir(dir)
    pkl_files = [f for f in files if f.endswith('.pkl')]
    for file in pkl_files:
        path = os.path.join(dir, file)
        full_filename = file.split('.')[0].split('_')
        ds_name = full_filename[0]
        solver_name = full_filename[1]
        model_name = full_filename[2] if len(full_filename) > 2 else 'none'
        res_dict[(ds_name,solver_name,model_name)] = load_pkl(path)
    return res_dict

def refine_results(raw_res):
    return raw_res
def match_objs(act_objs, pred_objs, words):
    act_obj_names = [words[ind] for ind in act_objs]
    pred_obj_names = [words[ind] for ind in pred_objs]
    if set(act_obj_names) != set(pred_obj_names):
        return False
    return True


def match(act, pred, words):
    act_idx = act['act_idx']
    act_name = words[act_idx]
    pred_idx = pred['act_idx']
    pred_name = words[pred_idx]
    if act_name != pred_name:
        return False
    act_obj_idxs = act['obj_idxs'][0]
    pred_obj_idxs = pred['obj_idxs'][0]
    if set(act_obj_idxs) != set(pred_obj_idxs):
        return False
    return True

def evaluation(dataset, preds):
    tp, fp, tn, fn = 0, 0, 0, 0
    recall, precision, f1 = 0.0, 0.0, 0.0
    for item in preds:
        words = item['words']
        acts = item['acts']
        pred = item['pred']
        seq_act = 0
        seq_pred_verb = 0
        while seq_act < len(acts):
            act_idx = acts[seq_act]['act_idx']
            act_name = words[act_idx]
            obj_idxs = acts[seq_act]['obj_idxs']
            obj_names = [words[ind] for ind in obj_idxs[0]]
            matched = False

            while seq_pred_verb < len(pred):
                matched = match(acts[seq_act], pred[seq_pred_verb], words)
                if matched:
                    seq_act += 1
                    seq_pred_verb += 1
                    tp += 1
                    break
            if not matched:
                fn += 1
                seq_act += 1
            
        
    return (tp, fp, tn, fn, precision, recall, f1)

def run_evaluation(datasets, predicates):
    results = {}
    for names, raw_res in predicates.items():
        ds_name, solver_name, model_name = names
        if ds_name not in datasets:
            print(f"Dataset {ds_name} not found in provided datasets. Skipping evaluation for this entry.")
            continue
        dataset = datasets[ds_name]
        res = evaluation(dataset, raw_res)
        results[(ds_name, solver_name, model_name)] = res
    return results

def write_results( results: dict):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    outpath = os.path.join('./results/evaluation_result.csv')
    with open(outpath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'solver', 'model', 'TP', 'FP', 'TN', 'FN', 'Precision', 'Recall', 'F1'])
        for k,v in results.items():
            ds_name, solver, model_name = k
            tp, fp, tn, fn, precision, recall, f1 = v
            writer.writerow([ds_name, solver, model_name, tp, fp, tn, fn, precision, recall, f1])
    print('Results written to %s' % outpath)

def main(args):
    # Define datasets
    datasets = {
        'cooking': 'cooking_labeled_text_data',
        'wikihow': 'wikihow_labeled_text_data',
        'win2k': 'win2k_labeled_text_data'
    }
    if args.d:
        if args.d not in datasets:
            print('Dataset %s not found!' % args.d)
            sys.exit(1)
        target_ds = [datasets[args.d]]
    else:
        target_ds = datasets.values()

    # Debug mode
    if args.debug:
        global DEBUG
        DEBUG = True
        print('Debug mode is on!')

    datasets = {}
    for ds_name in target_ds:
        datasets[ds_name] = read_from_labeled_dataset(ds_name, limit=args.l)
    predicates = read_from_predicted_dataset(ds_name + '_refined', limit=args.l)
    results = run_evaluation(datasets, predicates)
    print('Evaluation done!')
    write_results(results)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, help='optional, for llm based solve only, model name: gpt-4o-mini...')
    parser.add_argument('-d', type=str, help='dataset: cookin,wikihow,win2k')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()
    main(args)