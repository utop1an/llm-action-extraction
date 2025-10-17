
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

def test(dataset, methodology):
    for i in range(len(dataset)):
        paragraph = []
        paragraph_acts = []
        for j in range(len(dataset[i])):
            item = dataset[i][j]
            sent = item['this_sent']
            paragraph += sent
            acts = []

            sent_long = item['last_sent'] + item['this_sent']

            for k in range(len(item['acts'])):
                act_idx = item['acts'][k]['act_idx']
                act_name = sent_long[act_idx]

                obj_inds = item['acts'][k]['obj_idxs']
                obj_names = [sent_long[ind] for ind in obj_inds[0]]
                acts.append((act_name, obj_names))
            paragraph_acts.append(acts)

def refine_results(raw_res):
    return raw_res

def evaluation(results, dataset):
    pass

def run_experiment(dataset, solver):
    results = []
    for i in range(len(dataset)):
        sents = dataset[i]['sents']
        sentences = []
        for sent in sents:
            sentence = " ".join(sent)
            sentences.append(sentence)
        paragraph = '. '.join(sentences) + '.'

        raw_res = solver.solve(paragraph)
        results.append(raw_res)
        res = refine_results(raw_res)
        dataset[i]['pred'] = res

    return results

def write_results(ds_name, solver_name, results, model_name=""):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    outpath = os.path.join('./results', ds_name + '_' + solver_name + '_' + (model_name if model_name else '') + '.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=4)
    print('Results written to %s' % outpath)

def write_pkl_results(ds_name, solver_name, results, model_name=""):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    outpath = os.path.join('./results', ds_name + '_' + solver_name + '_' + (model_name if model_name else '') + '.pkl')
    import pickle
    with open(outpath, 'wb') as f:
        pickle.dump(results, f)
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

    # Define models
    models = ['gpt-4o-mini']
    if args.m and args.m not in models:
        print('Model %s not found!' % args.m)
        sys.exit(1)
    model_name = args.m

    # Define solvers
    solver_name = args.s
    match solver_name:
        case 'gpt3_to_plan':
            if not model_name:
                print('Please specify a model name for llm based solver!')
                sys.exit(1)
            solver = GPT3ToPlan(model_name=model_name)
        case 'nl2p':
            if not model_name:
                print('Please specify a model name for llm based solver!')
                sys.exit(1)
            solver = NL2P(model_name=model_name)
        case 'ceasdrl':
            solver = cEASDRL()
        case 'naruto':
            solver = Naruto()
        case _:
            print('Unknown solver: %s' % solver_name)
            sys.exit(1)

    # Debug mode
    if args.debug:
        global DEBUG
        DEBUG = True
        if not args.l:
            args.l = 2
        print('Debug mode is on!')

    for ds_name in target_ds:
        dataset = read_from_labeled_dataset(ds_name, limit=args.l)
        results = run_experiment(dataset, solver)
        write_results(ds_name, solver_name, results, model_name)
        write_pkl_results(ds_name, solver_name, dataset, model_name)
        print('Experiment on %s dataset (%s, %s) done!' % (ds_name, solver_name, model_name if model_name else ''))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, default='nl2p', help='solvers: gp3_to_plan, ceasdrl, nl2p, naruto')
    parser.add_argument('-m', type=str, help='optional, for llm based solve only, model name: gpt-4o-mini...')
    parser.add_argument('-d', type=str, help='dataset: cookin,wikihow,win2k')
    parser.add_argument('-l', type=int, help='limit the number of instances to run')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()
    main(args)