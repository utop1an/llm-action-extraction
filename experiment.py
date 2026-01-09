
import json
import os, sys
from src.utils import load_pkl
from src.solvers import NL2P_1, NL2P_2, NL2P_3, VerbArgs, GPT3ToPlan
from tqdm import tqdm

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

def refine_results(raw_res):
    return raw_res

def run_experiment(dataset, solver, ds_name=""):
    results = []
    for i in tqdm(range(len(dataset)), desc="Processing instances", unit="sample"):
        sents = dataset[i]['sents']
        sentences = []
        for sent in sents:
            sentence = " ".join(sent)
            sentences.append(sentence)
        paragraph = '. '.join(sentences) + '.'

        raw_res = solver.solve(paragraph, ds_name=ds_name)
        results.append(raw_res)
        res = refine_results(raw_res)
        dataset[i]['pred'] = res

    return results

def write_results(ds_name, solver_name, results, model_name=""):
    if not os.path.exists("./results/%s/%s" % (solver_name, model_name)):
        os.makedirs("./results/%s/%s" % (solver_name, model_name))
    outpath = os.path.join('./results/%s/%s' % (solver_name, model_name), ds_name + '_' + solver_name + '_' + (model_name if model_name else '') + '.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=4)
    print('Results written to %s' % outpath)

def write_pkl_results(ds_name, solver_name, results, model_name=""):
    if not os.path.exists("./results/%s/%s" % (solver_name, model_name)):
        os.makedirs("./results/%s/%s" % (solver_name, model_name))
    outpath = os.path.join('./results/%s/%s' % (solver_name, model_name), ds_name + '_' + solver_name + '_' + (model_name if model_name else '') + '.pkl')
    import pickle
    with open(outpath, 'wb') as f:
        pickle.dump(results, f)
    print('Results written to %s' % outpath)

def main(args):
    # Debug mode
    if args.debug:
        global DEBUG
        DEBUG = True
        if not args.l:
            args.l = 2
        print('Debug mode is on!')
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
        target_ds = {args.d: read_from_labeled_dataset(datasets[args.d], limit=args.l)}
    else:
        target_ds = {k: read_from_labeled_dataset(v, limit=args.l) for k, v in datasets.items()}

    # Define models
    models = ['gpt-4o-mini', 'gpt-5-mini', 'gpt-5-nano', "gemma3", "gemma3:12b"]
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
            solver = GPT3ToPlan(datasets=target_ds, model_name=model_name)
        case 'nl2p_1':
            if not model_name:
                print('Please specify a model name for llm based solver!')
                sys.exit(1)
            solver = NL2P_1(model_name=model_name)
        case 'nl2p_2':
            if not model_name:
                print('Please specify a model name for llm based solver!')
                sys.exit(1)
            solver = NL2P_2(model_name=model_name)
        case 'nl2p_3':
            if not model_name:
                print('Please specify a model name for llm based solver!')
                sys.exit(1)
            solver = NL2P_3(model_name=model_name)
        case 'verb_args':
            if not model_name:
                print('Please specify a model name for llm based solver!')
                sys.exit(1)
            solver = VerbArgs(model_name=model_name)
        case _:
            print('Unknown solver: %s' % solver_name)
            sys.exit(1)

    print("Starting experiment with solver: %s, model: %s" % (solver_name, model_name if model_name else ''))
    for ds_name, dataset in target_ds.items():
        print('Running experiment on %s dataset...' % ds_name)
        results = run_experiment(dataset, solver, ds_name=ds_name)
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
    parser.add_argument('-t', type=int, help='temperature for llm based solver')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()
    main(args)