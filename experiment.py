
import json
import os, sys
from src.utils import load_pkl
from src.solvers import NL2P_1, NL2P_1_Ablation, NL2P_2, NL2P_3, VerbArgs, GPT3ToPlan
from tqdm import tqdm

DEBUG = False

DATA_DIR = os.path.join(".", "data", "easdrl")
RESULTS_DIR = os.path.join(".", "results")

DATASETS = {
    "cooking": "cooking_labeled_text_data",
    "wikihow": "wikihow_labeled_text_data",
    "win2k": "win2k_labeled_text_data",
}

MODELS = [
    "gpt-5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gemma3",
    "gemma3:12b",
    "llama3.2",
]


def dataset_path(filename):
    return os.path.join(DATA_DIR, filename + ".pkl")


def apply_limit(dataset, limit=None):
    if limit is None:
        return dataset
    if limit < 0:
        raise ValueError("limit must be non-negative")
    return dataset[:limit]


def sample_to_sentences(sample):
    return [" ".join(sent) for sent in sample["sents"]]


def sample_to_paragraph(sample):
    sentences = sample_to_sentences(sample)
    return ". ".join(sentences) + "."


def sample_gold_actions(sample):
    words = sample.get("words", [])
    gold = []
    for act in sample.get("acts", []):
        act_idx = act.get("act_idx")
        obj_idxs = act.get("obj_idxs", [[], []])
        gold.append(
            {
                "act_idx": act_idx,
                "verb": words[act_idx] if isinstance(act_idx, int) and act_idx < len(words) else None,
                "obj_idxs": obj_idxs,
                "arguments": [
                    [words[idx] for idx in group if isinstance(idx, int) and idx < len(words)]
                    for group in obj_idxs
                ],
                "act_type": act.get("act_type"),
                "related_acts": act.get("related_acts", []),
            }
        )
    return gold


def build_result_record(ds_name, doc_id, source_file, sample, prediction):
    return {
        "dataset": ds_name,
        "doc_id": doc_id,
        "sentences": sample_to_sentences(sample),
        "prediction": prediction,
        "gold_actions": sample_gold_actions(sample),
    }


def read_from_refined_dataset(filename, limit=None):
    """
    Read data from a dataset file
    """
    path = dataset_path(filename)
    dataset = load_pkl(path)[-1]
    return apply_limit(dataset, limit)

def read_from_labeled_dataset(filename, limit=None):
    """
    Read data from a dataset file
    """
    path = dataset_path(filename)
    dataset = load_pkl(path)
    return apply_limit(dataset, limit)

def refine_results(raw_res):
    return raw_res

def run_experiment(dataset, solver, ds_name="", source_file=""):
    results = []
    for i in tqdm(range(len(dataset)), desc="Processing instances", unit="sample"):
        sample = dataset[i]
        paragraph = sample_to_paragraph(sample)

        raw_res = solver.solve(paragraph, ds_name=ds_name)
        res = refine_results(raw_res)
        sample["pred"] = res
        sample["doc_id"] = i
        sample["docId"] = f"{ds_name}:{i}"
        sample["domain"] = ds_name
        sample["source_file"] = source_file
        sample["original_text"] = paragraph
        results.append(build_result_record(ds_name, i, source_file, sample, res))

    return results

def write_results(ds_name, solver_name, results, model_name=""):
    model_name = (model_name or "").replace(":", "_")
    out_dir = os.path.join(RESULTS_DIR, solver_name, model_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    outpath = os.path.join(out_dir, ds_name + '_' + solver_name + '_' + (model_name if model_name else '') + '.json')
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print('Results written to %s' % outpath)

def write_pkl_results(ds_name, solver_name, results, model_name=""):
    model_name = (model_name or "").replace(":", "_")
    out_dir = os.path.join(RESULTS_DIR, solver_name, model_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    outpath = os.path.join(out_dir, ds_name + '_' + solver_name + '_' + (model_name if model_name else '') + '.pkl')
    import pickle
    with open(outpath, 'wb') as f:
        pickle.dump(results, f)
    print('Results written to %s' % outpath)

def write_summary(ds_name, solver_name, results, model_name=""):
    model_name = (model_name or "").replace(":", "_")
    out_dir = os.path.join(RESULTS_DIR, solver_name, model_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    outpath = os.path.join(out_dir, ds_name + '_' + solver_name + '_' + (model_name if model_name else '') + '_summary.json')
    summary = {
        "dataset": ds_name,
        "solver": solver_name,
        "model": model_name,
        "num_docs": len(results),
        "doc_ids": [item["doc_id"] for item in results],
        "source_file": results[0].get("source_file") if results else None,
    }
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print('Summary written to %s' % outpath)

def main(args):
    # Debug mode
    if args.debug:
        global DEBUG
        DEBUG = True
        if not args.l:
            args.l = 2
        print('Debug mode is on!')
    if args.d:
        if args.d not in DATASETS:
            print('Dataset %s not found!' % args.d)
            sys.exit(1)
        target_ds = {args.d: read_from_labeled_dataset(DATASETS[args.d], limit=args.l)}
    else:
        target_ds = {k: read_from_labeled_dataset(v, limit=args.l) for k, v in DATASETS.items()}

    if args.m and args.m not in MODELS:
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
        case 'nl2p_1_ablation':
            if not model_name:
                print('Please specify a model name for llm based solver!')
                sys.exit(1)
            solver = NL2P_1_Ablation(model_name=model_name)
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
        source_file = dataset_path(DATASETS[ds_name])
        results = run_experiment(dataset, solver, ds_name=ds_name, source_file=source_file)
        write_results(ds_name, solver_name, results, model_name)
        write_pkl_results(ds_name, solver_name, dataset, model_name)
        write_summary(ds_name, solver_name, results, model_name)
        print('Experiment on %s dataset (%s, %s) done!' % (ds_name, solver_name, model_name if model_name else ''))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, default='nl2p_1', help='solvers: gpt3_to_plan, nl2p_1, nl2p_1_ablation, nl2p_2, nl2p_3, verb_args')
    parser.add_argument('-m', type=str, help='optional, for llm based solve only, model name: gpt-5-mini, gpt-4.1-mini...')
    parser.add_argument('-d', type=str, help='dataset: cooking,wikihow,win2k')
    parser.add_argument('-l', type=int, help='limit the number of instances to run')
    parser.add_argument('-t', type=int, help='temperature for llm based solver')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()
    main(args)
