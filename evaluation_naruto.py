import os,sys
from collections import defaultdict

from tqdm import tqdm
from evaluation import match, write_results
import spacy
import pandas as pd
import ast

nlp = spacy.load("en_core_web_sm")

gt_path = './extracted_labels.csv'

DEBUG = False

def make_join_key(val):
    if isinstance(val, list):
        return "".join(str(x) for x in val).replace(" ", "").lower()
    return ""


def read_from_naruto_predicted_dataset(dir):
    res_dict = defaultdict(list)
    if not os.path.exists(dir):
        raise FileNotFoundError(f"The results dir {dir} does not exist.")
    files = os.listdir(dir)
    csv_files = [f for f in files if f.endswith('.csv')]
    for file in csv_files:
        print(f"Loading {file}")
        path = os.path.join(dir, file)
        full_filename = file.split('.')[0].split('_')
        ds_name = full_filename[0]
        solver_name = full_filename[1]
        model_name = full_filename[2] if len(full_filename) > 2 else 'none'
        df = pd.read_csv(path)
        df['event_elements'] = df['event_elements'].apply(ast.literal_eval)
        df['subevent_elements'] = df['subevent_elements'].apply(ast.literal_eval)
        df['event_words'] = df['event_words'].apply(ast.literal_eval)
        df['join_key'] = df['event_words'].apply(make_join_key)
        res_dict[(ds_name,solver_name,model_name)] = df
    return res_dict


def get_naruto_event_verb(event):
    return event.get('verb', None)


def get_naruto_event_arguments(event):
    exclude_keys = {'verb', 'event_id'}
    return [v for k, v in event.items() if k not in exclude_keys]


def get_naruto_predicted_events(predicted_df):
    predicted_events = []
    for _, item in predicted_df.iterrows():
        event_verb = get_naruto_event_verb(item['event_elements'])
        event_args = get_naruto_event_arguments(item['event_elements'])
        predicted_events.append({'verb': event_verb, 'arguments': event_args})

        # subevent_elements = item['subevent_elements']
        # for _, subevent in subevent_elements.items():
        #     subevent_verb = get_naruto_event_verb(subevent)
        #     subevent_args = get_naruto_event_arguments(subevent)
        #     predicted_events.append({'verb': subevent_verb, 'arguments': subevent_args})

    return predicted_events


def naruto_evaluation(predicted_df, gt_df):

    total_right = total_truth = total_tagged = 0
    obj_total_right = obj_total_truth = obj_total_tagged = 0

    for i, item in (tqdm(gt_df.iterrows(), desc="Processing", unit="item")):
    # for i, item in gt_df.iterrows():
        acts = item['acts']
        words = item['words']
        words_key = item['join_key']

        # find matching preidicted rows
        matched_rows = predicted_df[predicted_df['join_key'].apply(lambda x: x in words_key)]

        # Collect all naruto results and format into pkl way
        pred = get_naruto_predicted_events(matched_rows)
        if not pred:
            print(f"No predictions found for item {i}.")
            continue

        counted_exclusive_acts = set()
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

            for i, pred_act in enumerate(pred):
                if used[i]:
                    continue
                matched, obj_right, obj_true, obj_tagged = match(act, pred_act, words)

                if matched:
                    if act_type == 2:
                        total_truth += 1
                    total_right += 1
                    
                    obj_total_tagged += obj_tagged
                    obj_total_truth += obj_true
                    obj_total_right += obj_right
                    
                    used[i] = True
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



def run_evaluation(predicates, gt_df):
    results = {}
    for names, predicted_df in predicates.items():
        ds_name, solver_name, model_name = names
        ds_gt_df = gt_df[gt_df['ds_name'] == ds_name]

        tp, fp, fn, precision, recall, f1 = naruto_evaluation(predicted_df, ds_gt_df)
        results[(ds_name, solver_name, model_name)] = (tp, fp, fn, precision, recall, f1)
    return results


def main(args):
    # # Debug mode
    # if args.debug:
    #     global DEBUG
    #     DEBUG = True
    #     print('Debug mode is on!')

    # dir = args.d
    # if not os.path.exists(dir):
    #     print(f"The results dir {dir} does not exist.")
    #     sys.exit(1)
    dir = './results/naruto'
    output_dir = './results/naruto/output'
    os.makedirs(output_dir, exist_ok=True)

    predicts = read_from_naruto_predicted_dataset(dir)
    with open(gt_path, 'r') as f:
        gt_df = pd.read_csv(f)
        gt_df['words'] = gt_df['words'].apply(ast.literal_eval)
        gt_df['acts'] = gt_df['acts'].apply(ast.literal_eval)
        gt_df['join_key'] = gt_df['words'].apply(make_join_key)

    results = run_evaluation(predicts, gt_df)
    print('Evaluation done!')
    print(results)
    write_results(results, output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='./results', help='results directory')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()
    main(args)