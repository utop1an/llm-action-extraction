import sys
import os

# 1. Get the folder of the current file (subfolder)
current = os.path.dirname(os.path.realpath(__file__))

# 2. Get the parent folder (root)
parent = os.path.dirname(current)

# 3. Add root to the search path
sys.path.append(parent)

from evaluation import read_from_predicted_dataset
import pandas as pd

target_dir = './results/gpt3-to-plan'
output_dir = './helper'

def extraction(pred_dict):
    all_rows = []
    for names, raw_res in pred_dict.items():
        ds_name, solver_name, model_name = names
        for item in raw_res:
            row = item.copy()
            row['ds_name'] = ds_name
            all_rows.append(row)
    df = pd.DataFrame(all_rows)
    df.drop(columns='pred', inplace=True, errors='ignore')
    col_data = df.pop('ds_name')
    df.insert(0, 'ds_name', col_data)
    output_path = os.path.join(output_dir, 'extracted_labels.csv')
    df.to_csv(output_path, index=False)


def main():
    res_dict = read_from_predicted_dataset(target_dir)
    extraction(res_dict)


if __name__ == "__main__":
    main()