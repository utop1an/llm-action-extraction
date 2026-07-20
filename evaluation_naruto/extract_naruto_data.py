import ast
import os
import pandas as pd

def split_miglani(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    df["group_key"] = df["d_id"].astype(str).str.replace(r"\d+$", "", regex=True)

    print(f"Duplicated rows: {df.duplicated().sum()}")
    clean_df = df.drop_duplicates()

    for group_key, group in clean_df.groupby("group_key"):
        filename = f"{output_dir}/{group_key}_naruto_naruto.csv"
        group["event_words"] = group["event_words"].apply(ast.literal_eval)
        group["join_key"] = group["event_words"].apply(
            lambda x: "".join(x).replace(" ", "").lower()
        )

        group.to_csv(filename, index=False)
        print(f"Saved: {filename}")


def extraction(pred_dict, output_dir):
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


def contains_csv(dir):
    return any(filename.endswith(".csv") for filename in os.listdir(dir))