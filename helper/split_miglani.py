import pandas as pd
import os
import ast

input_path = 'miglani_data/miglani_all_events.csv'
output_dir = 'results/naruto'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_path)
df['group_key'] = df['d_id'].astype(str).str.replace(r'\d+$', '', regex=True)

# Check for duplicated rows
print(f"Duplicated rows: {df.duplicated().sum()}")
clean_df = df.drop_duplicates()

# Iterate through groups and save them
for group_key, group in clean_df.groupby('group_key'):
    filename = f"{output_dir}/{group_key}_naruto_naruto.csv"
    group['event_words'] = group['event_words'].apply(ast.literal_eval)
    group['join_key'] = group['event_words'].apply(lambda x: "".join(x).replace(" ", "").lower())
    
    group.to_csv(filename, index=False)
    print(f"Saved: {filename}")