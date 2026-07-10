from pathlib import Path

import pandas as pd


DOMAIN_LABELS = {
    "win2k": "WHS",
    "cooking": "CT",
    "wikihow": "WHG",
}
DOMAIN_ORDER = ["win2k", "cooking", "wikihow"]

MODEL_LABELS = {
    "gemma3-12b": "Gemma3-12B",
    "gemma3-27b": "Gemma3-27B",
    "llama3-70b": "Llama3.3-70B",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-mini": "GPT-5.4-mini",
}
MODEL_ORDER = [
    "gemma3-12b",
    "gemma3-27b",
    "llama3-70b",
    "gpt-5.4",
    "gpt-5.4-mini",
]

METHOD_LABELS = {
    "nl2p_1": "NL2P-1",
    "nl2p_1_ablation": "Ablation",
    "nl2p_1_coref": "Coref",
}
METHOD_ORDER = ["nl2p_1", "nl2p_1_ablation", "nl2p_1_coref"]


def read_eval_results(result_dir, domain_labels, method_labels, model_order=None):
    """Read evaluation_result.csv files into one flat DataFrame."""
    result_dir = Path(result_dir)
    rows = []

    for method_key in method_labels.keys():
        method_root = result_dir / method_key
        if not method_root.exists():
            print(f"Missing result directory: {method_root}")
            continue
        
        model_dirs = sorted(path for path in method_root.iterdir() if path.is_dir())
        for model_dir in model_dirs:
            csv_path = model_dir / "evaluation_result.csv"
            if not csv_path.exists():
                print(f"Missing evaluation_result.csv for {model_dir.name}")
                continue

            for _, row in pd.read_csv(csv_path).iterrows():
                domain_key = str(row["dataset"]).strip()
                if domain_key not in domain_labels:
                    continue

                result_row = {
                    "domain": domain_key,
                    "method": method_key,
                    "model": model_dir.name,
                    "Action Precision": float(row["Precision"]),
                    "Action Recall": float(row["Recall"]),
                    "Action F1": float(row["F1"]),
                    "Argument Precision": float(row["Object Precision"]),
                    "Argument Recall": float(row["Object Recall"]),
                    "Argument F1": float(row["Object F1"]),
                }
                for output_column, csv_column in [
                    ("Adjusted Argument Precision", "adjusted_precision"),
                    ("Adjusted Argument Recall", "adjusted_recall"),
                    ("Adjusted Argument F1", "adjusted_f1"),
                ]:
                    if csv_column in row:
                        result_row[output_column] = float(row[csv_column])

                for column in [
                    "perfect_action_argument_matches",
                    "argument_mismatch_actions",
                    "matched_action_events",
                ]:
                    if column in row:
                        result_row[column] = int(row[column])
                rows.append(result_row)

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df

    if model_order is not None:
        result_df = result_df[result_df["model"].isin(model_order)].copy()
        result_df["model"] = pd.Categorical(result_df["model"], model_order, ordered=True)

    domain_order = [key for key in DOMAIN_ORDER if key in domain_labels]
    method_order = [key for key in METHOD_ORDER if key in method_labels]
    method_order += [key for key in method_labels if key not in method_order]
    result_df["domain"] = pd.Categorical(result_df["domain"], domain_order, ordered=True)
    result_df["method"] = pd.Categorical(result_df["method"], method_order, ordered=True)
    return result_df.sort_values(["domain", "model", "method"]).reset_index(drop=True)



def read_diagnostics_by_mismatch_type(result_dir, domain_labels, method_labels, model_order=None, mismatch_type=None):
    """
    Read diagnostic rows based on mismatch type into one flat DataFrame.
    Default `mismatch_type = None` will read all diagnostic rows
    """
    result_dir = Path(result_dir)
    rows = []

    for method_key in method_labels:
        method_root = result_dir / method_key
        if not method_root.exists():
            print(f"Missing result directory: {method_root}")
            continue

        for model_dir in sorted(path for path in method_root.iterdir() if path.is_dir()):
            csv_path = model_dir / "evaluation_mismatch_diagnostics.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            if mismatch_type is not None:
                df = df[df["mismatch_type"] == mismatch_type].copy()
            
            for _, row in df.iterrows():
                domain_key = str(row["dataset"]).strip()
                if domain_key not in domain_labels:
                    continue

                issue_text = "|".join(
                    str(row.get(column, ""))
                    for column in [
                        "strong_dataset_issue",
                        "candidate_dataset_issue",
                        "candidate_llm_issue",
                    ]
                    if pd.notna(row.get(column, ""))
                )
                rows.append(
                    {
                        "domain": domain_key,
                        "method": method_key,
                        "model": model_dir.name,
                        "issue_text": issue_text,
                    }
                )

    diagnostics_df = pd.DataFrame(rows)
    if diagnostics_df.empty:
        return diagnostics_df

    if model_order is not None:
        diagnostics_df = diagnostics_df[diagnostics_df["model"].isin(model_order)].copy()
        diagnostics_df["model"] = pd.Categorical(diagnostics_df["model"], model_order, ordered=True)

    domain_order = [key for key in DOMAIN_ORDER if key in domain_labels]
    method_order = [key for key in METHOD_ORDER if key in method_labels]
    diagnostics_df["domain"] = pd.Categorical(diagnostics_df["domain"], domain_order, ordered=True)
    diagnostics_df["method"] = pd.Categorical(diagnostics_df["method"], method_order, ordered=True)
    return diagnostics_df.sort_values(["domain", "model", "method"]).reset_index(drop=True)
