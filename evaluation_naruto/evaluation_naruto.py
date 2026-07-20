"""
Adapt Naruto CSV predictions for the format of the shared action evaluator.
"""

import ast
from fileinput import filename
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import evaluation, write_results
from src.evaluation_helpers import write_diagnostics
from extract_naruto_data import extraction, split_miglani, contains_csv


CURRENT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "results" / "gpt3_to_plan" / "gpt-5.4"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "naruto"
GROUND_TRUTH_PATH = CURRENT_DIR / "extracted_labels.csv"
OUTPUT_DIR = PREDICTIONS_DIR / "output"
MIGLANI_PATH = PROJECT_ROOT / "miglani_data" / "miglani_all_events.csv"
COLLECT_DIAGNOSTICS = False


def make_join_key(value):
    """Return a whitespace-insensitive key for tokenized text."""
    if isinstance(value, list):
        return "".join("".join(str(token).split()) for token in value).casefold()
    return ""


def _literal_value(value, *, column, path):
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse column '{column}' in {path}: {value!r}") from exc


def read_from_naruto_predicted_dataset(dir: Path):
    """Read and parse every Naruto prediction CSV in a directory."""
    def parse_naruto_result_filename(filename):
        parts = filename.stem.split("_")
        if len(parts) != 3:
            raise ValueError(
                f"Naruto result filename must be '<dataset>_<solver>_<model>.csv': {filename}"
            )
        dataset, solver, model = parts[:3]
        return dataset, solver, model
    
    results = {}
    for path in sorted(dir.glob("*.csv")):
        print(f"Loading {path.name}")
        names = parse_naruto_result_filename(path)
        frame = pd.read_csv(path)
        required = {"d_id", "s_id", "event_elements", "subevent_elements", "event_words"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"Naruto result {path} is missing required columns: {', '.join(sorted(missing))}"
            )
        for column in ("event_elements", "subevent_elements", "event_words"):
            frame[column] = frame[column].apply(
                lambda value, column=column: _literal_value(value, column=column, path=path)
            )
        frame["join_key"] = frame["event_words"].apply(make_join_key)
        results[names] = frame
    return results


def read_ground_truth_dataset(path: Path):
    """Read and parse the labeled CSV consumed by the adapter."""
    frame = pd.read_csv(path)
    required = {"ds_name", "words", "acts"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Ground-truth file {path} is missing required columns: {', '.join(sorted(missing))}"
        )
    for column in (name for name in ("words", "acts", "sents") if name in frame):
        frame[column] = frame[column].apply(
            lambda value, column=column: _literal_value(value, column=column, path=path)
        )
    return frame


def get_naruto_event_verb(event):
    if not isinstance(event, Mapping):
        raise ValueError(f"Naruto event must be a mapping, got {type(event).__name__}")
    return event.get("verb")


def get_naruto_event_arguments(event):
    if not isinstance(event, Mapping):
        raise ValueError(f"Naruto event must be a mapping, got {type(event).__name__}")
    excluded_keys = {"verb", "event_id"}
    return [value for key, value in event.items() if key not in excluded_keys]


def _naruto_event_record(event):
    return {
        "verb": get_naruto_event_verb(event),
        "arguments": get_naruto_event_arguments(event),
    }


def get_naruto_predicted_events(predicted_df):
    """Flatten main events and subevents while preserving row order."""
    predicted_events = []
    for _, item in predicted_df.iterrows():
        predicted_events.append(_naruto_event_record(item["event_elements"]))
        subevents = sorted(
            item["subevent_elements"].values(),
            key=lambda event: int(event["event_id"]),
        )
        predicted_events.extend(_naruto_event_record(event) for event in subevents)
    return predicted_events


def _sort_document_rows(rows):
    """Sort prediction rows by window and event order."""
    if rows.empty:
        return rows

    rows = rows.copy()
    numeric_s_ids = pd.to_numeric(rows["s_id"], errors="coerce")
    rows["_event_id"] = rows["event_elements"].apply(
        lambda event: int(event["event_id"])
    )
    if numeric_s_ids.notna().all():
        rows["_numeric_s_id"] = numeric_s_ids
        return (
            rows.sort_values(["_numeric_s_id", "_event_id"], kind="stable")
            .drop(columns=["_numeric_s_id", "_event_id"])
        )
    return rows.sort_values(["s_id", "_event_id"], kind="stable").drop(
        columns="_event_id"
    )


def _validate_document_windows(rows, words, document_id):
    """Check that d_id-selected windows occur in the aligned gold text."""
    gold_key = make_join_key(words)
    for row_index, window_key in rows["join_key"].items():
        if not window_key:
            raise ValueError(f"Naruto row {row_index} for {document_id} has empty event_words")
        if window_key not in gold_key:
            raise ValueError(
                f"Naruto row {row_index} selected by d_id={document_id!r} does not occur "
                "in the corresponding ground-truth text"
            )


def build_evaluation_items(ds_name, predicted_df, gt_df):
    """Convert one Naruto run to the shared evaluator's item structure."""
    evaluation_items = []
    document_ids = predicted_df["d_id"].astype(str)
    for document_number, (_, gold_item) in enumerate(gt_df.iterrows(), start=1):
        document_id = f"{ds_name}{document_number}"
        rows = _sort_document_rows(predicted_df.loc[document_ids == document_id])
        _validate_document_windows(rows, gold_item["words"], document_id)

        item = {
            "words": gold_item["words"],
            "acts": gold_item["acts"],
            "pred": get_naruto_predicted_events(rows),
            "doc_id": document_id,
            "docId": document_id,
            "original_text": " ".join(str(word) for word in gold_item["words"]),
        }
        if "sents" in gt_df.columns and isinstance(gold_item["sents"], list):
            item["sents"] = gold_item["sents"]
        evaluation_items.append(item)
    return evaluation_items


def run_evaluation(predicts, gt_df, collect_diagnostics=False):
    """Adapt Naruto runs and call the shared evaluator."""
    results = {}
    all_diagnostics = []
    for names, predicted_df in predicts.items():
        ds_name, solver_name, model_name = names
        print(f"Evaluating {ds_name} with solver {solver_name} and model {model_name}")
        ds_gt_df = gt_df.loc[gt_df["ds_name"] == ds_name]
        if ds_gt_df.empty:
            raise ValueError(f"No ground-truth documents found for dataset {ds_name!r}")

        items = build_evaluation_items(ds_name, predicted_df, ds_gt_df)
        if collect_diagnostics:
            metrics, diagnostics = evaluation(items, names=names, collect_diagnostics=True)
            all_diagnostics.extend(diagnostics)
        else:
            metrics = evaluation(items, names=names)
        results[names] = metrics

    if collect_diagnostics:
        return results, all_diagnostics
    return results


def main():
    if not GROUND_TRUTH_PATH.is_file():
        extraction(DATA_DIR, GROUND_TRUTH_PATH)
    if not contains_csv(PREDICTIONS_DIR):
        if not MIGLANI_PATH.is_file():
            raise FileNotFoundError(f"The Miglani dataset file {MIGLANI_PATH} does not exist.")
        split_miglani(MIGLANI_PATH, PREDICTIONS_DIR)
    
    predicts = read_from_naruto_predicted_dataset(PREDICTIONS_DIR)
    gt_df = read_ground_truth_dataset(GROUND_TRUTH_PATH)
    if COLLECT_DIAGNOSTICS:
        results, diagnostics = run_evaluation(predicts, gt_df, collect_diagnostics=True)
    else:
        results = run_evaluation(predicts, gt_df)
        diagnostics = []

    print("Evaluation done!")
    print(results)
    write_results(results, os.fspath(OUTPUT_DIR))
    if COLLECT_DIAGNOSTICS:
        write_diagnostics(diagnostics, os.fspath(OUTPUT_DIR))
    return results


if __name__ == "__main__":
    main()
