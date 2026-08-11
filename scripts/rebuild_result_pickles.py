"""Rebuild evaluator-ready PKL files from compact prediction JSON files.

The supplementary archive stores one copy of each labeled dataset and compact
JSON predictions for every method/model condition.  This script reconstructs
the derived PKL files expected by ``evaluation.py`` without model inference.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path


DATASET_FILES = {
    "cooking": "cooking_labeled_text_data.pkl",
    "wikihow": "wikihow_labeled_text_data.pkl",
    "win2k": "win2k_labeled_text_data.pkl",
}


def dataset_name_from_result(path: Path) -> str | None:
    for dataset_name in DATASET_FILES:
        if path.name.startswith(f"{dataset_name}_"):
            return dataset_name
    return None


def load_coref_texts(coref_dir: Path, dataset_name: str) -> dict[int, str]:
    path = coref_dir / f"{dataset_name}_llm_coref.jsonl"
    records: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("domain") != dataset_name:
                continue
            doc_id = int(record["doc_id"])
            resolved_text = record.get("resolved_text")
            if not isinstance(resolved_text, str) or not resolved_text:
                raise ValueError(f"Missing resolved_text in {path}:{line_number}")
            records[doc_id] = resolved_text
    return records


def rebuild_result_file(
    result_json: Path,
    data_dir: Path,
    coref_dir: Path,
    overwrite: bool = False,
) -> Path:
    dataset_name = dataset_name_from_result(result_json)
    if dataset_name is None or result_json.name.endswith("_summary.json"):
        raise ValueError(f"Not a supported result JSON: {result_json}")

    output_path = result_json.with_suffix(".pkl")
    if output_path.exists() and not overwrite:
        return output_path

    dataset_path = data_dir / DATASET_FILES[dataset_name]
    with dataset_path.open("rb") as handle:
        samples = pickle.load(handle)
    samples = copy.deepcopy(samples)

    with result_json.open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    if not isinstance(results, list):
        raise TypeError(f"Expected a result list in {result_json}")

    by_doc_id: dict[int, dict] = {}
    for result in results:
        doc_id = result.get("doc_id")
        if not isinstance(doc_id, int):
            raise ValueError(f"Invalid doc_id in {result_json}: {doc_id!r}")
        if doc_id in by_doc_id:
            raise ValueError(f"Duplicate doc_id {doc_id} in {result_json}")
        by_doc_id[doc_id] = result

    expected_ids = set(range(len(samples)))
    actual_ids = set(by_doc_id)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"Document IDs do not cover {dataset_name}: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    is_coref = "nl2p_1_coref" in result_json.name
    coref_texts = load_coref_texts(coref_dir, dataset_name) if is_coref else {}
    source_file = f"data/easdrl/{DATASET_FILES[dataset_name]}"

    for doc_id, sample in enumerate(samples):
        result = by_doc_id[doc_id]
        sentences = result.get("sentences")
        if not isinstance(sentences, list) or not all(
            isinstance(sentence, str) for sentence in sentences
        ):
            raise ValueError(f"Invalid sentences for doc_id {doc_id} in {result_json}")
        if is_coref:
            if doc_id not in coref_texts:
                raise ValueError(f"Missing coreference text for {dataset_name}:{doc_id}")
            original_text = coref_texts[doc_id]
        else:
            original_text = ". ".join(sentences) + "."

        sample["pred"] = result.get("prediction")
        sample["doc_id"] = doc_id
        sample["docId"] = f"{dataset_name}:{doc_id}"
        sample["domain"] = dataset_name
        sample["source_file"] = source_file
        sample["original_text"] = original_text

    with output_path.open("wb") as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path


def rebuild_tree(
    results_dir: Path,
    data_dir: Path,
    coref_dir: Path,
    overwrite: bool = False,
) -> list[Path]:
    candidates = sorted(
        path
        for path in results_dir.rglob("*.json")
        if dataset_name_from_result(path) is not None
        and not path.name.endswith("_summary.json")
    )
    if not candidates:
        raise FileNotFoundError(f"No compact result JSON files found under {results_dir}")
    return [
        rebuild_result_file(path, data_dir, coref_dir, overwrite=overwrite)
        for path in candidates
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/easdrl"))
    parser.add_argument("--coref-dir", type=Path, default=Path("data/coref_llm"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs = rebuild_tree(
        args.results_dir,
        args.data_dir,
        args.coref_dir,
        overwrite=args.overwrite,
    )
    print(f"Rebuilt or verified {len(outputs)} evaluator PKL files.")


if __name__ == "__main__":
    main()
