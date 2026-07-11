r"""Prepare, submit, and collect OpenAI Batch experiments.

This keeps the normal synchronous experiment path untouched. It currently
supports one-call solvers whose output can be parsed independently per sample.

prepare
C:\Users\Apexmod\miniforge3\envs\llm\python.exe scripts\openai_batch_experiment.py prepare -s nl2p_1 -m gpt-5.4-mini --run-id full

submit
C:\Users\Apexmod\miniforge3\envs\llm\python.exe scripts\openai_batch_experiment.py submit results\batch\nl2p_1\gpt-5.4-mini\full\manifest.json

check status
C:\Users\Apexmod\miniforge3\envs\llm\python.exe scripts\openai_batch_experiment.py status results\batch\nl2p_1\gpt-5.4-mini\full\manifest.json

collect
C:\Users\Apexmod\miniforge3\envs\llm\python.exe scripts\openai_batch_experiment.py collect results\batch\nl2p_1\gpt-5.4-mini\full\manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment import (  # noqa: E402
    DATASETS,
    RESULTS_DIR,
    build_result_record,
    dataset_path,
    load_coref_texts,
    read_from_labeled_dataset,
    sample_to_input_text,
)
from src.llm import MODELS, generate_prompt  # noqa: E402
from src.solvers import GPT3ToPlan, NL2P_1, NL2P_1_Ablation  # noqa: E402


BATCH_ROOT = ROOT / "results" / "batch"
SUPPORTED_SOLVERS = {
    "gpt3_to_plan": GPT3ToPlan,
    "nl2p_1": NL2P_1,
    "nl2p_1_ablation": NL2P_1_Ablation,
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def run_dir(solver_name: str, model_name: str, run_id: str) -> Path:
    return BATCH_ROOT / solver_name / safe_name(model_name) / run_id


def result_solver_name(manifest: dict[str, Any], override: str | None = None) -> str:
    if override:
        return safe_name(override)
    solver_name = manifest["solver"]
    coref_mode = manifest.get("coref") or "none"
    if coref_mode == "none":
        return solver_name
    return f"{solver_name}_coref"


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_chat_body(model_key: str, prompt: str, temperature: float) -> dict[str, Any]:
    model_config = MODELS[model_key]
    body: dict[str, Any] = {
        "model": model_config["model_name"],
        "messages": [{"role": "user", "content": prompt}],
    }
    if model_config.get("supports_custom_sampling", True):
        body.update(
            {
                "temperature": temperature,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            }
        )
    if model_config.get("max_tokens") is not None:
        body["max_tokens"] = model_config["max_tokens"]
    return body


def build_solver(solver_name: str, model_name: str, datasets: dict[str, list[dict[str, Any]]] | None = None):
    if solver_name == "gpt3_to_plan":
        return GPT3ToPlan(datasets=datasets or {}, model_name=model_name)
    return SUPPORTED_SOLVERS[solver_name](model_name=model_name)


def build_prompt(solver, paragraph: str, ds_name: str, doc_id: int | None = None) -> str:
    if hasattr(solver, "build_prompt"):
        return solver.build_prompt(paragraph, ds_name=ds_name, doc_id=doc_id)
    return generate_prompt(solver.prompt_name, {"nl": paragraph})


def get_openai_client(model_key: str):
    model_config = MODELS[model_key]
    if model_config.get("provider") != "openai":
        raise ValueError(f"Batch is only supported for OpenAI provider models, got {model_key}")
    if not model_config.get("api_key"):
        raise ValueError("OPENAI_API_KEY is required to submit, check, or collect a batch")

    from openai import OpenAI

    kwargs = {
        "api_key": model_config["api_key"],
        "base_url": model_config.get("base_url") or "https://api.openai.com/v1",
    }
    if model_config.get("timeout") is not None:
        kwargs["timeout"] = model_config["timeout"]
    return OpenAI(**kwargs)


def prepare(args: argparse.Namespace) -> None:
    if args.s not in SUPPORTED_SOLVERS:
        raise ValueError(f"Batch solver must be one of {sorted(SUPPORTED_SOLVERS)}")
    if args.m not in MODELS:
        raise ValueError(f"Unknown model {args.m}. Available: {sorted(MODELS)}")
    if MODELS[args.m].get("provider") != "openai":
        raise ValueError("OpenAI Batch preparation is only for provider=openai models")

    target_datasets = [args.d] if args.d else list(DATASETS)
    for ds_name in target_datasets:
        if ds_name not in DATASETS:
            raise ValueError(f"Unknown dataset {ds_name}. Available: {sorted(DATASETS)}")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = run_dir(args.s, args.m, run_id)
    coref_by_domain = {
        ds_name: load_coref_texts(ds_name, coref=args.coref, coref_dir=args.coref_dir)
        for ds_name in target_datasets
    }
    datasets = {
        ds_name: read_from_labeled_dataset(DATASETS[ds_name], limit=args.l)
        for ds_name in target_datasets
    }
    solver = build_solver(args.s, args.m, datasets=datasets)

    requests = []
    records = []
    for ds_name in target_datasets:
        source_file = dataset_path(DATASETS[ds_name])
        dataset = datasets[ds_name]
        coref_texts = coref_by_domain[ds_name] or None
        for doc_id, sample in enumerate(dataset):
            paragraph = sample_to_input_text(sample, ds_name=ds_name, doc_id=doc_id, coref_texts=coref_texts)
            prompt = build_prompt(solver, paragraph, ds_name, doc_id=doc_id)
            custom_id = f"{args.s}|{safe_name(args.m)}|{ds_name}|{doc_id}"
            requests.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": build_chat_body(args.m, prompt, temperature=args.t),
                }
            )
            records.append(
                {
                    "custom_id": custom_id,
                    "dataset": ds_name,
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "original_text": paragraph,
                }
            )

    manifest = {
        "solver": args.s,
        "prompt_name": solver.prompt_name,
        "model": args.m,
        "model_name": MODELS[args.m]["model_name"],
        "datasets": target_datasets,
        "limit": args.l,
        "temperature": args.t,
        "coref": args.coref,
        "coref_dir": args.coref_dir,
        "run_id": run_id,
        "endpoint": "/v1/chat/completions",
        "input_jsonl": str(out_dir / "input.jsonl"),
        "records": records,
        "request_count": len(requests),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_jsonl(out_dir / "input.jsonl", requests)
    write_json(out_dir / "manifest.json", manifest)
    print(f"Prepared {len(requests)} requests")
    print(f"Input: {out_dir / 'input.jsonl'}")
    print(f"Manifest: {out_dir / 'manifest.json'}")


def submit(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    client = get_openai_client(manifest["model"])

    input_path = Path(manifest["input_jsonl"])
    if not input_path.is_absolute():
        input_path = (manifest_path.parent / input_path).resolve()
    with input_path.open("rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint=manifest["endpoint"],
        completion_window="24h",
        metadata={
            "solver": manifest["solver"],
            "model": manifest["model"],
            "run_id": manifest["run_id"],
        },
    )

    manifest["input_file_id"] = batch_file.id
    manifest["batch_id"] = batch.id
    manifest["batch_status"] = batch.status
    manifest["submitted_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(manifest_path, manifest)
    print(f"Submitted batch {batch.id} with input file {batch_file.id}")
    print(f"Status: {batch.status}")


def status(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    batch_id = args.batch_id or manifest.get("batch_id")
    if not batch_id:
        raise ValueError("No batch_id found. Pass --batch-id or submit first.")

    client = get_openai_client(manifest["model"])
    batch = client.batches.retrieve(batch_id)
    manifest["batch_status"] = batch.status
    manifest["output_file_id"] = batch.output_file_id
    manifest["error_file_id"] = batch.error_file_id
    manifest["request_counts"] = batch.request_counts.model_dump() if batch.request_counts else None
    manifest["last_checked_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(manifest_path, manifest)
    print(json.dumps(batch.model_dump(), indent=2, ensure_ascii=False))


def parse_response_content(row: dict[str, Any]) -> str | None:
    response = row.get("response")
    if not response or response.get("status_code") != 200:
        return None
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return None
    return choices[0].get("message", {}).get("content")


def collect(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    client = get_openai_client(manifest["model"]) if not args.output_jsonl else None

    output_path = Path(args.output_jsonl).resolve() if args.output_jsonl else manifest_path.parent / "output.jsonl"
    if not args.output_jsonl:
        output_file_id = args.output_file_id or manifest.get("output_file_id")
        if not output_file_id:
            batch_id = manifest.get("batch_id")
            if not batch_id:
                raise ValueError("No output_file_id or batch_id found. Check status first.")
            batch = client.batches.retrieve(batch_id)
            output_file_id = batch.output_file_id
            if not output_file_id:
                raise ValueError(f"Batch {batch_id} has no output file yet; status is {batch.status}")
        content = client.files.content(output_file_id).read()
        output_path.write_bytes(content)
        manifest["output_file_id"] = output_file_id
        manifest["output_jsonl"] = str(output_path)

    output_rows = {row["custom_id"]: row for row in read_jsonl(output_path)}
    record_map = {record["custom_id"]: record for record in manifest["records"]}
    solver = build_solver(manifest["solver"], manifest["model"])

    grouped_results: dict[str, list[dict[str, Any]]] = {ds_name: [] for ds_name in manifest["datasets"]}
    grouped_samples: dict[str, list[dict[str, Any]]] = {}
    failures = []

    for ds_name in manifest["datasets"]:
        grouped_samples[ds_name] = read_from_labeled_dataset(DATASETS[ds_name], limit=manifest["limit"])

    for custom_id, record in record_map.items():
        row = output_rows.get(custom_id)
        if row is None:
            failures.append({"custom_id": custom_id, "error": "missing_output"})
            continue
        content = parse_response_content(row)
        if content is None:
            failures.append({"custom_id": custom_id, "error": row.get("error") or row.get("response")})
            prediction = None
        else:
            prediction = solver.parse_json(content)
            if prediction is None:
                failures.append({"custom_id": custom_id, "error": "json_parse_failed", "content": content})

        ds_name = record["dataset"]
        doc_id = record["doc_id"]
        sample = grouped_samples[ds_name][doc_id]
        sample["pred"] = prediction
        sample["doc_id"] = doc_id
        sample["docId"] = f"{ds_name}:{doc_id}"
        sample["domain"] = ds_name
        sample["source_file"] = record["source_file"]
        sample["original_text"] = record["original_text"]
        grouped_results[ds_name].append(
            build_result_record(ds_name, doc_id, record["source_file"], sample, prediction)
        )

    output_solver = result_solver_name(manifest, getattr(args, "result_solver", None))
    result_root = ROOT / RESULTS_DIR / output_solver / safe_name(manifest["model"])
    result_root.mkdir(parents=True, exist_ok=True)
    for ds_name, results in grouped_results.items():
        results.sort(key=lambda item: item["doc_id"])
        result_path = result_root / f"{ds_name}_{output_solver}_{safe_name(manifest['model'])}.json"
        pkl_path = result_root / f"{ds_name}_{output_solver}_{safe_name(manifest['model'])}.pkl"
        summary_path = result_root / f"{ds_name}_{output_solver}_{safe_name(manifest['model'])}_summary.json"
        write_json(result_path, results)
        with pkl_path.open("wb") as f:
            pickle.dump(grouped_samples[ds_name], f)
        write_json(
            summary_path,
            {
                "dataset": ds_name,
                "solver": output_solver,
                "source_solver": manifest["solver"],
                "coref": manifest.get("coref", "none"),
                "model": safe_name(manifest["model"]),
                "num_docs": len(results),
                "doc_ids": [item["doc_id"] for item in results],
                "source_file": results[0].get("source_file") if results else None,
                "batch_id": manifest.get("batch_id"),
            },
        )
        print(f"Wrote {len(results)} results to {result_path}")

    manifest["collected_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["output_jsonl"] = str(output_path)
    manifest["result_solver"] = output_solver
    manifest["result_dir"] = str(result_root)
    manifest["failure_count"] = len(failures)
    write_json(manifest_path, manifest)
    if failures:
        failures_path = manifest_path.parent / "failures.json"
        write_json(failures_path, failures)
        print(f"Collected with {len(failures)} failures: {failures_path}")
    else:
        print("Collected with 0 failures")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Create batch input JSONL")
    prepare_parser.add_argument("-s", required=True, choices=sorted(SUPPORTED_SOLVERS))
    prepare_parser.add_argument("-m", required=True, help="OpenAI model key from src.llm.config.MODELS")
    prepare_parser.add_argument("-d", choices=sorted(DATASETS), help="Dataset; omit to prepare all datasets")
    prepare_parser.add_argument("-l", type=int, help="Limit documents per dataset")
    prepare_parser.add_argument("-t", type=float, default=0, help="Temperature for models that support it")
    prepare_parser.add_argument("--coref", choices=["none", "llm", "nlp"], default="none", help="Use precomputed coreference-resolved input text")
    prepare_parser.add_argument("--coref-dir", help="Directory containing *_llm_coref.jsonl or *_coref.json files")
    prepare_parser.add_argument("--run-id", help="Stable run id; defaults to timestamp")
    prepare_parser.set_defaults(func=prepare)

    submit_parser = subparsers.add_parser("submit", help="Upload input JSONL and create OpenAI batch")
    submit_parser.add_argument("manifest", help="Path to manifest.json from prepare")
    submit_parser.set_defaults(func=submit)

    status_parser = subparsers.add_parser("status", help="Refresh and print batch status")
    status_parser.add_argument("manifest", help="Path to manifest.json")
    status_parser.add_argument("--batch-id", help="Override batch id")
    status_parser.set_defaults(func=status)

    collect_parser = subparsers.add_parser("collect", help="Download and convert batch output")
    collect_parser.add_argument("manifest", help="Path to manifest.json")
    collect_parser.add_argument("--output-file-id", help="Override output file id")
    collect_parser.add_argument("--output-jsonl", help="Use an already downloaded output JSONL")
    collect_parser.add_argument(
        "--result-solver",
        help="Write to a separate solver result directory/name, e.g. gpt3_to_plan_reparsed",
    )
    collect_parser.set_defaults(func=collect)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
