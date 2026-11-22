r"""Prepare, submit, and collect OpenAI Batch experiments.

This keeps the normal synchronous experiment path untouched. It currently
supports one-call solvers whose output can be parsed independently per sample.

prepare
python scripts\openai_batch_experiment.py prepare -s nl2p_1 -m gpt-5.4-mini --run-id full

submit
python scripts\openai_batch_experiment.py submit results\batch\nl2p_1\gpt-5.4-mini\full\manifest.json

check status
python scripts\openai_batch_experiment.py status results\batch\nl2p_1\gpt-5.4-mini\full\manifest.json

collect
python scripts\openai_batch_experiment.py collect results\batch\nl2p_1\gpt-5.4-mini\full\manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
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
from src.solvers import GPT3ToPlan, NL2P_1, NL2P_1_Ablation, NL2P_Minimal  # noqa: E402
from src.solvers.nl2p_1 import NL2P_WithoutDefinitions, NL2P_WithoutRules


BATCH_ROOT = ROOT / "results" / "batch"
SUPPORTED_SOLVERS = {
    "gpt3_to_plan": GPT3ToPlan,
    "gpt3_to_plan_random": GPT3ToPlan,
    "nl2p_1": NL2P_1,
    "nl2p_minimal": NL2P_Minimal,
    NL2P_WithoutDefinitions.prompt_name: NL2P_WithoutDefinitions,
    NL2P_WithoutRules.prompt_name: NL2P_WithoutRules,
    "nl2p_1_ablation": NL2P_1_Ablation,
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def run_dir(solver_name: str, model_name: str, run_id: str) -> Path:
    return BATCH_ROOT / solver_name / safe_name(model_name) / run_id


def result_solver_name(manifest: dict[str, Any], override: str | None = None) -> str:
    if override:
        return safe_name(override)
    if manifest.get("result_solver"):
        return safe_name(manifest["result_solver"])
    solver_name = manifest["solver"]
    if solver_name == "gpt3_to_plan_random":
        solver_name += f"_seed{manifest['selection_seed']}"
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


def build_chat_body(model_key: str, prompt: str, temperature: float, api_model: str | None = None) -> dict[str, Any]:
    model_config = MODELS[model_key]
    body: dict[str, Any] = {
        "model": api_model or model_config["model_name"],
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


def build_solver(solver_name: str, model_name: str, datasets: dict[str, list[dict[str, Any]]] | None = None, selection_seed: int | None = None):
    if solver_name in {"gpt3_to_plan", "gpt3_to_plan_random"}:
        return GPT3ToPlan(
            datasets=datasets or {}, model_name=model_name,
            example_selection="random" if solver_name == "gpt3_to_plan_random" else "op_ex",
            selection_seed=selection_seed,
        )
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
    if (out_dir / "manifest.json").exists() or (out_dir / "input.jsonl").exists():
        raise FileExistsError(f"Run already exists; use its manifest or a new run id: {out_dir}")
    selection_seed = getattr(args, "selection_seed", None)
    if args.s == "gpt3_to_plan_random" and selection_seed is None:
        raise ValueError("gpt3_to_plan_random requires --selection-seed")
    if args.s != "gpt3_to_plan_random" and selection_seed is not None:
        raise ValueError("--selection-seed is only valid for gpt3_to_plan_random")
    api_model = getattr(args, "api_model", None)
    coref_by_domain = {
        ds_name: load_coref_texts(ds_name, coref=args.coref, coref_dir=args.coref_dir)
        for ds_name in target_datasets
    }
    datasets = {
        ds_name: read_from_labeled_dataset(DATASETS[ds_name], limit=args.l)
        for ds_name in target_datasets
    }
    split_path = getattr(args, "example_train_split", None)
    if split_path:
        if args.s not in {"gpt3_to_plan", "gpt3_to_plan_random"} or args.l is not None:
            raise ValueError("Train-pool selection requires a two-shot solver and full domain data")
        split = load_manifest(Path(split_path))["domains"]
        train_ids = {}
        for domain, data in datasets.items():
            spec = split[domain]
            if hashlib.sha256(Path(dataset_path(DATASETS[domain])).read_bytes()).hexdigest() != spec["sha256"]:
                raise ValueError(f"{domain}: source dataset differs from demonstration split")
            groups = spec["doc_ids"]
            all_ids = groups["train"] + groups["val"] + groups["test"]
            if sorted(all_ids) != list(range(len(data))) or len(groups["train"]) < 3:
                raise ValueError(f"{domain}: invalid split or fewer than three training examples")
            train_ids[domain] = sorted(groups["train"])
        train_data = {d: [datasets[d][i] for i in ids] for d, ids in train_ids.items()}
        solver = build_solver(args.s, args.m, datasets=train_data, selection_seed=selection_seed)
        # Map training-local positions back to the original IDs used by query exclusion.
        solver.ranked_example_indices = {
            d: [train_ids[d][i] for i in indices] for d, indices in solver.ranked_example_indices.items()
        }
        solver.datasets = datasets
    else:
        solver = build_solver(args.s, args.m, datasets=datasets, selection_seed=selection_seed)
    if isinstance(solver, GPT3ToPlan) and any(len(ds) < 3 for ds in datasets.values()):
        raise ValueError("Two demonstrations with query exclusion require at least three documents per dataset")

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
                    "body": build_chat_body(args.m, prompt, temperature=args.t, api_model=api_model),
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
            records[-1]["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if isinstance(solver, GPT3ToPlan):
                records[-1]["example_doc_ids"] = solver.selected_example_indices(ds_name, doc_id)

    manifest = {
        "solver": args.s,
        "prompt_name": solver.prompt_name,
        "model": args.m,
        "model_name": api_model or MODELS[args.m]["model_name"],
        "datasets": target_datasets,
        "limit": args.l,
        "temperature": args.t,
        "request_parameters": {k: v for k, v in requests[0]["body"].items() if k != "messages"} if requests else {},
        "selection_seed": selection_seed,
        "example_selection": solver.example_selection if isinstance(solver, GPT3ToPlan) else None,
        "example_permutations": solver.ranked_example_indices if isinstance(solver, GPT3ToPlan) else None,
        "dataset_sha256": {
            name: hashlib.sha256(Path(dataset_path(DATASETS[name])).read_bytes()).hexdigest()
            for name in target_datasets if Path(dataset_path(DATASETS[name])).is_file()
        },
        "coref": args.coref,
        "coref_dir": args.coref_dir,
        "run_id": run_id,
        "endpoint": "/v1/chat/completions",
        "input_jsonl": str(out_dir / "input.jsonl"),
        "records": records,
        "request_count": len(requests),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if split_path:
        manifest["example_train_split_sha256"] = hashlib.sha256(Path(split_path).read_bytes()).hexdigest()
        manifest["example_train_split"] = str(Path(split_path).resolve())
    manifest["result_solver"] = result_solver_name(manifest, getattr(args, "result_solver", None))
    write_jsonl(out_dir / "input.jsonl", requests)
    manifest["input_sha256"] = hashlib.sha256((out_dir / "input.jsonl").read_bytes()).hexdigest()
    write_json(out_dir / "manifest.json", manifest)
    print(f"Prepared {len(requests)} requests")
    print(f"Input: {out_dir / 'input.jsonl'}")
    print(f"Manifest: {out_dir / 'manifest.json'}")


def submit(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    if manifest.get("batch_id"):
        print(f"Already submitted: {manifest['batch_id']} ({manifest.get('batch_status')})")
        return
    client = get_openai_client(manifest["model"]).with_options(max_retries=0, timeout=60)

    input_path = Path(manifest["input_jsonl"])
    if not input_path.is_absolute():
        input_path = (manifest_path.parent / input_path).resolve()
    if manifest.get("input_sha256") and hashlib.sha256(input_path.read_bytes()).hexdigest() != manifest["input_sha256"]:
        raise ValueError("Input JSONL changed after preparation; refusing to submit")
    if not manifest.get("input_file_id"):
        with input_path.open("rb") as f:
            batch_file = client.files.create(file=f, purpose="batch")
        manifest["input_file_id"] = batch_file.id
        write_json(manifest_path, manifest)
    if manifest.get("submission_started_at"):
        # A previous create may have succeeded before the response was lost.
        # Recover the matching remote job before allowing another create.
        for previous in client.batches.list(limit=100):
            if previous.input_file_id == manifest["input_file_id"]:
                manifest["batch_id"] = previous.id
                manifest["batch_status"] = previous.status
                write_json(manifest_path, manifest)
                print(f"Recovered submitted batch {previous.id}")
                return
        raise RuntimeError("Unresolved prior submission attempt; inspect remote batches before resubmitting")
    manifest["submission_started_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_json(manifest_path, manifest)
    batch = client.batches.create(
        input_file_id=manifest["input_file_id"],
        endpoint=manifest["endpoint"],
        completion_window="24h",
        metadata={
            "solver": manifest["solver"],
            "model": manifest["model"],
            "run_id": manifest["run_id"],
        },
    )

    manifest["batch_id"] = batch.id
    manifest["batch_status"] = batch.status
    manifest["submitted_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(manifest_path, manifest)
    print(f"Submitted batch {batch.id} with input file {manifest['input_file_id']}")
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
    solver = build_solver(manifest["solver"], manifest["model"], selection_seed=manifest.get("selection_seed"))

    grouped_results: dict[str, list[dict[str, Any]]] = {ds_name: [] for ds_name in manifest["datasets"]}
    grouped_samples: dict[str, list[dict[str, Any]]] = {}
    failures = []

    for ds_name in manifest["datasets"]:
        grouped_samples[ds_name] = read_from_labeled_dataset(DATASETS[ds_name], limit=manifest["limit"])

    for custom_id, record in record_map.items():
        row = output_rows.get(custom_id)
        if row is None:
            failures.append({"custom_id": custom_id, "error": "missing_output"})
            prediction = None
        elif (content := parse_response_content(row)) is None:
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
    prepare_parser.add_argument("--example-train-split", type=Path, help="Restrict demonstration ranking or sampling to the named training partition")
    prepare_parser.add_argument("--selection-seed", type=int, help="Required fixed demonstration seed for gpt3_to_plan_random")
    prepare_parser.add_argument("--api-model", help="Exact API model snapshot; retains the model key for configuration and output paths")
    prepare_parser.add_argument("--result-solver", help="Separate result namespace to use when collecting this run")
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
