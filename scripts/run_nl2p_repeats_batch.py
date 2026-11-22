"""Frozen-prompt, three-repeat NL2P/minimal Batch campaign (5040 requests)."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import openai_batch_experiment as batch

CAMPAIGN = "nl2p_repeats_20261027_v1"
DIRECTORY = ROOT / "results" / "batch" / CAMPAIGN
MATRIX = DIRECTORY / "matrix.json"
MODELS = {"gpt-5.4": "gpt-5.4-2026-03-05", "gpt-5.4-mini": "gpt-5.4-mini-2026-03-17"}
COUNTS = {"cooking": 116, "wikihow": 150, "win2k": 154}
ACTIVE = {"validating", "in_progress", "finalizing", "cancelling"}
TERMINAL = {"completed", "failed", "expired", "cancelled"}


def now():
    return datetime.now(ZoneInfo("Australia/Sydney")).isoformat(timespec="seconds")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    batch.write_json(temporary, value)
    temporary.replace(path)


def reference_manifest(model, condition):
    if condition == "full":
        return ROOT / "results" / "batch" / "nl2p_1" / model / "full" / "manifest.json"
    return ROOT / "results" / "batch" / "nl2p_minimal" / model / "rebuttal-20261027-minimal" / "manifest.json"


def prepare():
    if DIRECTORY.exists():
        raise FileExistsError(f"Campaign already exists: {DIRECTORY}")
    DIRECTORY.mkdir(parents=True)
    matrix = {
        "campaign": CAMPAIGN, "created_at": now(), "status": "preparing",
        "models": MODELS, "datasets": COUNTS, "rounds": [1, 2, 3],
        "expected_batches": 12, "expected_requests": 5040, "runs": [],
        "protocol": {
            "prompts": "Exact messages from the previous submitted inputs; no prompt edits.",
            "parameters": "Exact snapshots; only model and messages, matching prior actual sampling settings.",
            "repeats": "Three fresh complete runs per condition/model, no best-of selection or voting.",
            "evaluation": "Preserve all document IDs, failed/empty outputs, and unchanged strict evaluation.",
            "retries": "Only zero-request token_limit_exceeded batch rejections; never rerun completed inference.",
            "output_isolation": "Every condition and round has a new result_solver namespace.",
        },
        "reference_inputs": [], "protected_results": [], "source_files": [],
    }
    references = {}
    for model in MODELS:
        for condition in ["minimal", "full"]:
            path = reference_manifest(model, condition)
            manifest = batch.load_manifest(path)
            input_path = path.parent / "input.jsonl"
            rows = batch.read_jsonl(input_path)
            assert len(rows) == 420
            references[(model, condition)] = {r["custom_id"]: r for r in rows}
            matrix["reference_inputs"].append({"path": str(input_path), "sha256": digest(input_path)})
            result_dir = Path(manifest["result_dir"]) if manifest.get("result_dir") else None
            if result_dir and result_dir.is_dir():
                for result in sorted(result_dir.glob("*")):
                    if result.is_file() and result.suffix in {".json", ".pkl"}:
                        matrix["protected_results"].append({"path": str(result), "sha256": digest(result)})

    prior_messages = {}
    for repeat in [1, 2, 3]:
        for model, snapshot in MODELS.items():
            for condition, solver in [("minimal", "nl2p_minimal"), ("full", "nl2p_1")]:
                run_id = f"repeat-20261027-v1-r{repeat}-{condition}"
                result_solver = f"nl2p_{condition}_repeat{repeat}_20261027_v1"
                result_dir = ROOT / "results" / result_solver / model
                if result_dir.exists():
                    raise FileExistsError(result_dir)
                args = SimpleNamespace(s=solver, m=model, d=None, l=None, t=0, coref="none",
                    coref_dir=None, run_id=run_id, selection_seed=None, api_model=snapshot,
                    result_solver=result_solver)
                with contextlib.redirect_stdout(io.StringIO()):
                    batch.prepare(args)
                path = batch.run_dir(solver, model, run_id) / "manifest.json"
                manifest = batch.load_manifest(path)
                rows = batch.read_jsonl(Path(manifest["input_jsonl"]))
                assert len(rows) == manifest["request_count"] == 420
                assert Counter(r["dataset"] for r in manifest["records"]) == COUNTS
                assert len({r["custom_id"] for r in rows}) == 420
                messages = {}
                for row, record in zip(rows, manifest["records"]):
                    reference = references[(model, condition)][row["custom_id"]]
                    assert row["body"]["messages"] == reference["body"]["messages"]
                    assert set(row["body"]) == {"model", "messages"}
                    assert row["body"]["model"] == snapshot
                    assert row["custom_id"] == record["custom_id"]
                    assert len(row["body"]["messages"]) == 1
                    assert row["body"]["messages"][0]["role"] == "user"
                    messages[(record["dataset"], record["doc_id"])] = row["body"]["messages"]
                for ds, count in COUNTS.items():
                    assert sorted(doc for dataset, doc in messages if dataset == ds) == list(range(count))
                if condition in prior_messages:
                    assert prior_messages[condition] == messages
                prior_messages[condition] = messages
                manifest.update(campaign=CAMPAIGN, repeat=repeat, condition=condition,
                    reference_input_jsonl=str(reference_manifest(model, condition).parent / "input.jsonl"),
                    expected_result_dir=str(result_dir), prompt_identity_verified=True)
                save(path, manifest)
                matrix["runs"].append({"model": model, "api_model": snapshot, "condition": condition,
                    "repeat": repeat, "manifest": str(path), "input_sha256": manifest["input_sha256"],
                    "request_count": 420, "result_dir": str(result_dir), "status": "prepared"})
                save(MATRIX, matrix)
    assert len(matrix["runs"]) == 12
    for path in ["scripts/run_nl2p_repeats_batch.py", "scripts/openai_batch_experiment.py",
                 "src/llm/config.py", "src/solvers/nl2p_1.py", "src/solvers/__init__.py",
                 "experiment.py", "evaluation.py", "src/evaluation_helpers.py"]:
        source = ROOT / path
        destination = DIRECTORY / "source_snapshot" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        matrix["source_files"].append({"path": str(source), "sha256": digest(source)})
    matrix.update(status="preflight_passed", preflight={"requests_checked": 5040,
        "historical_prompt_mismatches": 0, "cross_model_or_repeat_prompt_mismatches": 0,
        "dataset_counts_per_run": COUNTS, "sampling_parameters_sent": False})
    save(MATRIX, matrix)
    print(json.dumps({"matrix": str(MATRIX), "runs": 12, "requests": 5040, "preflight": matrix["preflight"]}))


def capacity_rejected(remote):
    errors = (remote.get("errors") or {}).get("data") or []
    counts = remote.get("request_counts") or {}
    return (remote.get("status") == "failed" and bool(errors)
            and all(e.get("code") == "token_limit_exceeded" for e in errors)
            and all(counts.get(k, 0) == 0 for k in ["total", "completed", "failed"]))


def verify_frozen(matrix):
    for item in matrix["reference_inputs"] + matrix["source_files"]:
        if digest(item["path"]) != item["sha256"]:
            raise RuntimeError(f"Frozen input/source changed: {item['path']}")
    for run in matrix["runs"]:
        manifest = batch.load_manifest(Path(run["manifest"]))
        if digest(manifest["input_jsonl"]) != run["input_sha256"]:
            raise RuntimeError(f"Prepared input changed: {run['manifest']}")
        if str(ROOT / "results" / batch.result_solver_name(manifest) / run["model"]) != run["result_dir"]:
            raise RuntimeError("Result namespace changed")
        for dataset, expected in manifest["dataset_sha256"].items():
            if digest(batch.dataset_path(batch.DATASETS[dataset])) != expected:
                raise RuntimeError(f"Dataset changed: {dataset}")


def refresh_and_collect(run, client):
    path = Path(run["manifest"])
    manifest = batch.load_manifest(path)
    if not manifest.get("batch_id"):
        run["status"] = "prepared"
        return
    remote = client.batches.retrieve(manifest["batch_id"]).model_dump()
    run.update(batch_id=remote["id"], status=remote["status"], request_counts=remote.get("request_counts"),
               errors=remote.get("errors"), capacity_rejected=capacity_rejected(remote))
    manifest.update(batch_status=remote["status"], request_counts=remote.get("request_counts"),
        errors=remote.get("errors"), output_file_id=remote.get("output_file_id"),
        error_file_id=remote.get("error_file_id"), last_checked_at=now())
    save(path, manifest)
    if remote["status"] in TERMINAL and not capacity_rejected(remote):
        for label in ["output", "error"]:
            file_id = remote.get(label + "_file_id")
            target = path.parent / (label + ".jsonl")
            if file_id and not target.exists():
                content = client.files.content(file_id).read()
                temporary = target.with_suffix(".jsonl.tmp")
                temporary.write_bytes(content)
                temporary.replace(target)
        output = path.parent / "output.jsonl"
        if output.exists() and not manifest.get("collected_at"):
            rows = batch.read_jsonl(output)
            ids = [r["custom_id"] for r in rows]
            assert len(ids) == len(set(ids))
            assert set(ids) <= {r["custom_id"] for r in manifest["records"]}
            with contextlib.redirect_stdout(io.StringIO()):
                batch.collect(SimpleNamespace(manifest=str(path), output_jsonl=str(output),
                    output_file_id=None, result_solver=None))
            manifest = batch.load_manifest(path)
            assert manifest["result_dir"] == run["result_dir"]
            response_models = Counter((r.get("response") or {}).get("body", {}).get("model") for r in rows)
            run["collection"] = {"collected_at": manifest["collected_at"],
                "failure_count": manifest["failure_count"], "output_sha256": digest(output),
                "response_models": dict(response_models), "result_dir": manifest["result_dir"]}
            save(path.parent / "collection_audit.json", run["collection"])
            print(f"COLLECTED {run['model']} r{run['repeat']} {run['condition']}: failures={manifest['failure_count']}", flush=True)
        elif manifest.get("collected_at") and "collection" not in run:
            run["collection"] = batch.load_manifest(path.parent / "collection_audit.json")


def cycle(clients):
    matrix = batch.load_manifest(MATRIX)
    verify_frozen(matrix)
    # Submit each fresh manifest exactly once; submit() recovers an uncertain create.
    for run in matrix["runs"]:
        manifest = batch.load_manifest(Path(run["manifest"]))
        if not manifest.get("batch_id"):
            batch.submit(SimpleNamespace(manifest=run["manifest"]))
            manifest = batch.load_manifest(Path(run["manifest"]))
            run.update(batch_id=manifest["batch_id"], status=manifest["batch_status"])
            save(MATRIX, matrix)
    for run in matrix["runs"]:
        refresh_and_collect(run, clients[run["model"]])
        save(MATRIX, matrix)
    unexpected = [r for r in matrix["runs"] if r["status"] in {"failed", "expired", "cancelled"} and not r["capacity_rejected"]]
    matrix["last_checked_at"] = now()
    matrix["status_counts"] = dict(Counter(r["status"] for r in matrix["runs"]))
    matrix["completed_requests"] = sum((r.get("request_counts") or {}).get("completed", 0) for r in matrix["runs"])
    matrix["pending_capacity"] = [{k: r[k] for k in ["model", "repeat", "condition"]} for r in matrix["runs"] if r.get("capacity_rejected")]
    matrix["status"] = "requires_attention" if unexpected else "running"
    if all(r["status"] == "completed" and r.get("collection") for r in matrix["runs"]):
        matrix["status"] = "completed"
        matrix["finished_at"] = now()
    save(MATRIX, matrix)
    print(json.dumps({k: matrix.get(k) for k in ["last_checked_at", "status", "status_counts", "completed_requests", "pending_capacity"]}), flush=True)
    if unexpected:
        raise RuntimeError("Non-capacity terminal failure; preserved partial outputs and stopped without rerunning inference")
    if matrix["status"] == "completed":
        for item in matrix["protected_results"]:
            if digest(item["path"]) != item["sha256"]:
                raise RuntimeError(f"Historical result changed: {item['path']}")
        return True
    # Retry at most one round's pair per model, once other queued work has drained.
    pending = [r for r in matrix["runs"] if r.get("capacity_rejected")]
    if pending:
        existing = list(clients["gpt-5.4"].batches.list(limit=100))
        for model in MODELS:
            eligible = [r for r in pending if r["model"] == model]
            busy = any(b.status in ACTIVE and (b.metadata or {}).get("model") in {model, MODELS[model]} for b in existing)
            if not eligible or busy:
                continue
            repeat = min(r["repeat"] for r in eligible)
            for run in [r for r in eligible if r["repeat"] == repeat]:
                path = Path(run["manifest"])
                manifest = batch.load_manifest(path)
                history = manifest.setdefault("submission_history", [])
                if len(history) >= 3:
                    raise RuntimeError("Three capacity retries exhausted; inspect queue before another attempt")
                history.append({k: manifest.get(k) for k in ["batch_id", "batch_status", "input_file_id",
                    "submission_started_at", "submitted_at", "last_checked_at", "request_counts", "errors"]})
                for key in ["batch_id", "batch_status", "submission_started_at", "submitted_at", "request_counts", "errors", "output_file_id", "error_file_id"]:
                    manifest.pop(key, None)
                save(path, manifest)
                batch.submit(SimpleNamespace(manifest=str(path)))
                manifest = batch.load_manifest(path)
                run.update(batch_id=manifest["batch_id"], status=manifest["batch_status"], capacity_rejected=False,
                           request_counts=None, errors=None, submission_history=history)
                save(MATRIX, matrix)
    return False


def work(watch):
    import msvcrt
    lock = (DIRECTORY / "worker.lock").open("a+b")
    lock.write(b"0")
    lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    worker_path = DIRECTORY / "worker.json"
    save(worker_path, {"pid": os.getpid(), "started_at": now(), "status": "running", "watch": watch})
    clients = {m: batch.get_openai_client(m).with_options(timeout=45, max_retries=0) for m in MODELS}
    assert all(str(c.base_url).rstrip("/") == "https://api.openai.com/v1" for c in clients.values())
    try:
        while True:
            complete = cycle(clients)
            save(worker_path, {"pid": os.getpid(), "last_checked_at": now(),
                "status": "completed" if complete else "running" if watch else "one_pass_finished"})
            if complete or not watch:
                return
            time.sleep(60)
    except Exception as exc:
        save(worker_path, {"pid": os.getpid(), "status": "requires_attention", "at": now(),
            "error_type": type(exc).__name__, "error": str(exc)[:1200]})
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "once", "watch"])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        work(args.command == "watch")
