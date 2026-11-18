"""Frozen, resumable Ollama repetitions; never uses hosted-model APIs.

prepare builds an isolated portable campaign. run executes one 420-document cell.
evaluate uses the existing strict evaluator and retains failed/empty predictions.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODELS = {"gemma3-27b": "gemma3:27b", "gemma3-12b": "gemma3:12b", "llama3-70b": "llama3.3:70b"}
CONDITIONS = {"minimal": ("nl2p_minimal", 0), "full": ("nl2p_1", 0), "rand2": ("gpt3_to_plan_random", 0.5)}
COUNTS = {"cooking": 116, "wikihow": 150, "win2k": 154}
CAMPAIGN = "rebuttal_open_20261028_v1"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


@contextlib.contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def load_data():
    from experiment import DATASETS, read_from_labeled_dataset
    data = {ds: read_from_labeled_dataset(name) for ds, name in DATASETS.items()}
    if {ds: len(rows) for ds, rows in data.items()} != COUNTS:
        raise ValueError("Unexpected dataset coverage")
    return data


def make_requests(data, condition, repeat):
    from experiment import sample_to_paragraph
    from scripts.openai_batch_experiment import build_solver, build_prompt
    solver_name, temperature = CONDITIONS[condition]
    solver = build_solver(solver_name, "unused", datasets=data,
                          selection_seed=repeat - 1 if condition == "rand2" else None)
    rows = []
    for ds, samples in data.items():
        for doc_id, sample in enumerate(samples):
            prompt = build_prompt(solver, sample_to_paragraph(sample), ds, doc_id)
            row = {"id": f"{ds}:{doc_id}", "dataset": ds, "doc_id": doc_id,
                   "messages": [{"role": "user", "content": prompt}],
                   "options": {"temperature": temperature},
                   "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
            if condition == "rand2":
                row["selection_seed"] = repeat - 1
                row["example_doc_ids"] = solver.selected_example_indices(ds, doc_id)
                assert doc_id not in row["example_doc_ids"]
            rows.append(row)
    return rows


def prepare(directory, priority):
    os.chdir(ROOT)
    target = Path(directory).resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to replace campaign: {target}")
    data = load_data()
    files = [ROOT / "experiment.py", ROOT / "evaluation.py", ROOT / "requirements.txt"]
    files += sorted((ROOT / "src").rglob("*.py"))
    files += [ROOT / "scripts" / name for name in ["openai_batch_experiment.py", "evaluate_rebuttal_batch.py", "run_hpc_rebuttal.py"]]
    files += [ROOT / "hpc" / name for name in ["rebuttal_open.slurm", "rebuttal_evaluate.slurm", "rebuttal_setup.slurm"]]
    files += [ROOT / "data/easdrl" / f"{ds}_labeled_text_data.pkl" for ds in COUNTS]
    files += [ROOT / "dist/code_data_supplement/expected_metrics.csv"]
    for model in MODELS:
        for solver in ["gpt3_to_plan", "nl2p_1"]:
            files += [ROOT / "results" / solver / model / f"{ds}_{solver}_{model}.json" for ds in COUNTS]
    for source in files:
        dest = target / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    cells = []
    for repeat in range(1, 11):
        for condition in CONDITIONS:
            rel = f"inputs/r{repeat:02d}_{condition}.json"
            write(target / rel, make_requests(data, condition, repeat))
            cells.append({"index": len(cells), "repeat": repeat, "condition": condition,
                          "input": rel, "request_count": 420})
    frozen = [p for p in target.rglob("*") if p.is_file()]
    manifest = {"campaign": CAMPAIGN, "created_at": now(), "models": MODELS,
                "priority_model": priority, "datasets": COUNTS, "repeats": list(range(1, 11)),
                "selection_seeds": list(range(10)), "conditions": CONDITIONS, "cells": cells,
                "expected_requests": 37800, "expected_requests_per_model": 12600,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "files": {p.relative_to(target).as_posix(): sha(p) for p in frozen},
                "notes": ["Working-tree source snapshot includes existing rebuttal implementations.",
                          "One user message; no coreference preprocessing; original parsers and strict scoring.",
                          "Original Ollama temperatures: rand2=0.5, minimal/full=0. No decoding seed supplied.",
                          "Ten rand2 selection seeds shared across models. Query excluded, no global held-out demonstration split.",
                          "No automatic model pulls or inference retries; unknown in-flight requests block resume.",
                          "Temperature-zero repetition may be identical; documents are reused, not independent new data.",
                          "Historical OP/EX scores are contextual; historical model digests/runtime may be unavailable."]}
    write(target / "campaign.json", manifest)
    print(json.dumps({"directory": str(target), "cells_per_model": len(cells), "requests": 37800}))


def verify(directory):
    directory = Path(directory).resolve()
    manifest = read(directory / "campaign.json")
    for rel, digest in manifest["files"].items():
        if sha(directory / rel) != digest:
            raise ValueError(f"Frozen file changed: {rel}")
    return manifest


def api(host, route, body=None, timeout=120):
    payload = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(host.rstrip("/") + route, data=payload,
                                    headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def pin_runtime(directory, model, host):
    tag = MODELS[model]
    tags = api(host, "/api/tags")["models"]
    entry = next((m for m in tags if m.get("name") == tag or m.get("model") == tag), None)
    if not entry or not entry.get("digest"):
        raise ValueError(f"Cached model unavailable: {tag}; model pull is not automatic")
    version = api(host, "/api/version")
    details = api(host, "/api/show", {"model": tag})
    identity = {"model": tag, "digest": entry["digest"], "version": version,
                "show_sha256": hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest(),
                "container_sha256": os.environ.get("REBUTTAL_SIF_SHA256"),
                "context_environment": os.environ.get("OLLAMA_CONTEXT_LENGTH")}
    path = directory / "runtime" / f"{model}.json"
    with lock(path.with_suffix(".lock")):
        if path.exists():
            if read(path)["identity"] != identity:
                raise ValueError("Model/runtime changed since campaign began")
        else:
            write(path, {"identity": identity, "tags_entry": entry, "show": details, "recorded_at": now()})
    return identity


def load_outputs(path, requests):
    allowed = {r["id"]: r for r in requests}
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = row["id"]
            if key not in allowed or key in rows or row["prompt_sha256"] != allowed[key]["prompt_sha256"]:
                raise ValueError("Duplicate, unknown, or changed output record")
            rows[key] = row
    return rows


def parse_prediction(condition, response):
    from scripts.openai_batch_experiment import build_solver
    content = response.get("message", {}).get("content")
    if not response.get("done") or not isinstance(content, str):
        return None, "invalid_response"
    from src.llm.ollama import split_think_content
    _, content = split_think_content(content)
    parser = build_solver(CONDITIONS[condition][0], "unused", selection_seed=0 if condition == "rand2" else None)
    try:
        prediction = parser.parse_json(content)
        if not isinstance(prediction, list) or any(
            not isinstance(p, dict) or not isinstance(p.get("verb"), str)
            or not isinstance(p.get("arguments"), list)
            or any(not isinstance(arg, str) for arg in p["arguments"]) for p in prediction
        ):
            return None, "parse_failure"
        return prediction, "parsed"
    except (TypeError, ValueError, KeyError):
        return None, "parse_failure"


def run_cell(directory, model, index, host, request_timeout=900, transport=api):
    directory = Path(directory).resolve()
    os.chdir(directory)
    manifest = verify(directory)
    cell = manifest["cells"][index]
    requests = read(directory / cell["input"])
    output = directory / "outputs" / model / f"r{cell['repeat']:02d}_{cell['condition']}"
    output.mkdir(parents=True, exist_ok=True)
    with lock(output / "worker.lock"):
        rows = load_outputs(output / "raw.jsonl", requests)
        pending = output / "inflight.json"
        if pending.exists():
            if read(pending)["id"] not in rows:
                raise RuntimeError("Uncertain previous request; preserve evidence and inspect before any retry")
            pending.unlink()
        if len(rows) == len(requests):
            write(output / "status.json", {"status": "complete", "responses": len(rows), "expected": len(requests), "updated_at": now(),
                  "parse_failures": sum(r["parse_status"] != "parsed" for r in rows.values()),
                  "truncated_responses": sum(r["truncated"] for r in rows.values())})
            return
        identity = pin_runtime(directory, model, host)
        runtime = {"host": socket.gethostname(), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                   "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"), "python": sys.version,
                   "identity": identity, "started_at": now()}
        write(output / f"execution_{os.getenv('SLURM_JOB_ID', str(os.getpid()))}.json", runtime)
        for request in requests:
            if request["id"] in rows:
                continue
            body = {"model": MODELS[model], "messages": request["messages"],
                    "options": request["options"], "stream": False}
            write(pending, {"id": request["id"], "started_at": now(), "prompt_sha256": request["prompt_sha256"]})
            start = time.monotonic()
            try:
                response = transport(host, "/api/chat", body, timeout=request_timeout)
            except Exception as exc:
                write(output / "status.json", {"status": "requires_attention", "responses": len(rows),
                      "id": request["id"], "error": str(exc), "updated_at": now()})
                raise
            prediction, status = parse_prediction(cell["condition"], response)
            row = {"id": request["id"], "prompt_sha256": request["prompt_sha256"], "response": response,
                   "prediction": prediction, "parse_status": status, "elapsed_seconds": time.monotonic() - start,
                   "received_at": now(), "runtime_identity": identity,
                   "truncated": response.get("done_reason") == "length"}
            with (output / "raw.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            rows[row["id"]] = row
            pending.unlink()
            write(output / "status.json", {"status": "complete" if len(rows) == len(requests) else "running",
                  "responses": len(rows), "expected": len(requests), "updated_at": now(),
                  "parse_failures": sum(r["parse_status"] != "parsed" for r in rows.values()),
                  "truncated_responses": sum(r["truncated"] for r in rows.values())})
            if len(rows) == 1:
                write(output / "loaded_model.json", transport(host, "/api/ps"))
            if len(rows) % 10 == 0:
                print(f"{model} {cell['condition']} r{cell['repeat']}: {len(rows)}/{len(requests)}", flush=True)


def summarize(metrics):
    from scripts.evaluate_rebuttal_batch import METRICS
    summary = []
    for ds in COUNTS:
        for condition in CONDITIONS:
            rows = sorted([r for r in metrics if r["dataset"] == ds and r["condition"] == condition], key=lambda r: r["repeat"])
            if [r["repeat"] for r in rows] != list(range(1, 11)):
                continue
            item = {"dataset": ds, "condition": condition}
            for metric in METRICS[:6]:
                vals = [r[metric] * 100 for r in rows]
                item[metric] = {"runs_percent": vals, "mean": statistics.mean(vals), "sample_std": statistics.stdev(vals)}
            summary.append(item)
    return summary


def evaluate(directory, model):
    directory = Path(directory).resolve()
    os.chdir(directory)
    manifest = verify(directory)
    validate_evaluator(directory, model)
    from scripts.evaluate_rebuttal_batch import STRICT_EVALUATION, METRICS
    data = load_data()
    report = {"model": model, "created_at": now(), "metrics": [], "pending_cells": [],
              "notes": manifest["notes"], "score": "strict corpus micro metrics; arguments conditional on matched actions"}
    for cell in manifest["cells"]:
        requests = read(directory / cell["input"])
        folder = directory / "outputs" / model / f"r{cell['repeat']:02d}_{cell['condition']}"
        rows = load_outputs(folder / "raw.jsonl", requests)
        if len(rows) != len(requests):
            report["pending_cells"].append({**cell, "responses": len(rows)})
            continue
        cache = folder / "metrics.json"
        digest = sha(folder / "raw.jsonl")
        if cache.exists() and read(cache)["raw_sha256"] == digest:
            report["metrics"].extend(read(cache)["metrics"])
            continue
        cell_metrics = []
        for ds, samples in data.items():
            samples = copy.deepcopy(samples)
            ds_rows = [rows[f"{ds}:{i}"] for i in range(len(samples))]
            for i, (sample, row) in enumerate(zip(samples, ds_rows)):
                sample.update(pred=row["prediction"] if row["prediction"] is not None else [], doc_id=i, domain=ds)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                values = STRICT_EVALUATION(samples, names=(ds, CONDITIONS[cell["condition"]][0], model))
            item = {"dataset": ds, "repeat": cell["repeat"], "condition": cell["condition"],
                    "num_docs": len(samples), "parse_failures": sum(r["parse_status"] != "parsed" for r in ds_rows),
                    "empty_predictions": sum(r["prediction"] == [] for r in ds_rows),
                    "truncated_responses": sum(r["truncated"] for r in ds_rows),
                    **dict(zip(METRICS, values[:6] + values[9:]))}
            cell_metrics.append(item)
            print(json.dumps(item), flush=True)
        write(cache, {"raw_sha256": digest, "metrics": cell_metrics})
        report["metrics"].extend(cell_metrics)
    report["all_requests_returned"] = not report["pending_cells"]
    report["clean_completion"] = report["all_requests_returned"] and not any(
        r["parse_failures"] or r["truncated_responses"] for r in report["metrics"])
    report["summary"] = summarize(report["metrics"])
    report["paired_full_minus_minimal"] = []
    if report["all_requests_returned"]:
        lookup = {(r["dataset"], r["condition"], r["repeat"]): r for r in report["metrics"]}
        for ds in COUNTS:
            item = {"dataset": ds}
            for metric in METRICS[:6]:
                vals = [(lookup[ds, "full", n][metric] - lookup[ds, "minimal", n][metric]) * 100 for n in range(1, 11)]
                item[metric] = {"differences_pp": vals, "mean": statistics.mean(vals), "sample_std": statistics.stdev(vals)}
            report["paired_full_minus_minimal"].append(item)
    import csv
    with (directory / "dist/code_data_supplement/expected_metrics.csv").open(encoding="utf-8", newline="") as stream:
        report["historical_baselines"] = [r for r in csv.DictReader(stream) if r["model"] == model and r["solver"] in {"nl2p_1", "gpt3_to_plan"}]
    write(directory / "reports" / f"{model}.json", report)
    print(json.dumps({"model": model, "completed_cells": len(report["metrics"]) // 3,
                      "all_requests_returned": report["all_requests_returned"], "clean_completion": report["clean_completion"]}))


def validate_evaluator(directory, model):
    """Check all six historical strict metrics before scoring fresh predictions."""
    import csv
    from scripts.evaluate_rebuttal_batch import STRICT_EVALUATION
    directory = Path(directory).resolve()
    os.chdir(directory)
    verify(directory)
    data = load_data()
    columns = ["Precision", "Recall", "F1", "Object Precision", "Object Recall", "Object F1"]
    with (directory / "dist/code_data_supplement/expected_metrics.csv").open(encoding="utf-8", newline="") as stream:
        targets = [r for r in csv.DictReader(stream) if r["model"] == model and r["solver"] in {"gpt3_to_plan", "nl2p_1"}]
    assert len(targets) == 6
    checked = []
    for target in targets:
        ds, solver = target["dataset"], target["solver"]
        records = read(directory / "results" / solver / model / f"{ds}_{solver}_{model}.json")
        samples = copy.deepcopy(data[ds])
        assert len(samples) == len(records)
        assert [r["doc_id"] for r in records] == list(range(len(samples)))
        for sample, record in zip(samples, records):
            sample["pred"] = record["prediction"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            values = STRICT_EVALUATION(samples, names=(ds, solver, model))
        for value, column in zip(values[:6], columns):
            if abs(value - float(target[column])) > 1e-12:
                raise ValueError(f"Historical evaluator mismatch: {model}/{solver}/{ds}/{column}")
        checked.append({"dataset": ds, "solver": solver})
    write(directory / "reports" / f"{model}_evaluator_validation.json", {"checked_at": now(), "configurations": checked, "strict_metric_checks": 36})
    print(f"Historical evaluator checks passed: {model}, 36 metrics", flush=True)


def submit(directory, sif, model_dir, concurrency, prerequisite=None):
    directory = Path(directory).resolve()
    manifest = verify(directory)
    if not Path(sif).is_file() or not Path(model_dir).is_dir():
        raise FileNotFoundError("Existing SIF/model cache required")
    (directory / "logs").mkdir(exist_ok=True)
    with lock(directory / "submit.lock"):
        state_path = directory / "submissions.json"
        state = read(state_path) if state_path.exists() else {"jobs": {}}
        priority = manifest["priority_model"]
        ordered = [priority] + [m for m in MODELS if m != priority]
        for model in ordered:
            for kind in ["inference", "evaluation"]:
                key = f"{model}:{kind}"
                if key in state["jobs"]:
                    if not state["jobs"][key].get("job_id"):
                        old = state["jobs"][key]
                        if old.get("returncode") == 1 and not old.get("stdout", "").strip() and "QOSMaxSubmitJobPerUserLimit" in old.get("stderr", ""):
                            state.setdefault("rejected_attempts", []).append({"key": key, **old})
                        else:
                            raise RuntimeError(f"Uncertain submission: {key}; inspect scheduler before retrying")
                    else:
                        continue
                short = model.replace("gemma3-", "g").replace("llama3-", "l")
                cmd = ["sbatch", "--parsable", f"--job-name=rb28-{short}-{kind[:3]}", f"--chdir={directory}",
                       f"--output={directory}/logs/%x_%A_%a.log", "--open-mode=append"]
                if kind == "inference":
                    shards = concurrency if model == priority else 1
                    cmd += ["--partition=gpu", f"--gres=gpu:a100:{2 if model == 'llama3-70b' else 1}",
                            "--time=6-00:00:00",
                            f"--export=ALL,CAMPAIGN_ROOT={directory},MODEL_KEY={model},SIF={sif},MODEL_DIR={model_dir},SHARDS={shards}"]
                    if shards > 1:
                        cmd += [f"--array=0-{shards - 1}%{shards}"]
                    if model != priority:
                        cmd += [f"--dependency=afterok:{state['jobs'][priority + ':inference']['job_id']}"]
                    elif prerequisite:
                        cmd += [f"--dependency=afterok:{prerequisite}"]
                    cmd += [str(directory / "hpc/rebuttal_open.slurm")]
                else:
                    cmd += [f"--dependency=afterany:{state['jobs'][model + ':inference']['job_id']}",
                            f"--export=ALL,CAMPAIGN_ROOT={directory},MODEL_KEY={model},SIF={sif}",
                            str(directory / "hpc/rebuttal_evaluate.slurm")]
                state["jobs"][key] = {"status": "submitting", "command": cmd, "started_at": now()}
                write(state_path, state)
                result = subprocess.run(cmd, capture_output=True, text=True)
                state["jobs"][key].update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
                if result.returncode != 0:
                    write(state_path, state)
                    raise RuntimeError(f"sbatch rejected {key}: {result.stderr}")
                job_id = result.stdout.strip().split(";")[0]
                if not job_id.isdigit():
                    write(state_path, state)
                    raise RuntimeError("Unrecognized submission response; inspect scheduler")
                state["jobs"][key].update(job_id=job_id, status="submitted", submitted_at=now())
                write(state_path, state)
                print(f"{key}: {job_id}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--directory", default=f"results/hpc/{CAMPAIGN}")
    p.add_argument("--priority", choices=MODELS, default="gemma3-27b")
    for command in ["verify", "run", "evaluate", "validate-evaluator", "submit"]:
        p = sub.add_parser(command)
        p.add_argument("--directory", default=".")
        if command in {"run", "evaluate", "validate-evaluator"}:
            p.add_argument("--model", choices=MODELS, required=True)
        if command == "run":
            p.add_argument("--index", type=int, choices=range(30), required=True)
            p.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
        if command == "submit":
            p.add_argument("--sif", required=True)
            p.add_argument("--model-dir", required=True)
            p.add_argument("--concurrency", type=int, choices=range(1, 5), default=2)
            p.add_argument("--prerequisite", type=int, help="Successful environment-validation Slurm job required before inference")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.directory, args.priority)
    elif args.command == "verify":
        m = verify(args.directory)
        print(json.dumps({"verified_files": len(m["files"]), "expected_requests": m["expected_requests"]}))
    elif args.command == "run":
        host = args.host if args.host.startswith("http") else "http://" + args.host
        run_cell(args.directory, args.model, args.index, host)
    elif args.command == "evaluate":
        evaluate(args.directory, args.model)
    elif args.command == "validate-evaluator":
        validate_evaluator(args.directory, args.model)
    elif args.command == "submit":
        submit(args.directory, args.sif, args.model_dir, args.concurrency, args.prerequisite)


if __name__ == "__main__":
    main()
