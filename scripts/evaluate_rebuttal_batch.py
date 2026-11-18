"""Evaluate completed rebuttal predictions against the frozen original baselines.

Uses the original strict scoring with explicit condition names, avoiding
filename parsing ambiguity for the seed-specific result namespaces. Skips only
the expensive adjusted-argument diagnostic branch; no adjusted metrics reported.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import csv
from datetime import datetime
import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path
import pickle
import statistics
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluation
import src.evaluation_helpers as helpers

# Repeated action phrases have identical deterministic token lemmas. Memoize
# this pure operation without changing the evaluator's matching or scoring.
helpers.action_lemma_tokens = lru_cache(maxsize=None)(helpers.action_lemma_tokens)

METRICS = [
    "action_precision", "action_recall", "action_f1", "argument_precision",
    "argument_recall", "argument_f1",
    "perfect_action_argument_matches", "argument_mismatch_actions", "matched_action_events",
]


def strict_evaluator():
    tree = ast.parse((ROOT / "evaluation.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluation")

    class SkipAdjustedDiagnostics(ast.NodeTransformer):
        removed = 0

        def visit_If(self, node):
            if ast.unparse(node.test) == "obj_right < obj_true or obj_right < obj_tagged":
                self.removed += 1
                return None
            return self.generic_visit(node)

    transform = SkipAdjustedDiagnostics()
    function = transform.visit(function)
    assert transform.removed == 1
    function.name = "strict_only_evaluation"
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(evaluation.__dict__)
    exec(compile(module, "<original evaluator without adjusted diagnostics>", "exec"), namespace)
    return namespace[function.name]


STRICT_EVALUATION = strict_evaluator()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--matrix", default="results/batch/rebuttal_20261027_v1/matrix.json")
    args = parser.parse_args()
    matrix_path = (ROOT / args.matrix).resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    outdir = ROOT / "results/rebuttal_20261027" / args.model
    outdir.mkdir(parents=True, exist_ok=True)
    for rel in ["evaluation.py", "src/evaluation_helpers.py"]:
        assert sha(ROOT / rel) == sha(matrix_path.parent / "source_snapshot" / rel), rel
    for record in matrix["baseline_files"]:
        assert sha(record["path"]) == record["sha256"], record["path"]
    with (ROOT / "dist/code_data_supplement/expected_metrics.csv").open(encoding="utf-8", newline="") as f:
        frozen = {(r["solver"], r["dataset"]): r for r in csv.DictReader(f) if r["model"] == args.model}

    runs = [r for r in matrix["runs"] if r["model"] == args.model]
    assert len(runs) == 6
    conditions = [
        {"condition": "gpt3_ranked", "solver": "gpt3_to_plan", "directory": ROOT / "results/gpt3_to_plan" / args.model},
        {"condition": "nl2p_full", "solver": "nl2p_1", "directory": ROOT / "results/nl2p_1" / args.model},
    ]
    for run in runs:
        manifest = json.loads(Path(run["manifest"]).read_text(encoding="utf-8"))
        assert manifest["batch_status"] == "completed", (run["condition"], manifest["batch_status"])
        assert manifest.get("collected_at"), run["condition"]
        assert manifest["model_name"] == matrix["models"][args.model]
        assert manifest["failure_count"] == 0, (run["condition"], manifest["failure_count"])
        conditions.append({"condition": run["condition"], "solver": manifest["result_solver"],
            "directory": Path(manifest["result_dir"]), "batch_id": manifest["batch_id"]})

    report = {
        "model": args.model, "api_model": matrix["models"][args.model],
        "created_at": datetime.now(ZoneInfo("Australia/Sydney")).isoformat(timespec="seconds"),
        "evaluator_sha256": sha(ROOT / "evaluation.py"),
        "helpers_sha256": sha(ROOT / "src/evaluation_helpers.py"),
        "analysis_script_sha256": sha(__file__),
        "data_counts": matrix["datasets"], "inputs": [], "metrics": [],
        "notes": [
            "F1 values are corpus micro-F1 from the unchanged original evaluator.",
            "Random summaries average the five corpus-level scores; std uses ddof=1.",
            "Argument scores are conditional on evaluator-matched actions.",
            "No significance testing; selection-run variability also includes model output variation.",
            "Original baselines are saved historical outputs from the same API snapshot.",
            "Memoized the deterministic action-token operation; all six strict baseline metrics are checked against the frozen table.",
            "Skipped only the adjusted-argument diagnostic branch; no adjusted metrics reported.",
        ],
    }
    for condition in conditions:
        for ds, expected_count in matrix["datasets"].items():
            stem = condition["directory"] / f"{ds}_{condition['solver']}_{args.model}"
            pkl_path = Path(str(stem) + ".pkl")
            json_path = Path(str(stem) + ".json")
            with pkl_path.open("rb") as f:
                samples = pickle.load(f)
            records = json.loads(json_path.read_text(encoding="utf-8"))
            assert len(samples) == len(records) == expected_count
            assert [r["doc_id"] for r in records] == list(range(expected_count))
            assert all(sample["pred"] == record["prediction"] for sample, record in zip(samples, records))
            print(f"Evaluating {condition['condition']} {ds} ({expected_count} documents)", flush=True)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                values = STRICT_EVALUATION(samples, names=(ds, condition["solver"], args.model))
            metrics = dict(zip(METRICS, values[:6] + values[9:]))
            if condition["condition"] in {"gpt3_ranked", "nl2p_full"}:
                target = frozen[(condition["solver"], ds)]
                for metric, column in [("action_precision", "Precision"), ("action_recall", "Recall"),
                    ("action_f1", "F1"), ("argument_precision", "Object Precision"),
                    ("argument_recall", "Object Recall"), ("argument_f1", "Object F1")]:
                    assert abs(metrics[metric] - float(target[column])) < 1e-12, (condition["condition"], ds, metric)
            report["metrics"].append({"condition": condition["condition"], "dataset": ds, "num_docs": expected_count,
                "empty_predictions": sum(not r["prediction"] for r in records), **metrics})
            report["inputs"].append({"path": str(json_path), "sha256": sha(json_path), "pkl_sha256": sha(pkl_path)})
            print(f"  action={metrics['action_f1']*100:.4f}, argument={metrics['argument_f1']*100:.4f}", flush=True)

    lookup = {(r["condition"], r["dataset"]): r for r in report["metrics"]}
    report["summary"] = []
    for ds in matrix["datasets"]:
        item = {"dataset": ds}
        for metric in ["action_f1", "argument_f1"]:
            random_values = [lookup[(f"random-seed{s}", ds)][metric] for s in matrix["selection_seeds"]]
            item[metric] = {
                "gpt3_ranked": lookup[("gpt3_ranked", ds)][metric],
                "random_mean": statistics.mean(random_values),
                "random_std": statistics.stdev(random_values),
                "random_min": min(random_values), "random_max": max(random_values),
                "random_minus_ranked_pp": 100*(statistics.mean(random_values)-lookup[("gpt3_ranked", ds)][metric]),
                "minimal": lookup[("minimal", ds)][metric],
                "nl2p_full": lookup[("nl2p_full", ds)][metric],
                "full_minus_minimal_pp": 100*(lookup[("nl2p_full", ds)][metric]-lookup[("minimal", ds)][metric]),
                "full_minus_random_mean_pp": 100*(lookup[("nl2p_full", ds)][metric]-statistics.mean(random_values)),
            }
        report["summary"].append(item)
    (outdir / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (outdir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report["metrics"][0]))
        writer.writeheader()
        writer.writerows(report["metrics"])
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved comparison: {outdir / 'comparison.json'}")


if __name__ == "__main__":
    main()
