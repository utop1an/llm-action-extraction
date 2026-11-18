"""Score completed fresh repeat batches, explicitly retaining incomplete coverage."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import csv
from datetime import datetime
import io
import json
from pathlib import Path
import pickle
import statistics
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_rebuttal_batch import METRICS, STRICT_EVALUATION, sha
import evaluation as original_evaluation

CAMPAIGN = "nl2p_repeats_20261027_v1"
MATRIX = ROOT / "results/batch" / CAMPAIGN / "matrix.json"
ADJUSTED_METRICS = ["adjusted_argument_precision", "adjusted_argument_recall", "adjusted_argument_f1"]


def score(stem, dataset, solver, model, count, adjusted=False):
    pkl_path, json_path = Path(str(stem) + ".pkl"), Path(str(stem) + ".json")
    with pkl_path.open("rb") as stream:
        samples = pickle.load(stream)
    records = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(samples) == len(records) == count
    assert [r["doc_id"] for r in records] == list(range(count))
    assert all(s["pred"] == r["prediction"] for s, r in zip(samples, records))
    audit = {"gold_deductions": 0, "prediction_deductions": 0,
        "affected_matched_actions": 0, "deductions_by_rule": {}}
    original_deductions = original_evaluation._adjusted_object_deductions

    def counted_deductions(arg_info):
        # Observe the frozen rule's return values; do not alter its decisions.
        gold, prediction = original_deductions(arg_info)
        audit["gold_deductions"] += gold
        audit["prediction_deductions"] += prediction
        audit["affected_matched_actions"] += int(gold > 0 or prediction > 0)
        issues = set(filter(None, arg_info.get("strong_dataset_issue", "").split("|")))
        for rule, field, side in [
            ("extra_arguments:unnecessary_head_or_modifier_split", "split_missing_from_pred", "gold"),
            ("extra_arguments:preposition_object", "missing_preposition_objects", "gold"),
            ("missing_arguments:preposition_object", "extra_preposition_objects", "prediction"),
        ]:
            if rule in issues:
                audit["deductions_by_rule"].setdefault(rule, {"side": side, "count": 0})["count"] += len(arg_info.get(field, []))
        return gold, prediction

    try:
        if adjusted:
            original_evaluation._adjusted_object_deductions = counted_deductions
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            function = original_evaluation.evaluation if adjusted else STRICT_EVALUATION
            values = function(samples, names=(dataset, solver, model))
    finally:
        original_evaluation._adjusted_object_deductions = original_deductions
    metrics = dict(zip(METRICS, values[:6] + values[9:]))
    if adjusted:
        metrics.update(zip(ADJUSTED_METRICS, values[6:9]))
        metrics["adjustment_audit"] = audit
    return {"num_docs": count, "empty_predictions": sum(not r["prediction"] for r in records),
        "missing_predictions": sum(r["prediction"] is None for r in records),
        **metrics}, {
        "path": str(json_path), "json_sha256": sha(json_path), "pkl_sha256": sha(pkl_path)}


def main(model, adjusted=False):
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    for relative in ["evaluation.py", "src/evaluation_helpers.py"]:
        assert sha(ROOT / relative) == sha(MATRIX.parent / "source_snapshot" / relative)
    historical_path = ROOT / "results/rebuttal_20261027" / model / "comparison.json"
    report = {"model": model, "api_model": matrix["models"][model],
        "created_at": datetime.now(ZoneInfo("Australia/Sydney")).isoformat(timespec="seconds"),
        "matrix": str(MATRIX), "matrix_sha256_at_start": sha(MATRIX),
        "evaluator_sha256": sha(ROOT / "evaluation.py"),
        "helpers_sha256": sha(ROOT / "src/evaluation_helpers.py"),
        "analysis_script_sha256": sha(__file__), "adjusted_argument_enabled": adjusted,
        "inputs": [], "metrics": [],
        "completed_runs": [], "pending_runs": [], "summary": [],
        "notes": ["Corpus micro scores from the unchanged evaluator.",
            "Argument metrics are conditional on evaluator-matched actions.",
            "Only terminal, collected runs are scored; no partial-output subset is scored.",
            "Means and standard deviations are emitted only when both conditions have all three repeats.",
            "Historical outputs are calibration only, excluded from the fresh-repeat summaries."]}
    if adjusted:
        report["notes"].extend([
            "Adjusted F1 is argument-only and leaves action scores and strict argument true positives unchanged.",
            "Frozen heuristic exclusions are prediction-dependent, not independent semantic gold labels.",
            "Adjustment audits report raw denominator deductions before the evaluator's true-positive lower-bound clamp.",
        ])
    else:
        report["notes"].append("Adjusted argument metrics omitted in this strict-only report.")
    frozen_adjusted = {}
    if adjusted:
        with (ROOT / "dist/code_data_supplement/expected_metrics.csv").open(encoding="utf-8", newline="") as stream:
            frozen_adjusted = {r["dataset"]: r for r in csv.DictReader(stream)
                if r["model"] == model and r["solver"] == "nl2p_1"}
    if historical_path.exists():
        historical = json.loads(historical_path.read_text(encoding="utf-8"))
        assert historical["evaluator_sha256"] == report["evaluator_sha256"]
        assert historical["helpers_sha256"] == report["helpers_sha256"]
        for ds, count in matrix["datasets"].items():
            stem = ROOT / "results/nl2p_1" / model / f"{ds}_nl2p_1_{model}"
            print(f"Calibrating historical full NL2P: {ds}, adjusted={adjusted}", flush=True)
            metrics, provenance = score(stem, ds, "nl2p_1", model, count, adjusted=adjusted)
            reference = next(r for r in historical["metrics"] if r["condition"] == "nl2p_full" and r["dataset"] == ds)
            for metric in METRICS[:6]:
                assert abs(metrics[metric] - reference[metric]) < 1e-12
            if adjusted:
                for metric, column in zip(ADJUSTED_METRICS, ["adjusted_precision", "adjusted_recall", "adjusted_f1"]):
                    assert abs(metrics[metric] - float(frozen_adjusted[ds][column])) < 1e-12
        report["historical_calibration"] = "All six strict metrics reproduced for each of the three full-NL2P historical datasets."
        if adjusted:
            report["historical_calibration"] += " All three adjusted argument metrics also reproduced against the frozen table."
    for run in [r for r in matrix["runs"] if r["model"] == model]:
        manifest = json.loads(Path(run["manifest"]).read_text(encoding="utf-8"))
        label = {"condition": run["condition"], "repeat": run["repeat"], "batch_id": manifest.get("batch_id")}
        if manifest.get("batch_status") != "completed" or not manifest.get("collected_at"):
            report["pending_runs"].append({**label, "status": manifest.get("batch_status")})
            continue
        output = Path(manifest["output_jsonl"])
        raw = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
        response_models = Counter((r.get("response") or {}).get("body", {}).get("model") for r in raw)
        assert all(name in {None, report["api_model"]} for name in response_models)
        assert manifest["model_name"] == report["api_model"]
        for ds, digest in manifest["dataset_sha256"].items():
            from scripts.openai_batch_experiment import DATASETS, dataset_path
            assert sha(dataset_path(DATASETS[ds])) == digest
        report["completed_runs"].append({**label, "failure_count": manifest["failure_count"],
            "raw_output_sha256": sha(output), "response_models": dict(response_models)})
        for ds, count in matrix["datasets"].items():
            solver = manifest["result_solver"]
            stem = Path(manifest["result_dir"]) / f"{ds}_{solver}_{model}"
            print(f"Evaluating {run['condition']} repeat {run['repeat']}: {ds}, adjusted={adjusted}", flush=True)
            metrics, provenance = score(stem, ds, solver, model, count, adjusted=adjusted)
            report["metrics"].append({**label, "dataset": ds, **metrics})
            report["inputs"].append(provenance)
            print(json.dumps({**label, "dataset": ds,
                **{k: round(v*100, 4) for k, v in metrics.items() if k in METRICS[:6] + ADJUSTED_METRICS}}), flush=True)
    report["complete_three_repeat_comparison"] = len(report["completed_runs"]) == 6
    report["paired_completed_rounds"] = [repeat for repeat in [1, 2, 3]
        if {r["condition"] for r in report["completed_runs"] if r["repeat"] == repeat} == {"minimal", "full"}]
    if report["complete_three_repeat_comparison"]:
        lookup = {(r["condition"], r["repeat"], r["dataset"]): r for r in report["metrics"]}
        for ds in matrix["datasets"]:
            item = {"dataset": ds}
            for metric in METRICS[:6] + (ADJUSTED_METRICS if adjusted else []):
                values = {condition: [lookup[condition, repeat, ds][metric]*100 for repeat in [1, 2, 3]] for condition in ["minimal", "full"]}
                delta = [f-m for f, m in zip(values["full"], values["minimal"])]
                item[metric] = {condition: {"runs": scores, "mean": statistics.mean(scores),
                    "sample_std": statistics.stdev(scores)} for condition, scores in values.items()}
                item[metric]["full_minus_minimal_pp"] = {"runs": delta, "mean": statistics.mean(delta),
                    "sample_std": statistics.stdev(delta), "full_higher_runs": sum(d > 0 for d in delta),
                    "minimal_higher_runs": sum(d < 0 for d in delta), "tied_runs": sum(d == 0 for d in delta)}
            report["summary"].append(item)
    outdir = ROOT / "results" / CAMPAIGN / model
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / ("comparison_adjusted.json" if adjusted else "comparison.json")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(target), "completed_runs": len(report["completed_runs"]),
        "paired_completed_rounds": report["paired_completed_rounds"],
        "complete_three_repeat_comparison": report["complete_three_repeat_comparison"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["gpt-5.4", "gpt-5.4-mini"], default="gpt-5.4-mini")
    parser.add_argument("--adjusted", action="store_true", help="Also compute the original adjusted argument metrics and denominator deduction audit")
    args = parser.parse_args()
    main(args.model, adjusted=args.adjusted)
