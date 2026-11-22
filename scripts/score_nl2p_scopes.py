"""Score a collected Batch run on the full corpus and the shared cEASDRL test set."""
import argparse
import contextlib
import io
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import sample_to_sentences
from scripts.evaluate_rebuttal_batch import METRICS, STRICT_EVALUATION, sha

SPLIT = ROOT / "data/easdrl/indices/ceasdrl_two_source_v1.json"


def score_domain(domain, specification, predictions, report_unscorable=False):
    source = ROOT / specification["source"]
    if sha(source) != specification["sha256"]:
        raise ValueError(f"{domain}: labeled dataset hash mismatch")
    with source.open("rb") as stream:
        samples = pickle.load(stream)
    rows = json.loads(predictions.read_text(encoding="utf-8"))
    expected_ids = set(range(len(samples)))
    ids = specification["doc_ids"]
    split_ids = ids["train"] + ids["val"] + ids["test"]
    if len(split_ids) != len(samples) or set(split_ids) != expected_ids:
        raise ValueError(f"{domain}: split does not partition the labeled documents")
    if len(rows) != len(samples) or {r["doc_id"] for r in rows} != expected_ids:
        raise ValueError(f"{domain}: prediction IDs incomplete or duplicated")
    for row in rows:
        sample = samples[row["doc_id"]]
        if row["dataset"] != domain or row["sentences"] != sample_to_sentences(sample):
            raise ValueError(f"{domain}:{row['doc_id']}: prediction/source mismatch")
        sample["pred"] = row["prediction"]

    result = {"prediction_sha256": sha(predictions)}
    for scope, indices in [("full", range(len(samples))), ("shared_test", ids["test"])]:
        subset = [samples[i] for i in indices]
        error = None
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                values = STRICT_EVALUATION(subset)
        except (AttributeError, TypeError, KeyError) as exc:
            if not report_unscorable:
                raise
            values = [None] * 6
            error = f"{type(exc).__name__}: {exc}"

        result[scope] = {
            "documents": len(subset),
            "evaluation_error": error,
            "missing_predictions": sum(s["pred"] is None for s in subset),
            "empty_arrays": sum(s["pred"] == [] for s in subset),
            **dict(zip(METRICS[:6], values[:6])),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, help="Collected results/<solver>/<model>")
    parser.add_argument("--report-unscorable", action="store_true", help="Preserve the full denominator and return null metrics for a scope the original evaluator cannot score")
    args = parser.parse_args()
    directory = args.result_dir.resolve()
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    report = {
        "split_sha256": sha(SPLIT),
        "evaluator_sha256": sha(ROOT / "evaluation.py"),
        "helpers_sha256": sha(ROOT / "src/evaluation_helpers.py"),
        "domains": {},
    }
    for domain, specification in split["domains"].items():
        path = directory / f"{domain}_{directory.parent.name}_{directory.name}.json"
        report["domains"][domain] = score_domain(domain, specification, path, args.report_unscorable)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
