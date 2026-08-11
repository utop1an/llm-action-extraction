"""Re-evaluate all supplementary predictions and verify expected metrics."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

from rebuild_result_pickles import rebuild_result_file, rebuild_tree


KEY_COLUMNS = ("dataset", "solver", "model")
METRIC_COLUMNS = (
    "Precision",
    "Recall",
    "F1",
    "Object Precision",
    "Object Recall",
    "Object F1",
    "adjusted_precision",
    "adjusted_recall",
    "adjusted_f1",
)
COUNT_COLUMNS = (
    "perfect_action_argument_matches",
    "argument_mismatch_actions",
    "matched_action_events",
)


def read_rows(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {tuple(row[column] for column in KEY_COLUMNS): row for row in rows}


def result_directories(results_root: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in results_root.rglob("*.json")
        if path.name.startswith(("cooking_", "wikihow_", "win2k_"))
        and not path.name.endswith("_summary.json")
    )


def compare_rows(
    expected: dict[tuple[str, str, str], dict[str, str]],
    actual: dict[tuple[str, str, str], dict[str, str]],
    tolerance: float,
) -> list[str]:
    errors: list[str] = []
    if set(expected) != set(actual):
        errors.append(
            f"row keys differ: missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    for key in sorted(set(expected) & set(actual)):
        for column in METRIC_COLUMNS:
            expected_value = float(expected[key][column])
            actual_value = float(actual[key][column])
            if not math.isclose(
                expected_value,
                actual_value,
                rel_tol=tolerance,
                abs_tol=tolerance,
            ):
                errors.append(
                    f"{key} {column}: expected {expected_value}, got {actual_value}"
                )
        for column in COUNT_COLUMNS:
            if expected[key][column] != actual[key][column]:
                errors.append(
                    f"{key} {column}: expected {expected[key][column]}, "
                    f"got {actual[key][column]}"
                )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="verify one dataset and method/model configuration")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--clean", action="store_true", help="remove reconstructed PKL files after verification")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    results_root = root / "results"
    expected_path = root / "expected_metrics.csv"
    expected_all = read_rows(expected_path)
    directories = sorted(set(result_directories(results_root)))
    if args.quick:
        directories = directories[:1]

    verified_rows = 0
    generated_pickles: list[Path] = []
    all_errors: list[str] = []
    for directory in directories:
        if args.quick:
            candidate = sorted(directory.glob("cooking_*.json"))[0]
            generated_pickles.append(
                rebuild_result_file(
                    candidate,
                    root / "data/easdrl",
                    root / "data/coref_llm",
                    overwrite=True,
                )
            )
        else:
            generated_pickles.extend(
                rebuild_tree(
                    directory,
                    root / "data/easdrl",
                    root / "data/coref_llm",
                    overwrite=True,
                )
            )
        command = [sys.executable, str(root / "evaluation.py"), "-d", str(directory)]
        if args.diagnostics:
            command.append("--diagnostics")
        subprocess.run(command, cwd=root, check=True)

        actual = read_rows(directory / "evaluation_result.csv")
        expected = {key: expected_all[key] for key in actual if key in expected_all}
        errors = compare_rows(expected, actual, args.tolerance)
        all_errors.extend(f"{directory.relative_to(root)}: {error}" for error in errors)
        verified_rows += len(actual)

    if args.clean:
        for path in generated_pickles:
            path.unlink(missing_ok=True)

    if all_errors:
        print("Metric verification failed:", file=sys.stderr)
        for error in all_errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)

    expected_count = 1 if args.quick else len(expected_all)
    if verified_rows != expected_count:
        raise SystemExit(
            f"Expected to verify {expected_count} rows, but verified {verified_rows}."
        )
    print(f"Verified {verified_rows} metric rows with tolerance {args.tolerance:g}.")


if __name__ == "__main__":
    main()
