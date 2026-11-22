"""Build and audit the anonymous code-and-data supplementary archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


METHODS = ("gpt3_to_plan", "nl2p_1", "nl2p_1_ablation", "nl2p_1_coref")
MODELS = ("gemma3-12b", "gemma3-27b", "llama3-70b", "gpt-5.4", "gpt-5.4-mini")
DATASETS = ("cooking", "wikihow", "win2k")
DATA_FILES = tuple(f"{name}_labeled_text_data.pkl" for name in DATASETS)
COREF_FILES = tuple(f"{name}_llm_coref.jsonl" for name in DATASETS)
FORBIDDEN_PARTS = {
    ".git",
    ".env",
    ".agents",
    ".codex",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
}
MAX_ARCHIVE_BYTES = 50 * 1024 * 1024


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def copy_python_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*.py")):
        copy_file(path, destination / path.relative_to(source))


def metric_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def scan_archive_tree(root: Path) -> list[str]:
    issues: list[str] = []
    literal_patterns = {
        "local user profile": b"C:" + b"\\" + b"Users" + b"\\",
        "local user profile (slash form)": b"C:/" + b"Users/",
        "workspace-specific absolute path": b"C:" + b"\\" + b"anu" + b"\\",
        "workspace-specific path (slash form)": b"C:/" + b"anu/",
        "local account identifier": b"Apex" + b"mod",
        "institution identifier": b"Australian National" + b" University",
        "submitted-source repository link": b"github.com/" + b"llm-action-extraction",
    }
    email_pattern = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    credential_patterns = {
        "OpenAI-style secret": re.compile(rb"\bsk-[A-Za-z0-9_-]{16,}\b"),
        "AWS access key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
        "literal credential assignment": re.compile(
            rb"(?i)(api[_-]?key|access[_-]?token|secret[_-]?key|password)"
            rb"\s*[:=]\s*['\"][A-Za-z0-9_./+=-]{8,}['\"]"
        ),
    }

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part.lower() in FORBIDDEN_PARTS for part in relative.parts):
            issues.append(f"forbidden path component: {relative.as_posix()}")
        if not path.is_file():
            continue
        data = path.read_bytes()
        lowered = data.lower()
        for label, pattern in literal_patterns.items():
            if pattern.lower() in lowered:
                issues.append(f"{label}: {relative.as_posix()}")
            utf16_pattern = pattern.decode("ascii").encode("utf-16le")
            if utf16_pattern.lower() in lowered:
                issues.append(f"{label} (UTF-16): {relative.as_posix()}")
        if email_pattern.search(data):
            issues.append(f"email address: {relative.as_posix()}")
        for label, pattern in credential_patterns.items():
            if pattern.search(data):
                issues.append(f"{label}: {relative.as_posix()}")
    return sorted(set(issues))


def write_manifest(root: Path) -> None:
    manifest_path = root / "MANIFEST.sha256"
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p != manifest_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(root).as_posix()
        lines.append(f"{digest}  {path.stat().st_size:>10}  {relative}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_deterministic_zip(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            relative = Path("code_data_supplement") / path.relative_to(source)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dist/code_data_supplement"))
    parser.add_argument("--zip", dest="zip_path", type=Path, default=Path("dist/code_and_data_supplement.zip"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = (root / args.output_dir).resolve()
    zip_path = (root / args.zip_path).resolve()
    allowed_parent = (root / "dist").resolve()
    if output_dir.parent != allowed_parent or zip_path.parent != allowed_parent:
        raise SystemExit("Output paths must be direct children of the repository dist directory")
    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"Output directory already exists: {output_dir}; use --force to rebuild")
        shutil.rmtree(output_dir)
    if zip_path.exists():
        if not args.force:
            raise SystemExit(f"ZIP already exists: {zip_path}; use --force to rebuild")
        zip_path.unlink()
    output_dir.mkdir(parents=True)

    copy_file(root / "supplement/README.md", output_dir / "README.md")
    copy_file(root / "supplement/CONTENTS.md", output_dir / "CONTENTS.md")
    copy_file(root / "supplement/requirements.txt", output_dir / "requirements.txt")
    for filename in ("evaluation.py", "experiment.py", "pyproject.toml"):
        copy_file(root / filename, output_dir / filename)
    copy_python_tree(root / "src", output_dir / "src")
    for filename in (
        "batch_evaluate.py",
        "openai_batch_experiment.py",
        "rebuild_result_pickles.py",
        "reproduce_supplement.py",
    ):
        copy_file(root / "scripts" / filename, output_dir / "scripts" / filename)
    for filename in (
        "test_batch_evaluate.py",
        "test_evaluation.py",
        "test_ollama_client.py",
        "test_openai_batch_experiment.py",
    ):
        copy_file(root / "tests" / filename, output_dir / "tests" / filename)
    for filename in DATA_FILES:
        copy_file(root / "data/easdrl" / filename, output_dir / "data/easdrl" / filename)
    for filename in COREF_FILES:
        copy_file(root / "data/coref_llm" / filename, output_dir / "data/coref_llm" / filename)

    frozen_table = root / "supplement/expected_metrics.csv"
    frozen = metric_rows(frozen_table)
    keys = {(row["solver"], row["model"], row["dataset"]) for row in frozen}
    expected_keys = {(method, model, dataset) for method in METHODS for model in MODELS for dataset in DATASETS}
    if keys != expected_keys or len(frozen) != len(expected_keys):
        raise ValueError("The frozen table must contain one row for every method, model and dataset")

    for method in METHODS:
        for model in MODELS:
            directory = Path("results") / method / model
            expected = [row for row in frozen if row["solver"] == method and row["model"] == model]
            actual = metric_rows(root / directory / "evaluation_result.csv")
            if sorted(actual, key=lambda r: r["dataset"]) != sorted(expected, key=lambda r: r["dataset"]):
                raise ValueError(f"Saved metrics differ from the frozen table in {directory}")
            for dataset in DATASETS:
                prediction = directory / f"{dataset}_{method}_{model}.json"
                copy_file(root / prediction, output_dir / prediction)
    copy_file(frozen_table, output_dir / "expected_metrics.csv")

    package_info = {
        "archive_type": "anonymous code and data supplement",
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "python_recommended": "3.12",
        "methods": list(METHODS),
        "datasets": list(DATASETS),
        "method_model_configurations": len(METHODS) * len(MODELS),
        "prediction_files": len(expected_keys),
        "expected_metric_rows": len(frozen),
    }
    (output_dir / "PACKAGE_INFO.json").write_text(
        json.dumps(package_info, indent=2) + "\n", encoding="utf-8"
    )
    audit_text = """Anonymous supplement build checks: PASS

Checked every staged filename and file payload (including binary data) for:
- local user-profile and workspace-specific absolute paths;
- account and institution identifiers used by the build environment;
- email addresses and common literal credential formats;
- version-control, environment, IDE, cache, and agent-only paths; and
- an external repository link for the submitted source.

The ZIP writer uses fixed timestamps and generic file permissions. Version-control
history, environment files, credentials, notebooks, and host filesystem metadata
are not included.
"""
    (output_dir / "ANONYMITY_CHECK.txt").write_text(audit_text, encoding="utf-8")

    issues = scan_archive_tree(output_dir)
    if issues:
        print("Anonymity audit failed:", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        raise SystemExit(1)

    write_manifest(output_dir)
    write_deterministic_zip(output_dir, zip_path)
    archive_size = zip_path.stat().st_size
    if archive_size > MAX_ARCHIVE_BYTES:
        raise SystemExit(
            f"Archive is {archive_size / 1024 / 1024:.2f} MiB, exceeding the 50 MiB limit"
        )
    with zipfile.ZipFile(zip_path, "r") as archive:
        bad_entry = archive.testzip()
        if bad_entry:
            raise SystemExit(f"ZIP integrity check failed at {bad_entry}")

    print(f"Built {zip_path}")
    print(f"Archive size: {archive_size / 1024 / 1024:.2f} MiB")
    print(f"Files: {sum(1 for path in output_dir.rglob('*') if path.is_file())}")
    print("Anonymity audit: PASS")


if __name__ == "__main__":
    main()
