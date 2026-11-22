"""Rebuild the November CSV reports offline from frozen scores and predictions."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.result_paths import recorded_path


def rebuild(output, workspace):
    split = Path("data/easdrl/indices/ceasdrl_two_source_v1.json")
    (workspace / split.parent).mkdir(parents=True)
    shutil.copyfile(ROOT / split, workspace / split)
    for campaign in ("four_arms_20261122", "gpt3toplan_trainpool_20261122"):
        source = ROOT / "results/batch" / campaign
        scratch = workspace / "results/batch" / campaign
        scratch.mkdir(parents=True)
        matrix = json.loads((source / "matrix.json").read_bytes())
        for index, cell in enumerate(matrix):
            original = recorded_path(cell["manifest"])
            manifest = json.loads(original.read_bytes())
            manifest["result_dir"] = str(recorded_path(manifest["result_dir"]))
            local = scratch / "cells" / str(index)
            local.mkdir(parents=True)
            receipt = local / "manifest.json"
            receipt.write_text(json.dumps(manifest), encoding="utf-8")
            cell["manifest"] = str(receipt)
            shutil.copyfile(original.with_name("scores.json"), local / "scores.json")
            if campaign == "four_arms_20261122":
                shutil.copyfile(original.with_name("interface_diagnostic.json"), local / "interface_diagnostic.json")
        (scratch / "matrix.json").write_text(json.dumps(matrix), encoding="utf-8")
        if campaign == "four_arms_20261122":
            commands = [("summarize.py",), ("report_scopes.py",)]
        else:
            commands = [("summary.py", model) for model in sorted({cell["model"] for cell in matrix})]
        for script, *args in commands:
            shutil.copyfile(source / script, scratch / script)
            subprocess.run([sys.executable, str(scratch / script), *args], check=True)
        report_dir = workspace / "reports" / campaign
        report_dir.mkdir(parents=True)
        for table in scratch.glob("*.csv"):
            shutil.copyfile(table, report_dir / table.name)
    shutil.copytree(workspace / "reports", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory for the regenerated tables")
    destination = parser.parse_args().output
    if destination.exists():
        parser.error("Choose a new output directory")
    with TemporaryDirectory() as temp:
        rebuild(destination, Path(temp))
