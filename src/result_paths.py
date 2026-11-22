"""Locate grouped runs without changing the run IDs stored in experiment receipts."""
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
_GROUPS = {
    r"four_arms_20261122_r[1-5]_nl2p_(?:1|minimal|ablation_definitions|ablation_rules)": "nl2p_four_conditions_20261122",
    r"gpt3toplan_trainpool_20261122_r[1-5]_(?:random2|ranked2)": "gpt3toplan_trainpool_20261122",
    r"gpt3_to_plan_random_seed[0-4]_rebuttal_20261027": "rebuttal_20261027/random_two_shot",
    r"nl2p_(?:full|minimal)_repeat[1-3]_20261027_v1(?:_server_repaired)?": "nl2p_repeats_20261027_v1/runs",
}


def run_directory(results_root, run_name):
    root = Path(results_root)
    for pattern, group in _GROUPS.items():
        if re.fullmatch(pattern, run_name):
            return root / group / run_name
    return root / run_name


def recorded_path(path):
    """Resolve an old results/<run> path, including paths to retired worktrees."""
    parts = str(path).replace("\\", "/").split("/")
    for i, part in enumerate(parts[:-1]):
        if part == "results":
            run = parts[i + 1]
            if i and parts[i - 1] in {
                "llm-action-extraction-prompt-d6a5cd0",
                "llm-action-extraction-integrated-20261123",
            }:
                return run_directory(RESULTS, run).joinpath(*parts[i + 2:])
            root = Path("/".join(parts[:i + 1]))
            return run_directory(root, run).joinpath(*parts[i + 2:])
    return Path(path)


def verify_integration_manifest():
    import hashlib
    import json

    root = RESULTS.parent
    manifest = root / "docs/dev-notes/2026-11-23-result-file-map.json"
    records = json.loads(manifest.read_bytes())["files"]
    for record in records:
        path = recorded_path(root / record.get("variant_path", record["path"]))
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != record["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {path}")
    print(f"Verified {len(records)} source records.")


if __name__ == "__main__":
    verify_integration_manifest()
