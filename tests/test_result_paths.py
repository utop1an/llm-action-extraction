from pathlib import Path

import pytest

from src.result_paths import RESULTS, recorded_path, run_directory


@pytest.mark.parametrize(("run", "group"), [
    ("four_arms_20261122_r1_nl2p_ablation_rules", "nl2p_four_conditions_20261122"),
    ("gpt3toplan_trainpool_20261122_r5_ranked2", "gpt3toplan_trainpool_20261122"),
    ("gpt3_to_plan_random_seed0_rebuttal_20261027", "rebuttal_20261027/random_two_shot"),
    ("nl2p_full_repeat2_20261027_v1_server_repaired", "nl2p_repeats_20261027_v1/runs"),
])
def test_historical_receipt_and_new_output_share_one_location(tmp_path, run, group):
    prefix = tmp_path / "results"
    expected = prefix / group / run / "model"
    assert run_directory(prefix, run) / "model" == expected
    assert recorded_path(prefix / run / "model") == expected
    assert recorded_path(expected) == expected


def test_custom_runs_and_frozen_batch_paths():
    run = "four_arms_20261122_r6_nl2p_1"
    assert run_directory(RESULTS, run) == RESULTS / run
    retired = Path("llm-action-extraction-prompt-d6a5cd0/results/batch/manifest.json")
    assert recorded_path(retired) == RESULTS / "batch/manifest.json"
