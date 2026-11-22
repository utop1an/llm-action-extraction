import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_hpc_rebuttal as runner
from tests.test_rebuttal_batch_experiment import samples


def test_requests_preserve_prompt_sampling_and_all_ten_selection_seeds():
    data = {"toy": samples()}
    chosen = []
    for repeat in range(1, 11):
        rows = runner.make_requests(data, "rand2", repeat)
        assert [r["id"] for r in rows] == [f"toy:{i}" for i in range(12)]
        assert all(r["options"] == {"temperature": 0.5} for r in rows)
        assert all(r["selection_seed"] == repeat - 1 for r in rows)
        assert all(r["doc_id"] not in r["example_doc_ids"] for r in rows)
        assert rows == runner.make_requests(copy.deepcopy(data), "rand2", repeat)
        chosen.append(tuple(rows[0]["example_doc_ids"]))
    assert len(set(chosen)) > 1
    for condition in ["minimal", "full"]:
        assert runner.make_requests(data, condition, 1) == runner.make_requests(data, condition, 10)
        assert all(r["options"] == {"temperature": 0} for r in runner.make_requests(data, condition, 1))


def fake_campaign(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    requests = runner.make_requests({"toy": samples()[:3]}, "minimal", 1)
    runner.write(tmp_path / "input.json", requests)
    manifest = {"cells": [{"index": 0, "condition": "minimal", "repeat": 1, "input": "input.json"}], "files": {}}
    runner.write(tmp_path / "campaign.json", manifest)
    monkeypatch.setattr(runner, "pin_runtime", lambda *a: {"digest": "frozen"})
    return requests, tmp_path / "outputs/gemma3-27b/r01_minimal"


def test_resume_never_reissues_parsing_failures_or_successes(tmp_path, monkeypatch):
    requests, folder = fake_campaign(tmp_path, monkeypatch)
    calls = []

    def transport(host, route, body=None, **kwargs):
        if route == "/api/ps":
            return {"models": []}
        calls.append(body)
        return {"done": True, "message": {"content": "invalid JSON" if len(calls) == 1 else "[]"}}

    runner.run_cell(tmp_path, "gemma3-27b", 0, "http://localhost", transport=transport)
    runner.run_cell(tmp_path, "gemma3-27b", 0, "http://localhost", transport=transport)
    assert len(calls) == 3
    rows = runner.load_outputs(folder / "raw.jsonl", requests)
    assert len(rows) == 3
    assert rows["toy:0"]["parse_status"] == "parse_failure"
    assert rows["toy:1"]["prediction"] == []


def test_transport_failure_blocks_uncertain_retry(tmp_path, monkeypatch):
    _, folder = fake_campaign(tmp_path, monkeypatch)
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise TimeoutError("uncertain")

    with pytest.raises(TimeoutError):
        runner.run_cell(tmp_path, "gemma3-27b", 0, "http://localhost", transport=fail)
    with pytest.raises(RuntimeError, match="Uncertain"):
        runner.run_cell(tmp_path, "gemma3-27b", 0, "http://localhost", transport=fail)
    assert len(calls) == 1
    assert runner.read(folder / "status.json")["status"] == "requires_attention"


def test_crash_after_durable_response_resumes_without_duplicate(tmp_path, monkeypatch):
    requests, folder = fake_campaign(tmp_path, monkeypatch)
    calls = []

    def transport(host, route, body=None, **kwargs):
        if route == "/api/ps":
            return {}
        calls.append(1)
        return {"done": True, "message": {"content": "[]"}}

    runner.run_cell(tmp_path, "gemma3-27b", 0, "http://localhost", transport=transport)
    runner.write(folder / "inflight.json", {"id": requests[-1]["id"]})
    runner.run_cell(tmp_path, "gemma3-27b", 0, "http://localhost", transport=transport)
    assert len(calls) == 3
    assert not (folder / "inflight.json").exists()


def test_duplicate_or_changed_raw_records_rejected(tmp_path):
    requests = runner.make_requests({"toy": samples()[:3]}, "full", 1)
    path = tmp_path / "raw.jsonl"
    row = {"id": requests[0]["id"], "prompt_sha256": requests[0]["prompt_sha256"]}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        runner.load_outputs(path, requests)
    row["prompt_sha256"] = "changed"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        runner.load_outputs(path, requests)


@pytest.mark.parametrize("text", ['null', '{}', '[{"verb":"mix","arguments":"flour"}]'])
def test_invalid_schema_is_a_failure_not_empty_success(text):
    prediction, status = runner.parse_prediction("full", {"done": True, "message": {"content": text}})
    assert prediction is None
    assert status == "parse_failure"


def test_runtime_change_fails_closed(tmp_path, monkeypatch):
    digest = ["one"]
    def api(host, route, body=None):
        return {"/api/tags": {"models": [{"name": "gemma3:27b", "digest": digest[0]}]},
                "/api/show": {"parameters": "fixed"}, "/api/version": {"version": "fixed"}}[route]
    monkeypatch.setattr(runner, "api", api)
    runner.pin_runtime(tmp_path, "gemma3-27b", "http://localhost")
    digest[0] = "two"
    with pytest.raises(ValueError, match="changed"):
        runner.pin_runtime(tmp_path, "gemma3-27b", "http://localhost")


def test_file_change_invalidates_frozen_campaign(tmp_path):
    path = tmp_path / "data.json"
    path.write_text("original")
    runner.write(tmp_path / "campaign.json", {"files": {"data.json": runner.sha(path)}})
    runner.verify(tmp_path)
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        runner.verify(tmp_path)


def test_scheduler_prioritizes_27b_and_does_not_resubmit(tmp_path, monkeypatch):
    sif = tmp_path / "model.sif"
    sif.touch()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    runner.write(tmp_path / "campaign.json", {"files": {}, "priority_model": "gemma3-27b"})
    commands = []
    def submit(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(commands)), stderr="")
    monkeypatch.setattr(runner.subprocess, "run", submit)
    runner.submit(tmp_path, sif, model_dir, 2, prerequisite=99)
    runner.submit(tmp_path, sif, model_dir, 2, prerequisite=99)
    assert len(commands) == 6
    assert "--array=0-1%2" in commands[0]
    assert not any(x.startswith("--array") for x in commands[2] + commands[4])
    assert "--dependency=afterok:99" in commands[0]
    assert "--dependency=afterok:101" in commands[2]
    assert "--dependency=afterok:101" in commands[4]
    assert "--dependency=afterany:101" in commands[1]
    assert "--gres=gpu:a100:2" in commands[4]


def test_uncertain_submission_is_not_repeated(tmp_path, monkeypatch):
    sif = tmp_path / "model.sif"
    sif.touch()
    runner.write(tmp_path / "campaign.json", {"files": {}, "priority_model": "gemma3-27b"})
    runner.write(tmp_path / "submissions.json", {"jobs": {"gemma3-27b:inference": {"status": "submitting"}}})
    with pytest.raises(RuntimeError, match="Uncertain submission"):
        runner.submit(tmp_path, sif, tmp_path, 2)
