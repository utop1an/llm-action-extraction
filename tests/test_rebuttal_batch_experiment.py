import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import openai_batch_experiment as batch
from src.llm import generate_prompt
from src.solvers import GPT3ToPlan, NL2P_Minimal


def samples():
    return [
        {
            "sents": [[f"verb{i}", f"object{i}"]],
            "words": [f"verb{i}", f"object{i}"],
            "acts": [{"act_idx": 0, "obj_idxs": [[1], []], "act_type": 1, "related_acts": []}],
        }
        for i in range(12)
    ]


def test_random_selection_excludes_query_is_reproducible_and_ignores_labels():
    data = samples()
    first = GPT3ToPlan({"toy": data}, "gpt-5.4", "random", 0)
    changed = copy.deepcopy(data)
    for i, sample in enumerate(changed):
        sample["acts"][0]["act_type"] = 2 if i % 2 else 3
    second = GPT3ToPlan({"other": samples(), "toy": changed}, "gpt-5.4-mini", "random", 0)
    assert first.ranked_example_indices["toy"] == second.ranked_example_indices["toy"]
    for doc_id in range(len(data)):
        chosen = first.selected_example_indices("toy", doc_id)
        assert len(chosen) == len(set(chosen)) == 2
        assert doc_id not in chosen
        assert chosen == second.selected_example_indices("toy", doc_id)
        assert first.build_prompt("QUERY", "toy", doc_id) == second.build_prompt("QUERY", "toy", doc_id)
    distinct = {tuple(GPT3ToPlan({"toy": data}, "unused", "random", seed).ranked_example_indices["toy"][:2]) for seed in range(5)}
    assert len(distinct) == 5


def test_random_requires_seed_and_preserves_original_ranked_selection():
    with pytest.raises(ValueError, match="explicit selection_seed"):
        GPT3ToPlan({"toy": samples()}, example_selection="random")
    data = samples()
    data[2]["acts"][0]["act_type"] = 2
    data[7]["acts"][0]["act_type"] = 3
    solver = GPT3ToPlan({"toy": data})
    assert solver.selected_example_indices("toy", 0) == [7, 2]
    assert solver.selected_example_indices("toy", 7) == [2, 11]


def test_minimal_is_exact_deletion_preserving_output_contract_and_input():
    paragraph = "Definition: is part of this input. Move the box."
    full = generate_prompt("nl2p_1", {"nl": paragraph})
    start = full.index("            Definition:")
    end = full.index("            - Output valid JSON only.", start)
    minimal = generate_prompt("nl2p_minimal", {"nl": paragraph})
    assert minimal == full[:start] + full[end:]
    assert "Rules:" not in minimal
    assert "directly affected" not in minimal
    assert "Resolve pronouns" not in minimal
    assert minimal.endswith(full[full.index("            Paragraph:"):])
    assert NL2P_Minimal("unused").parse_json('[{"verb":"Move","arguments":["box"]}]') == [{"verb": "Move", "arguments": ["box"]}]


def prepare_toy(tmp_path, monkeypatch, seed=2):
    monkeypatch.setattr(batch, "BATCH_ROOT", tmp_path / "batch")
    monkeypatch.setattr(batch, "DATASETS", {"toy": "toy"})
    monkeypatch.setattr(batch, "read_from_labeled_dataset", lambda filename, limit=None: samples())
    args = SimpleNamespace(
        s="gpt3_to_plan_random", m="gpt-5.4-mini", d=None, l=None, t=0,
        coref="none", coref_dir=None, run_id=f"seed{seed}", selection_seed=seed,
        api_model="gpt-5.4-mini-2026-03-17", result_solver=None,
    )
    batch.prepare(args)
    path = batch.run_dir(args.s, args.m, args.run_id) / "manifest.json"
    return args, path, batch.load_manifest(path)


def test_prepare_records_actual_model_examples_hashes_and_seed_namespace(tmp_path, monkeypatch):
    args, path, manifest = prepare_toy(tmp_path, monkeypatch)
    rows = batch.read_jsonl(path.parent / "input.jsonl")
    assert manifest["result_solver"] == "gpt3_to_plan_random_seed2"
    assert manifest["selection_seed"] == 2
    assert manifest["input_sha256"] == hashlib.sha256((path.parent / "input.jsonl").read_bytes()).hexdigest()
    for record, row in zip(manifest["records"], rows):
        assert record["doc_id"] not in record["example_doc_ids"]
        assert len(record["example_doc_ids"]) == 2
        assert set(row["body"]) == {"model", "messages"}
        assert row["body"]["model"] == "gpt-5.4-mini-2026-03-17"
        prompt = row["body"]["messages"][0]["content"]
        assert hashlib.sha256(prompt.encode()).hexdigest() == record["prompt_sha256"]
    with pytest.raises(FileExistsError):
        batch.prepare(args)


def test_collect_random_keeps_missing_docs_and_separates_seeds(tmp_path, monkeypatch):
    _, path, manifest = prepare_toy(tmp_path, monkeypatch)
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    monkeypatch.setattr(batch, "RESULTS_DIR", "results")
    output = path.parent / "output.jsonl"
    batch.write_jsonl(output, [{
        "custom_id": manifest["records"][0]["custom_id"],
        "response": {"status_code": 200, "body": {"choices": [{"message": {"content": "verb0(object0)"}}]}},
    }])
    batch.collect(SimpleNamespace(manifest=str(path), output_jsonl=str(output), output_file_id=None, result_solver=None))
    folder = tmp_path / "results/gpt3_to_plan_random_seed2/gpt-5.4-mini"
    records = json.loads(next(folder.glob("*mini.json")).read_text(encoding="utf-8"))
    assert len(records) == 12
    assert records[0]["prediction"] == [{"verb": "verb0", "arguments": ["object0"]}]
    assert all(record["prediction"] is None for record in records[1:])
    assert batch.load_manifest(path)["failure_count"] == 11
    assert not (tmp_path / "results/gpt3_to_plan/gpt-5.4-mini").exists()
    assert batch.result_solver_name({"solver": "gpt3_to_plan_random", "selection_seed": 3}) == "gpt3_to_plan_random_seed3"


def test_submit_is_resumable_and_does_not_duplicate_remote_job(tmp_path, monkeypatch):
    _, path, manifest = prepare_toy(tmp_path, monkeypatch)
    calls = []

    def upload(**kwargs):
        calls.append("upload")
        assert kwargs["purpose"] == "batch"
        return SimpleNamespace(id="file_test")

    def create(**kwargs):
        calls.append("create")
        stored = batch.load_manifest(path)
        assert stored["input_file_id"] == "file_test"
        assert stored["submission_started_at"]
        return SimpleNamespace(id="batch_test", status="validating")

    client = SimpleNamespace(files=SimpleNamespace(create=upload), batches=SimpleNamespace(create=create))
    client.with_options = lambda **kwargs: client
    monkeypatch.setattr(batch, "get_openai_client", lambda model: client)
    batch.submit(SimpleNamespace(manifest=str(path)))
    batch.submit(SimpleNamespace(manifest=str(path)))
    assert calls == ["upload", "create"]
    assert batch.load_manifest(path)["batch_id"] == "batch_test"


def test_submit_recovers_uncertain_create_and_rejects_changed_input(tmp_path, monkeypatch):
    _, path, manifest = prepare_toy(tmp_path, monkeypatch)
    manifest.update(input_file_id="file_test", submission_started_at="previous_attempt")
    batch.write_json(path, manifest)
    client = SimpleNamespace(batches=SimpleNamespace(list=lambda **kwargs: [SimpleNamespace(input_file_id="file_test", id="batch_recovered", status="in_progress")]))
    client.with_options = lambda **kwargs: client
    monkeypatch.setattr(batch, "get_openai_client", lambda model: client)
    batch.submit(SimpleNamespace(manifest=str(path)))
    assert batch.load_manifest(path)["batch_id"] == "batch_recovered"
    batch.write_json(path, manifest)
    (path.parent / "input.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after preparation"):
        batch.submit(SimpleNamespace(manifest=str(path)))
