import json

from experiment import build_result_record, load_coref_texts, result_solver_name, sample_to_input_text
from scripts import openai_batch_experiment as batch
from src.solvers import GPT3ToPlan


def test_build_result_record_keeps_compact_eval_fields():
    sample = {
        "sents": [["Pick", "up", "the", "box"]],
        "words": ["Pick", "up", "the", "box"],
        "acts": [{"act_idx": 0, "obj_idxs": [[3], []], "act_type": 1, "related_acts": []}],
    }
    prediction = [{"verb": "Pick up", "arguments": ["box"]}]

    record = build_result_record("toy", 3, "unused.pkl", sample, prediction)

    assert set(record) == {"dataset", "doc_id", "sentences", "prediction", "gold_actions"}
    assert record["dataset"] == "toy"
    assert record["doc_id"] == 3
    assert record["sentences"] == ["Pick up the box"]
    assert record["prediction"] == prediction
    assert record["gold_actions"][0]["verb"] == "Pick"


def test_build_chat_body_omits_sampling_for_gpt54_models():
    body = batch.build_chat_body("gpt-5.4-mini", "Extract actions.", temperature=0)

    assert body["model"] == "gpt-5.4-mini"
    assert body["messages"] == [{"role": "user", "content": "Extract actions."}]
    assert "temperature" not in body
    assert "top_p" not in body


def test_prepare_writes_batch_jsonl_and_manifest(tmp_path, monkeypatch):
    sample = {
        "sents": [["Pick", "up", "the", "box"]],
        "words": ["Pick", "up", "the", "box"],
        "acts": [],
    }

    monkeypatch.setattr(batch, "BATCH_ROOT", tmp_path)
    monkeypatch.setattr(batch, "DATASETS", {"toy": "toy_labeled_text_data"})
    monkeypatch.setattr(batch, "dataset_path", lambda filename: f"data/easdrl/{filename}.pkl")
    monkeypatch.setattr(batch, "read_from_labeled_dataset", lambda filename, limit=None: [sample])

    args = type(
        "Args",
        (),
        {
            "s": "nl2p_1",
            "m": "gpt-5.4-mini",
            "d": "toy",
            "l": None,
            "t": 0,
            "coref": "none",
            "coref_dir": None,
            "run_id": "test-run",
        },
    )()

    batch.prepare(args)

    out_dir = tmp_path / "nl2p_1" / "gpt-5.4-mini" / "test-run"
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (out_dir / "input.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert manifest["request_count"] == 1
    assert manifest["datasets"] == ["toy"]
    assert rows[0]["custom_id"] == "nl2p_1|gpt-5.4-mini|toy|0"
    assert rows[0]["url"] == "/v1/chat/completions"
    assert rows[0]["body"]["model"] == "gpt-5.4-mini"
    assert "Pick up the box." in rows[0]["body"]["messages"][0]["content"]


def test_prepare_gpt3_to_plan_excludes_current_doc_from_examples(tmp_path, monkeypatch):
    samples = [
        {
            "sents": [["Open", "the", "box"]],
            "words": ["Open", "the", "box"],
            "acts": [{"act_idx": 0, "obj_idxs": [[2], []], "act_type": 1, "related_acts": []}],
        },
        {
            "sents": [["Close", "the", "lid"]],
            "words": ["Close", "the", "lid"],
            "acts": [{"act_idx": 0, "obj_idxs": [[2], []], "act_type": 1, "related_acts": []}],
        },
        {
            "sents": [["Move", "the", "crate"]],
            "words": ["Move", "the", "crate"],
            "acts": [{"act_idx": 0, "obj_idxs": [[2], []], "act_type": 1, "related_acts": []}],
        },
    ]

    monkeypatch.setattr(batch, "BATCH_ROOT", tmp_path)
    monkeypatch.setattr(batch, "DATASETS", {"toy": "toy_labeled_text_data"})
    monkeypatch.setattr(batch, "dataset_path", lambda filename: f"data/easdrl/{filename}.pkl")
    monkeypatch.setattr(batch, "read_from_labeled_dataset", lambda filename, limit=None: samples)

    args = type(
        "Args",
        (),
        {
            "s": "gpt3_to_plan",
            "m": "gpt-5.4-mini",
            "d": "toy",
            "l": None,
            "t": 0,
            "coref": "none",
            "coref_dir": None,
            "run_id": "gpt3-test",
        },
    )()

    batch.prepare(args)

    out_dir = tmp_path / "gpt3_to_plan" / "gpt-5.4-mini" / "gpt3-test"
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    row = json.loads((out_dir / "input.jsonl").read_text(encoding="utf-8").splitlines()[0])
    prompt = row["body"]["messages"][0]["content"]

    assert manifest["solver"] == "gpt3_to_plan"
    assert manifest["prompt_name"] == "gpt3_to_plan"
    assert row["custom_id"] == "gpt3_to_plan|gpt-5.4-mini|toy|0"
    assert "TEXT:" in prompt
    assert "ACTIONS:" in prompt
    assert "Open the box." in prompt
    assert "Open(box)" not in prompt
    assert "Close(lid)" in prompt
    assert "Move(crate)" in prompt


def test_gpt3_to_plan_parse_json_accepts_multiword_action_names():
    solver = GPT3ToPlan({}, model_name="unused")

    assert solver.parse_json("add in(hash browns);double-click(control panel)") == [
        {"verb": "add in", "arguments": ["hash browns"]},
        {"verb": "double-click", "arguments": ["control panel"]},
    ]


def test_load_llm_coref_texts_and_sample_input_replacement(tmp_path):
    coref_path = tmp_path / "toy_llm_coref.jsonl"
    coref_path.write_text(
        json.dumps(
            {
                "domain": "toy",
                "doc_id": 2,
                "original_text": "Move it.",
                "resolved_text": "Move the box.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    coref_texts = load_coref_texts("toy", coref="llm", coref_dir=str(tmp_path))
    sample = {"sents": [["Move", "it"]], "words": ["Move", "it"], "acts": []}

    assert coref_texts == {2: "Move the box."}
    assert sample_to_input_text(sample, ds_name="toy", doc_id=2, coref_texts=coref_texts) == "Move the box."


def test_prepare_uses_llm_coref_resolved_text(tmp_path, monkeypatch):
    sample = {
        "sents": [["Move", "it"]],
        "words": ["Move", "it"],
        "acts": [],
    }
    coref_path = tmp_path / "coref" / "toy_llm_coref.jsonl"
    coref_path.parent.mkdir()
    coref_path.write_text(
        json.dumps({"domain": "toy", "doc_id": 0, "resolved_text": "Move the box."}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(batch, "BATCH_ROOT", tmp_path / "batch")
    monkeypatch.setattr(batch, "DATASETS", {"toy": "toy_labeled_text_data"})
    monkeypatch.setattr(batch, "dataset_path", lambda filename: f"data/easdrl/{filename}.pkl")
    monkeypatch.setattr(batch, "read_from_labeled_dataset", lambda filename, limit=None: [sample])

    args = type(
        "Args",
        (),
        {
            "s": "nl2p_1",
            "m": "gpt-5.4-mini",
            "d": "toy",
            "l": None,
            "t": 0,
            "coref": "llm",
            "coref_dir": str(coref_path.parent),
            "run_id": "coref-test",
        },
    )()

    batch.prepare(args)

    input_path = tmp_path / "batch" / "nl2p_1" / "gpt-5.4-mini" / "coref-test" / "input.jsonl"
    row = json.loads(input_path.read_text(encoding="utf-8").splitlines()[0])
    prompt = row["body"]["messages"][0]["content"]

    assert "Move the box." in prompt
    assert "Move it." not in prompt


def test_result_solver_name_adds_coref_suffix():
    assert batch.result_solver_name({"solver": "nl2p_1", "coref": "none"}) == "nl2p_1"
    assert batch.result_solver_name({"solver": "nl2p_1", "coref": "llm"}) == "nl2p_1_coref"
    assert (
        batch.result_solver_name({"solver": "nl2p_1_ablation", "coref": "llm"})
        == "nl2p_1_ablation_coref"
    )
    assert result_solver_name("nl2p_1", "none") == "nl2p_1"
    assert result_solver_name("nl2p_1", "llm") == "nl2p_1_coref"
    assert result_solver_name("nl2p_1_ablation", "nlp") == "nl2p_1_ablation_coref"


def test_collect_writes_coref_results_under_coref_solver(tmp_path, monkeypatch):
    sample = {
        "sents": [["Pick", "up", "the", "box"]],
        "words": ["Pick", "up", "the", "box"],
        "acts": [{"act_idx": 0, "obj_idxs": [[3], []], "act_type": 1, "related_acts": []}],
    }

    class SolverStub:
        def __init__(self, model_name):
            self.model_name = model_name

        def parse_json(self, content):
            return [{"verb": "Pick up", "arguments": ["box"]}]

    monkeypatch.setattr(batch, "ROOT", tmp_path)
    monkeypatch.setattr(batch, "RESULTS_DIR", "results")
    monkeypatch.setattr(batch, "DATASETS", {"toy": "toy_labeled_text_data"})
    monkeypatch.setattr(batch, "SUPPORTED_SOLVERS", {"nl2p_1": SolverStub})
    monkeypatch.setattr(batch, "read_from_labeled_dataset", lambda filename, limit=None: [sample])

    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "output.jsonl"
    manifest = {
        "solver": "nl2p_1",
        "model": "gpt-5.4-mini",
        "datasets": ["toy"],
        "limit": None,
        "coref": "llm",
        "run_id": "coref-test",
        "batch_id": "batch_123",
        "records": [
            {
                "custom_id": "nl2p_1|gpt-5.4-mini|toy|0",
                "dataset": "toy",
                "doc_id": 0,
                "source_file": "data/easdrl/toy.pkl",
                "original_text": "Pick up the box.",
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_path.write_text(
        json.dumps(
            {
                "custom_id": "nl2p_1|gpt-5.4-mini|toy|0",
                "response": {
                    "status_code": 200,
                    "body": {"choices": [{"message": {"content": "[]"}}]},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {
            "manifest": str(manifest_path),
            "output_jsonl": str(output_path),
            "output_file_id": None,
        },
    )()

    batch.collect(args)

    coref_result = tmp_path / "results" / "nl2p_1_coref" / "gpt-5.4-mini" / "toy_nl2p_1_coref_gpt-5.4-mini.pkl"
    baseline_result = tmp_path / "results" / "nl2p_1" / "gpt-5.4-mini" / "toy_nl2p_1_gpt-5.4-mini.pkl"
    updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert coref_result.exists()
    assert not baseline_result.exists()
    assert updated_manifest["result_solver"] == "nl2p_1_coref"


def test_collect_gpt3_to_plan_writes_standard_results(tmp_path, monkeypatch):
    sample = {
        "sents": [["Open", "the", "box"], ["Close", "the", "file"]],
        "words": ["Open", "the", "box", "Close", "the", "file"],
        "acts": [
            {"act_idx": 0, "obj_idxs": [[2], []], "act_type": 1, "related_acts": []},
            {"act_idx": 3, "obj_idxs": [[5], []], "act_type": 1, "related_acts": []},
        ],
    }

    monkeypatch.setattr(batch, "ROOT", tmp_path)
    monkeypatch.setattr(batch, "RESULTS_DIR", "results")
    monkeypatch.setattr(batch, "DATASETS", {"toy": "toy_labeled_text_data"})
    monkeypatch.setattr(batch, "read_from_labeled_dataset", lambda filename, limit=None: [sample])

    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "output.jsonl"
    manifest = {
        "solver": "gpt3_to_plan",
        "model": "gpt-5.4-mini",
        "datasets": ["toy"],
        "limit": None,
        "coref": "none",
        "run_id": "gpt3-test",
        "batch_id": "batch_123",
        "records": [
            {
                "custom_id": "gpt3_to_plan|gpt-5.4-mini|toy|0",
                "dataset": "toy",
                "doc_id": 0,
                "source_file": "data/easdrl/toy.pkl",
                "original_text": "Open the box. Close the file.",
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_path.write_text(
        json.dumps(
            {
                "custom_id": "gpt3_to_plan|gpt-5.4-mini|toy|0",
                "response": {
                    "status_code": 200,
                    "body": {"choices": [{"message": {"content": "open(box);close(file)"}}]},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {
            "manifest": str(manifest_path),
            "output_jsonl": str(output_path),
            "output_file_id": None,
        },
    )()

    batch.collect(args)

    result_path = tmp_path / "results" / "gpt3_to_plan" / "gpt-5.4-mini" / "toy_gpt3_to_plan_gpt-5.4-mini.json"
    pkl_path = tmp_path / "results" / "gpt3_to_plan" / "gpt-5.4-mini" / "toy_gpt3_to_plan_gpt-5.4-mini.pkl"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert pkl_path.exists()
    assert result[0]["prediction"] == [
        {"verb": "open", "arguments": ["box"]},
        {"verb": "close", "arguments": ["file"]},
    ]
    assert set(result[0]) == {"dataset", "doc_id", "sentences", "prediction", "gold_actions"}
    assert updated_manifest["result_solver"] == "gpt3_to_plan"
