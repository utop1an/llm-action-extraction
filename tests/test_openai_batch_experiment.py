import json

from experiment import build_result_record, load_coref_texts, sample_to_input_text
from scripts import openai_batch_experiment as batch


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
