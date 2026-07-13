"""Recover malformed GPT3-to-Plan list responses from saved result artifacts."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

from src.evaluation_helpers import action_lemma_tokens
from src.solvers.gpt3_to_plan import GPT3ToPlan


ROOT = Path(__file__).resolve().parents[1]


def is_malformed_prediction(prediction: list[dict] | None) -> bool:
    if not prediction or len(prediction) != 1:
        return False
    verb = str(prediction[0].get("verb", ""))
    return "\n" in verb or "(" in verb or verb.lstrip().startswith(("*", "#"))


def source_grounded(actions: list[dict], sentences: list[str]) -> list[dict]:
    source_tokens = action_lemma_tokens(" ".join(sentences))
    grounded = []
    seen = set()
    for action in actions:
        verb = str(action.get("verb", "")).strip()
        verb_tokens = action_lemma_tokens(verb)
        if not verb_tokens or not verb_tokens.issubset(source_tokens):
            continue
        arguments = [str(arg).strip() for arg in action.get("arguments", []) if str(arg).strip()]
        key = (verb.casefold(), tuple(arg.casefold() for arg in arguments))
        if key in seen:
            continue
        seen.add(key)
        grounded.append({"verb": verb, "arguments": arguments})
    return grounded


def reparse(json_path: Path, pkl_path: Path, output_dir: Path) -> tuple[int, int]:
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    with pkl_path.open("rb") as handle:
        samples = pickle.load(handle)

    reparsed_rows = copy.deepcopy(rows)
    reparsed_samples = copy.deepcopy(samples)
    samples_by_id = {
        sample.get("doc_id", index): sample
        for index, sample in enumerate(reparsed_samples)
    }
    solver = GPT3ToPlan.__new__(GPT3ToPlan)
    repaired = failed = 0

    for row in reparsed_rows:
        prediction = row.get("prediction") or []
        if not is_malformed_prediction(prediction):
            continue
        raw_response = str(prediction[0].get("verb", ""))
        parsed = solver.parse_json(raw_response)
        parsed = source_grounded(parsed, row.get("sentences") or [])
        if not parsed:
            failed += 1
            continue
        row["prediction"] = parsed
        sample = samples_by_id[row["doc_id"]]
        sample["pred"] = parsed
        repaired += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / json_path.name.replace("gpt3_to_plan", "gpt3_to_plan_reparsed")
    output_pkl = output_dir / pkl_path.name.replace("gpt3_to_plan", "gpt3_to_plan_reparsed")
    output_json.write_text(json.dumps(reparsed_rows, indent=4, ensure_ascii=False), encoding="utf-8")
    with output_pkl.open("wb") as handle:
        pickle.dump(reparsed_samples, handle)
    return repaired, failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3-70b")
    parser.add_argument("--dataset", default="wikihow")
    args = parser.parse_args()

    source_dir = ROOT / "results" / "gpt3_to_plan" / args.model
    stem = f"{args.dataset}_gpt3_to_plan_{args.model}"
    output_dir = ROOT / "results" / "gpt3_to_plan_reparsed" / args.model
    repaired, failed = reparse(
        source_dir / f"{stem}.json",
        source_dir / f"{stem}.pkl",
        output_dir,
    )
    print(f"reparsed={repaired} failed={failed} output={output_dir}")


if __name__ == "__main__":
    main()
