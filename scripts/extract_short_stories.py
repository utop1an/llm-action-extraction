"""Extract ordered action/argument records from selected short stories.

The script reads the locally downloaded ``mulab/short-stories`` JSON file,
selects the requested stories, calls the repository's existing NL2P-1 prompt,
and saves both raw model responses and validated JSON actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.llm import MODELS, generate_prompt, generate_responses


DATASET_ID = "mulab/short-stories"
DATASET_URL = "https://huggingface.co/datasets/mulab/short-stories"
DEFAULT_DATASET = Path("data/short_stories/mulab_short_stories/stories.json")
DEFAULT_SELECTED_OUTPUT = Path(
    "data/short_stories/mulab_short_stories/selected_stories.json"
)
DEFAULT_OUTPUT_DIR = Path("results/short_stories/nl2p_1/gpt-5")
DEFAULT_TITLES = (
    "Hansel and Gretel",
    "Chicken-Licken",
    "Jack and the Beanstalk",
    "The Three Remarks",
    "The Three Little Pigs",
    "The Four Skillful Brothers",
)
TITLE_ALIASES = {
    "the four skillful brothers": "The Four Skilful Brothers",
}


def load_stories(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError(f"Expected a JSON list in {path}")
    stories: list[dict[str, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"Expected an object at {path}[{index}]")
        title = record.get("Title")
        content = record.get("Content")
        if not isinstance(title, str) or not isinstance(content, str):
            raise ValueError(f"Invalid Title/Content fields at {path}[{index}]")
        stories.append({"Title": title, "Content": content})
    return stories


def resolve_story(
    requested_title: str, stories: list[dict[str, str]]
) -> dict[str, str]:
    target = TITLE_ALIASES.get(requested_title.casefold(), requested_title)
    matches = [story for story in stories if story["Title"].casefold() == target.casefold()]
    if len(matches) != 1:
        available = [story["Title"] for story in stories]
        partial = [title for title in available if target.casefold() in title.casefold()]
        raise ValueError(
            f"Expected one match for {requested_title!r}, found {len(matches)}; "
            f"partial matches={partial[:10]}"
        )
    return matches[0]


def parse_json_response(raw_response: str) -> list[dict[str, Any]]:
    match = re.search(r"```(?:json|jsonc)?\s*([\s\S]*?)\s*```", raw_response, re.I)
    payload = match.group(1).strip() if match else raw_response.strip()
    actions = json.loads(payload)
    validate_actions(actions)
    return actions


def validate_actions(actions: Any) -> None:
    if not isinstance(actions, list):
        raise TypeError("Model response must be a JSON array")
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            raise TypeError(f"Action {index} must be an object")
        if set(action) != {"verb", "arguments"}:
            raise ValueError(
                f"Action {index} must contain exactly verb and arguments; "
                f"found {sorted(action)}"
            )
        if not isinstance(action["verb"], str) or not action["verb"].strip():
            raise ValueError(f"Action {index} has an invalid verb")
        if not isinstance(action["arguments"], list) or not all(
            isinstance(argument, str) for argument in action["arguments"]
        ):
            raise ValueError(f"Action {index} has invalid arguments")


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")
    if not slug:
        raise ValueError(f"Could not create a slug for {title!r}")
    return slug


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_selected_stories(
    path: Path, selected: list[tuple[str, dict[str, str]]]
) -> None:
    payload = {
        "dataset": DATASET_ID,
        "source_url": DATASET_URL,
        "stories": [
            {
                "requested_title": requested_title,
                "dataset_title": story["Title"],
                "content": story["Content"],
                "content_sha256": hashlib.sha256(
                    story["Content"].encode("utf-8")
                ).hexdigest(),
            }
            for requested_title, story in selected
        ],
    }
    write_json(path, payload)


def write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Short-story Action Extraction",
        "",
        f"- Dataset: `{DATASET_ID}`",
        f"- Model: `{results[0]['model'] if results else 'n/a'}`",
        f"- Method: `{results[0]['method'] if results else 'n/a'}`",
        "",
        "| Requested title | Dataset title | Actions | Input tokens | Output tokens |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for result in results:
        usage = result.get("usage") or {}
        lines.append(
            "| {requested} | {dataset} | {count} | {input_tokens} | {output_tokens} |".format(
                requested=result["requested_title"].replace("|", "\\|"),
                dataset=result["dataset_title"].replace("|", "\\|"),
                count=result["action_count"],
                input_tokens=usage.get("prompt_tokens", ""),
                output_tokens=usage.get("completion_tokens", ""),
            )
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_story(
    requested_title: str,
    story: dict[str, str],
    model: str,
    method: str,
) -> dict[str, Any]:
    content = story["Content"]
    prompt = generate_prompt(method, {"nl": content})
    response = generate_responses(model, prompt, temperature=0, log=False)
    raw_response = response["content"]
    if not isinstance(raw_response, str):
        raise TypeError(f"Model returned non-text content for {requested_title!r}")
    actions = parse_json_response(raw_response)
    return {
        "dataset": DATASET_ID,
        "source_url": DATASET_URL,
        "requested_title": requested_title,
        "dataset_title": story["Title"],
        "model": model,
        "resolved_model": response.get("model"),
        "method": method,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "content_characters": len(content),
        "content_words": len(content.split()),
        "action_count": len(actions),
        "actions": actions,
        "raw_response": raw_response,
        "usage": response.get("usage"),
        "response_time_seconds": response.get("response_time"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--selected-output", type=Path, default=DEFAULT_SELECTED_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="gpt-5", choices=sorted(MODELS))
    parser.add_argument("--method", default="nl2p_1", choices=("nl2p_1", "nl2p_1_ablation"))
    parser.add_argument("--titles", nargs="+", default=list(DEFAULT_TITLES))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stories = load_stories(args.dataset)
    selected = [
        (requested_title, resolve_story(requested_title, stories))
        for requested_title in args.titles
    ]
    write_selected_stories(args.selected_output, selected)

    if args.dry_run:
        for requested_title, story in selected:
            print(
                f"{requested_title} -> {story['Title']} "
                f"({len(story['Content'])} chars, {len(story['Content'].split())} words)"
            )
        return

    results: list[dict[str, Any]] = []
    for index, (requested_title, story) in enumerate(selected, start=1):
        output_path = args.output_dir / f"{slugify(requested_title)}.json"
        if args.resume and output_path.exists():
            result = json.loads(output_path.read_text(encoding="utf-8"))
            validate_actions(result.get("actions"))
            print(f"[{index}/{len(selected)}] Reused {requested_title}")
        else:
            print(f"[{index}/{len(selected)}] Extracting {requested_title}")
            result = extract_story(requested_title, story, args.model, args.method)
            write_json(output_path, result)
            print(f"[{index}/{len(selected)}] Saved {result['action_count']} actions")
        results.append(result)

    write_json(
        args.output_dir / "results.json",
        {
            "dataset": DATASET_ID,
            "source_url": DATASET_URL,
            "model": args.model,
            "method": args.method,
            "story_count": len(results),
            "total_actions": sum(result["action_count"] for result in results),
            "results": results,
        },
    )
    write_summary(args.output_dir, results)
    print(
        f"Completed {len(results)} stories with "
        f"{sum(result['action_count'] for result in results)} actions."
    )


if __name__ == "__main__":
    main()
