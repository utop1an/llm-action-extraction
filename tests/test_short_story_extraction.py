import json

import pytest

from scripts.extract_short_stories import (
    parse_json_response,
    resolve_story,
    slugify,
)
from src.llm.config import MODELS


def test_gpt5_model_configuration_is_available():
    assert MODELS["gpt-5"]["provider"] == "openai"
    assert MODELS["gpt-5"]["model_name"] == "gpt-5"
    assert MODELS["gpt-5"]["supports_custom_sampling"] is False


def test_resolve_story_handles_requested_spelling_alias():
    stories = [
        {"Title": "The Four Skilful Brothers", "Content": "Story text."},
        {"Title": "The Three Remarks", "Content": "Other text."},
    ]
    story = resolve_story("The Four Skillful Brothers", stories)
    assert story["Title"] == "The Four Skilful Brothers"


def test_resolve_story_requires_one_exact_match():
    with pytest.raises(ValueError, match="Expected one match"):
        resolve_story("Missing Story", [])


def test_parse_json_response_accepts_fenced_json_and_validates_schema():
    raw = """```json
    [{"verb": "opened", "arguments": ["the door"]}]
    ```"""
    assert parse_json_response(raw) == [
        {"verb": "opened", "arguments": ["the door"]}
    ]


def test_parse_json_response_rejects_invalid_action_schema():
    raw = json.dumps([{"verb": "opened", "arguments": [], "extra": True}])
    with pytest.raises(ValueError, match="exactly verb and arguments"):
        parse_json_response(raw)


def test_slugify_is_stable():
    assert slugify("Chicken-Licken") == "chicken-licken"
