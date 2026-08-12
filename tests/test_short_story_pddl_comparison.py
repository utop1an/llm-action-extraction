from pathlib import Path

import pytest

from scripts.compare_short_story_pddl import (
    PDDLAction,
    PDDLDomain,
    infer_argument_type,
    lexical_similarity,
    parse_domain,
    type_overlap,
)


def test_parse_domain_extracts_typed_action_parameters(tmp_path: Path):
    path = tmp_path / "domain.pddl"
    path.write_text(
        """(define (domain sample)
        (:requirements :strips :typing)
        (:types entity item location - object)
        (:constants wolf pig - entity pot - item house - location)
        (:predicates (alive ?x - entity))
        (:action wolf_eats_pig
          :parameters (?p ?w - entity ?h - location)
          :precondition (alive ?w)
          :effect (not (alive ?p))))""",
        encoding="utf-8",
    )

    domain = parse_domain(path)

    assert domain.name == "sample"
    assert domain.constants["pot"] == "item"
    assert domain.actions == (
        PDDLAction("wolf_eats_pig", ("entity", "entity", "location")),
    )


def test_lexical_similarity_normalizes_inflection_and_ignores_roles():
    assert lexical_similarity("ate up", "wolf_eats_pig") == pytest.approx(2 / 3)
    assert lexical_similarity("built", "build_house") > 0


def test_argument_type_inference_uses_constants_and_shared_nouns():
    domain = PDDLDomain(
        "sample",
        ("entity", "item", "location"),
        {"wolf": "entity", "straw_house": "location", "brick_house": "location"},
        (),
    )
    assert infer_argument_type("the wolf", domain)[:2] == ("entity", "wolf")
    assert infer_argument_type("the little pig's house", domain)[0] == "location"
    assert infer_argument_type("a mysterious promise", domain)[0] == "unknown"


def test_type_overlap_compares_type_multisets():
    assert type_overlap(["entity", "item"], ["entity", "item", "location"]) == pytest.approx(0.8)
    assert type_overlap(["unknown"], ["entity"]) is None
