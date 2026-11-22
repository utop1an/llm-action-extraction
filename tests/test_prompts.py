from pathlib import Path

import pytest

from src.llm import generate_prompt
from src.llm.config import MODELS
from src.solvers import NL2P_1_Ablation


def test_nl2p_1_ablation_prompt_is_available_and_contains_ablation_rules():
    prompt = generate_prompt("nl2p_1_ablation", {"nl": "Let the mixture simmer."})

    assert "Let the mixture simmer." in prompt
    assert "distinct event" in prompt
    assert "role in context, not merely from the presence of a verb" in prompt
    assert "main, subordinate, passive, participial, or conditional constructions" in prompt
    assert 'aspectual or control constructions such as "start to X"' in prompt
    assert "treat the embedded event X as an action candidate" in prompt
    assert "only when it also denotes a distinct procedural step" in prompt
    assert "Use the action trigger stated in the paragraph" in prompt
    assert "If no reliable argument is available, use an empty list" in prompt
    assert "Do not include prepositions inside arguments" in prompt
    assert "hypothetical background" in prompt
    assert "Coverage check" in prompt
    assert "use, check, start" not in prompt


def test_nl2p_1_ablation_solver_uses_ablation_prompt_name():
    solver = NL2P_1_Ablation(model_name="gpt-4o")

    assert solver.prompt_name == "nl2p_1_ablation"
    assert solver.log_prefix == "nl2p_1_ablation"


@pytest.mark.parametrize(("key", "model", "job"), [
    ("llama3-70b", "llama3.3:70b", "llama33_70.slurm"),
    ("gemma3-12b", "gemma3:12b", "gemma3_12.slurm"),
    ("gemma3-27b", "gemma3:27b", "gemma3_27.slurm"),
])
def test_benchmark_ollama_models_match_slurm_jobs(key, model, job):
    assert MODELS[key]["provider"] == "ollama"
    assert MODELS[key]["model_name"] == model
    script = (Path(__file__).resolve().parents[1] / "hpc" / job).read_text(encoding="utf-8")
    assert f'OLLAMA_MODEL="{model}"' in script
    assert f'MODEL_KEY="{key}"' in script
