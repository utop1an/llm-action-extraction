from src.llm import generate_prompt
from src.llm.config import MODELS
from src.solvers import NL2P_1_Ablation


def _read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def test_nl2p_1_ablation_prompt_is_available_and_contains_ablation_rules():
    prompt = generate_prompt("nl2p_1_ablation", {"nl": "Let the mixture simmer."})

    assert "Let the mixture simmer." in prompt
    assert "Process the paragraph sentence by sentence" in prompt
    assert "Prefer the concrete state-changing event over framing" in prompt
    assert "Do not include prepositions inside arguments" in prompt
    assert "Do not extract general background facts" in prompt
    assert "source, destination, or container" in prompt


def test_nl2p_1_ablation_solver_uses_ablation_prompt_name():
    solver = NL2P_1_Ablation(model_name="gpt-4o")

    assert solver.prompt_name == "nl2p_1_ablation"
    assert solver.log_prefix == "nl2p_1_ablation"


def test_experiment_open_source_model_aliases_are_available():
    assert MODELS["llama3-70b"]["provider"] == "ollama"
    assert MODELS["llama3-70b"]["model_name"] == "llama3.3:70b"
    assert MODELS["gemma3-12b"]["provider"] == "ollama"
    assert MODELS["gemma3-12b"]["model_name"] == "gemma3:12b"
    assert MODELS["gemma3-27b"]["provider"] == "ollama"
    assert MODELS["gemma3-27b"]["model_name"] == "gemma3:27b"
    assert MODELS["llama3-3b"]["provider"] == "ollama"
    assert MODELS["llama3-3b"]["model_name"] == "llama3.2:3b"


def test_llama_slurm_names_match_ollama_models():
    llama33_70 = _read_text("hpc/llama33_70.slurm")
    llama32_3 = _read_text("hpc/llama32_3.slurm")

    assert 'OLLAMA_MODEL="llama3.3:70b"' in llama33_70
    assert 'MODEL_KEY="llama3-70b"' in llama33_70
    assert 'OLLAMA_MODEL="llama3.2:3b"' in llama32_3
    assert 'MODEL_KEY="llama3-3b"' in llama32_3
