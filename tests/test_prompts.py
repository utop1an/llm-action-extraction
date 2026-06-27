from src.llm import generate_prompt
from src.solvers import NL2P_1_Ablation


def test_nl2p_1_ablation_prompt_is_available_and_contains_ablation_rules():
    prompt = generate_prompt("nl2p_1_ablation", {"nl": "Let the mixture simmer."})

    assert "Let the mixture simmer." in prompt
    assert "Process the paragraph sentence by sentence" in prompt
    assert "Prefer the concrete event over control/light/causative verbs" in prompt
    assert "Do not include prepositions inside arguments" in prompt


def test_nl2p_1_ablation_solver_uses_ablation_prompt_name():
    solver = NL2P_1_Ablation(model_name="gpt-4o")

    assert solver.prompt_name == "nl2p_1_ablation"
    assert solver.log_prefix == "nl2p_1_ablation"
