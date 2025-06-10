import os
import asyncio
from typing import Dict, Any, List
from config import MODELS, PROMPTS, TEMPERATURE
from llm_clients import get_llm_client

async def generate_async_responses(model: str, prompt: str) -> str:
    """
    Generate responses from multiple LLMs for the same input plan.
    
    Args:
        plan (str): The plan to convert
        model_name (str): The name of the model to use
        prompt_type (str): The type of prompt to use
    
    Returns:
        List[Dict[str, Any]]: List of responses with metadata from each model
    """
    
    
    client = get_llm_client(model)
    response = await client.generate_async(prompt)
    
    return response

def generate_responses(model_name: str, prompt_type: str, is_async: bool = False, kwargs: Dict[str, Any] = {}) -> str:

    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in config: {MODELS.keys()}")
    model = MODELS[model_name]
    
    if prompt_type not in PROMPTS:
        raise ValueError(f"Prompt type {prompt_type} not found in config: {PROMPTS.keys()}")
    prompt = PROMPTS[prompt_type]["template"].format(**kwargs)

    if is_async:
        return asyncio.run(generate_async_responses(model, prompt))
    else: 
        client = get_llm_client(model)
        return client.generate(prompt)

if __name__ == "__main__":
    plan = "pickup(A) stack(A, B) unstack(C,D) putdown(C)"
    response = generate_responses('deepseek-r1:8b', 'plan2nl', is_async=False, kwargs={"plan": plan})
    print(response["content"])
