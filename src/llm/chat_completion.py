import asyncio
from typing import Dict, Any, List
from .config import MODELS, PROMPTS, TEMPERATURE
from .base import BaseLLMClient
from .openai import OpenAIClient
from .ollama import OllamaClient

def get_llm_client(model_config: Dict[str, Any]) -> BaseLLMClient:
    
    if "provider" not in model_config:
        raise ValueError("Provider must be specified in model_config")
    if "model_name" not in model_config:
        raise ValueError("model_name must be specified in model_config")
    
    if model_config["provider"] == "openai":
        return OpenAIClient(model_config)
    elif model_config["provider"] == "ollama":
        return OllamaClient(model_config)
    else:
        raise ValueError(f"Unsupported provider: {model_config['provider']}")  

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

def generate_responses(model_name: str, prompt_type: str, prompt_args: Dict[str, Any] = {} , is_async: bool = False ) -> str:

    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in config: {MODELS.keys()}")
    model = MODELS[model_name]
    
    if prompt_type not in PROMPTS:
        raise ValueError(f"Prompt type {prompt_type} not found in config: {PROMPTS.keys()}")
    prompt = PROMPTS[prompt_type]["template"].format(**prompt_args)

    if is_async:
        return asyncio.run(generate_async_responses(model, prompt))
    else: 
        client = get_llm_client(model)
        return client.generate(prompt)

# if __name__ == "__main__":
#     plan = "pickup(A) stack(A, B) unstack(C,D) putdown(C)"
#     response = generate_responses('deepseek-r1:8b', 'plan2nl', is_async=False, kwargs={"plan": plan})
#     print(response["content"])
