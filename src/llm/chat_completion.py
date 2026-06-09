import asyncio
from typing import Dict, Any, List
from .config import MODELS
from .base import BaseLLMClient
from .openai import OpenAIClient
from .ollama import OllamaClient

_CLIENT_CACHE: Dict[tuple, BaseLLMClient] = {}


def get_llm_client(model_config: Dict[str, Any]) -> BaseLLMClient:
    if "provider" not in model_config:
        raise ValueError("Provider must be specified in model_config")
    if "model_name" not in model_config:
        raise ValueError("model_name must be specified in model_config")

    cache_key = (
        model_config["provider"],
        model_config["model_name"],
        model_config.get("base_url"),
        model_config.get("api_key"),
        model_config.get("host"),
    )
    if cache_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[cache_key]
    
    if model_config["provider"] == "openai":
        client = OpenAIClient(model_config)
    elif model_config["provider"] == "ollama":
        client = OllamaClient(model_config)
    else:
        raise ValueError(f"Unsupported provider: {model_config['provider']}")
    _CLIENT_CACHE[cache_key] = client
    return client

log_dir = './logs/llm_responses'

def _append_log(model_name, provider, prompt, response, temperature):
    import os
    import json
    from datetime import datetime

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"{model_name}_{provider}.jsonl")
    log_entry = {
        "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model_name": model_name,
        "provider": provider,
        "temperature": temperature,
        "prompt": prompt,
        "response": response
    }
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    

async def generate_async_responses(client, prompt: str, temperature: float = 0) -> Dict[str, Any]:
    """
    Generate responses from multiple LLMs for the same input plan.
    
    Args:
        model (str): The model specs
        prompt (str): The prompt to use
    
    Returns:
        List[Dict[str, Any]]: List of responses with metadata from each model
    """
    return await client.generate_async(prompt, temperature=temperature)
    
    


def generate_responses(
    model_name: str,
    prompt: str,
    temperature: float = 0.5,
    is_async: bool = False,
    log: bool = False,
) -> Dict[str, Any]:
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in config: {MODELS.keys()}")
    model = MODELS[model_name]
    effective_temperature = model.get("temperature", temperature)

    client = get_llm_client(model)
    if is_async:
        response = asyncio.run(generate_async_responses(client, prompt, temperature=effective_temperature))

    else: 
        response = client.generate(prompt, temperature=effective_temperature)

    if log:
        _append_log(model_name, model['provider'], prompt, response, effective_temperature)
    return response
