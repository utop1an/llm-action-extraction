import os
from dotenv import load_dotenv
from openai import OpenAI
import asyncio
from openai import AsyncOpenAI
from typing import Dict, Any, List
import pandas as pd
from config import MODELS, PROMPTS
from llm_clients import get_llm_client

# Load environment variables from .env file
load_dotenv()

# Example of how to access environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Create both sync and async clients
client = OpenAI(api_key=OPENAI_API_KEY)
async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def generate_responses(plan: str, models: List[str] = None) -> List[Dict[str, Any]]:
    """
    Generate responses from multiple LLMs for the same input plan.
    
    Args:
        plan (str): The plan to convert
        models (List[str], optional): List of model names to use. Defaults to all models in config.
    
    Returns:
        List[Dict[str, Any]]: List of responses with metadata from each model
    """
    if models is None:
        models = list(MODELS.keys())
    
    prompt = PROMPTS["plan_to_nl"]["template"].format(plan=plan)
    
    tasks = []
    for model_name in models:
        client = get_llm_client(MODELS[model_name])
        tasks.append(client.generate(prompt))
    
    return await asyncio.gather(*tasks)

def compare_responses(responses: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare responses from different models and create a DataFrame.
    
    Args:
        responses (List[Dict[str, Any]]): List of responses from different models
    
    Returns:
        pd.DataFrame: Comparison of responses and metrics
    """
    df = pd.DataFrame(responses)
    df['response_length'] = df['text'].str.len()
    return df

async def plan_to_nl_async(plan: str, models: List[str] = None) -> pd.DataFrame:
    """
    Convert a plan to natural language using multiple models and compare results.
    
    Args:
        plan (str): The plan to convert
        models (List[str], optional): List of model names to use
    
    Returns:
        pd.DataFrame: Comparison of responses from different models
    """
    responses = await generate_responses(plan, models)
    return compare_responses(responses)

def plan_to_nl(plan: str, models: List[str] = None) -> pd.DataFrame:
    """
    Synchronous wrapper for plan_to_nl_async.
    
    Args:
        plan (str): The plan to convert
        models (List[str], optional): List of model names to use
    
    Returns:
        pd.DataFrame: Comparison of responses from different models
    """
    return asyncio.run(plan_to_nl_async(plan, models))

if __name__ == "__main__":
    plan = "pickup(A) stack(A, B) unstack(C,D) putdown(C)"
    response = plan_to_nl(plan, ['gpt-4'])
    print(response)
