import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Dict, Any

load_dotenv()

# Model configurations
MODELS = {
    "gpt-4": {
        "provider": "openai",
        "model_name": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    "gpt-5": {
        "provider": "openai",
        "model_name": "gpt-5",
        "api_key": os.getenv("OPENAI_API_KEY"),  
    },
    "gpt-5-mini": {
        "provider": "openai",
        "model_name": "gpt-5-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    
    "gpt-4o-mini": {
        "provider": "openai",
        "model_name": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    "gpt-3.5-turbo": {
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    "deepseek-r1:8b": {
        "provider": "ollama",
        "model_name": "deepseek-r1:8b",
    }
}

# Evaluation metrics configuration
EVALUATION_METRICS = [
    "response_length",
    "response_time",
    "token_count",
    # Add more metrics as needed
] 

TEMPERATURE = 0.7


@dataclass
class PromptConfig:
    template: str
    parameters: List[str]

@dataclass
class TaskConfig:
    default: str
    description: str
    versions: Dict[str, PromptConfig]

PROMPTS = {
    "plan2nl": {
        "template": """You are an non-expert in planning, describe the following plan as how people tell instructions/processes, return only the description in a single paragraph, without formatting or any other text.
        A plan is a sequence of actions, each action is represented as an action verb and its arguments, in the form of (action_name arg1?arg1_type arg2?arg2_type ...).
            Plan: {plan}""",
        "parameters": ["plan"]
    },
    "plan2nl_refinement": {
        "template": """Refine the following natural language description generated from the following Plan, return only the description without formatting or any other text.
            NL: {nl}
            Plan: {plan}""",
        "parameters": ["nl", "plan"]
    },
    "nl2plan": {
        "template": """Convert the following natural language description into a plan, 
            A plan is a sequence of action signatures, each action signature is represented as an action verb and its arguments, in the format of (action_name arg1?arg1_type arg2?arg2_type ...)
            return only the plan in a single line in the above format and without any other text.:
            NL: {nl}""",
        "parameters": ["nl"]
    },
    "nl2plan_correction": {
        "template": """Correct the following plan:
            {plan}
            Natural Language: {nl}""",
        "parameters": ["plan", "nl"]
    }
}


TASKS = {
    "plan2nl": {
        "description": "Convert a plan into a natural language description.",
        "prompt": PROMPTS["plan2nl"]["template"],
        "parameters": PROMPTS["plan2nl"]["parameters"]
    },
}

SUBTASKS = {
    
}

