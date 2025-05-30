import os
from dotenv import load_dotenv

load_dotenv()

# Model configurations
MODELS = {
    "gpt-4": {
        "provider": "openai",
        "model_name": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "max_tokens": 1000,
    },
    "gpt-3.5-turbo": {
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "max_tokens": 1000,
    },
}

# Prompt templates
PROMPTS = {
    "plan2nl": {
        "template": """Convert the following plan into a natural language description:
            {plan}""",
        "parameters": ["plan"]
    },
    "plan2nl_correction": {
        "template": """Correct the following natural language description of a plan:
            {nl}
            Plan: {plan}""",
        "parameters": ["nl", "plan"]
    },
    "nl2plan": {
        "template": """Convert the following natural language description into a plan:
            {nl}""",
        "parameters": ["nl"]
    },
    "nl2plan_correction": {
        "template": """Correct the following plan:
            {plan}
            Natural Language: {nl}""",
        "parameters": ["plan", "nl"]
    }
}

# Evaluation metrics configuration
EVALUATION_METRICS = [
    "response_length",
    "response_time",
    "token_count",
    # Add more metrics as needed
] 