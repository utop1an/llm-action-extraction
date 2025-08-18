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
        "template": """You are a non-expert in planning. Describe the following plan as if you are giving simple step-by-step instructions to someone. 
            Return only the description in a single paragraph, without formatting or any extra text.
            A plan is a sequence of actions, where each action is represented as either a verb or a verb phrase followed by its arguments in the form (action_name arg1?arg1_type arg2?arg2_type ...)
            When generating the description, do not generalize or replace arguments with vague words like "some object". Instead, keep the argument names exactly as they are written.
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
            A plan is a sequence of action signatures, each action signature is represented as an action verb (or verb phrase) and its arguments, in the format of (action_name arg1?arg1_type arg2?arg2_type ...)
            Return only the plan as a single line with actions separated by commas. Do not include explanations, steps, or any other text outside of the required output.
            NL: {nl}""",
        "parameters": ["nl"]
    },
    "nl2plan_correction": {
        "template": """Correct the following plan:
            {plan}
            Natural Language: {nl}""",
        "parameters": ["plan", "nl"]
    },
    "nl2p&s": {
        "template": """Convert the following natural language description into a plan and its associated states.
            A plan is a sequence of action signatures, each action signature is represented as an action verb (or verb phrase) and its arguments, in the format of (action_name arg1?arg1_type arg2?arg2_type ...).
            States are lists of predicates describing the world before and after each action.  
            Each predicate is written in the format (predicate_name arg1?arg1_type arg2?arg2_type ...).
            Return the plan in a single line with actions separated by commas. 
            Return the states, showing the state before and after each action in order.
            Do not include explanations, steps, or any other text outside of the required output.
            NL: {nl}""",
        "parameters": ["nl"]
    },
        "nl2ps-preds": {
        "template": """Convert the following natural language description into a plan and its associated states.
            A plan is a sequence of action signatures, each action signature is represented as an action verb (or verb phrase) and its arguments, in the format of (action_name arg1?arg1_type arg2?arg2_type ...).
            States are lists of predicates describing the world before and after each action.  
            Each predicate is written in the format (predicate_name arg1?arg1_type arg2?arg2_type ...).
            Return the plan in a single line with actions separated by commas. 
            Return the states, showing the state before and after each action in order. Only use the predicates provided in {predicates}. 
            Do not include explanations, steps or any other text outside of the required output.
            NL: {nl}
            Predicates: {predicates}
            """,
        "parameters": ["nl", "predicates"]
    },
        "actions": {
            "template": """
                Extract all verbs and verb phrases from the following natural language plan description.  
                Unify different verbs or verb phrases that describe the same action into a single canonical action name.  
                Return the result in the format: [verbs or verb phrases]: action_name, where [verbs or verb phrases] is a list of the original expressions found in the text, and action_name is the normalized action label they map to.  
                Use only concise canonical names (single verb or short phrase) for action_name.  
                Do not include explanations, steps, or any extra text outside of the required output.  

                NL: {nl}
                """,
            "parameters": ["nl"]
    },
        "objects": {
            "template": """
                Extract all object mentions (nouns and noun phrases) from the following natural language plan description. 
                Resolve coreferences (e.g., “it”, “them”, “the same vehicle”) and unify synonymous or variant mentions that refer to the same entity into a single canonical object name. 
                Return the result in the format: [original mentions]: canonical_object_name, where [original mentions] is a list of the distinct surface forms found in the text (preserve casing and plurality as in the text), 
                and canonical_object_name is a concise normalized label. 
                Prefer keeping given identifiers if present. 
                Do not include explanations or any extra text.
                NL: {nl}
            """,
            "parameters": ["nl"]
        }
    
}

DOMAINS = {
    "transport": {
        "description": "Tasks related to transportation and logistics.",
        "actions": "",
        "predicates": """
            (road ?l1 ?l2 - location)
            (at ?x - locatable ?v - location)
            (in ?x - package ?v - vehicle)
            (capacity ?v - vehicle ?s1 - size)
            (capacity-predecessor ?s1 ?s2 - size)"""
        }
}


TASKS = {
    "p2nl": {
        "description": "Convert a plan into a natural language description.",
        "template": """You are a non-expert in planning. Describe the following plan as if you are giving simple step-by-step instructions to someone. 
            Return only the description in a single paragraph, without formatting or any extra text.
            A plan is a sequence of actions, where each action is represented as either a verb or a verb phrase followed by its arguments in the form (action_name arg1?arg1_type arg2?arg2_type ...)
            When generating the description, do not generalize or replace arguments with vague words like "some object". Instead, keep the argument names exactly as they are written.
            Plan: {plan}""",
        "parameters": ["plan"]
    },
    "nl2p": {
        "template": """Convert the following natural language description into a plan, 
            A plan is a sequence of action signatures, each action signature is represented as an action verb (or verb phrase) and its arguments, in the format of (action_name arg1?arg1_type arg2?arg2_type ...)
            Return only the plan. Do not include explanations, steps, lists, or any extra text. Write the plan as a single line with actions separated by commas.
            NL: {nl}""",
        "parameters": ["nl"]
    },
    "nl2ps": {
        "template": """Convert the following natural language description into a plan and its associated states.
            A plan is a sequence of action signatures, each action signature is represented as an action verb (or verb phrase) and its arguments, in the format of (action_name arg1?arg1_type arg2?arg2_type ...).
            States are lists of predicates describing the world before and after each action.  
            Each predicate is written in the format (predicate_name arg1?arg1_type arg2?arg2_type ...).
            Return the plan in a single line with actions separated by commas. 
            Return the states, showing the state before and after each action in order.
            Do not include explanations, steps, or any other text outside of the required output.
            NL: {nl}""",
        "parameters": ["nl"]
    },

}

SUBTASKS = {
    
}

