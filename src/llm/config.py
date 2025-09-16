import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Dict, Any
from string import Template

load_dotenv()

# Model configurations
MODELS = {
    "gpt-4": {
        "provider": "openai",
        "model_name": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL")
    },
    "gpt-5": {
        "provider": "openai",
        "model_name": "gpt-5",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL")
    },
    "gpt-5-mini": {
        "provider": "openai",
        "model_name": "gpt-5-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL")
    },
    
    "gpt-4o-mini": {
        "provider": "openai",
        "model_name": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL")
    },
    "gpt-3.5-turbo": {
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL")
    },
    "ds-chat": {
        "provider": "openai",
        "model_name": "deepseek-chat",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL")
    },
    "ds-reasoner": {
        "provider": "openai",
        "model_name": "deepseek-reasoner",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL")
    },
    "ds-r1:8b": {
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
    "p2nl": {
        "template": """Convert the plan into a SINGLE PARAGRAPH of step-by-step instructions in the SAME ORDER as the actions.
            A plan is a sequence of actions, where each action is represented as either a verb or a verb phrase followed by its arguments in the form (action_name arg1?arg1_type arg2?arg2_type ...)
            Rules:
            - Keep every action; do not add, drop, reorder, or reinterpret steps.
            - On first mention, write each object or argument EXACTLY as given; do not generalize (no phrases like “some object”). After that, you may use unambiguous pronouns (it/that/they), but never invent new names or paraphrases.
            - Do not omit arguments or add details not present in the plan.
            - Do not add facts not present in the plan. No lists, headings, quotes, or formatting.

            Output: only the paragraph.
            Plan: {plan}""",
        "parameters": ["plan"]
    },
    "p2nl-summary": {
        "template": """Summarize the following plan into a SINGLE PARAGRAPH of step-by-step instructions in the SAME ORDER as the actions, capturing the core steps.
            A plan is a sequence of actions, where each action is represented as either a verb or a verb phrase followed by its arguments in the form (action_name arg1?arg1_type arg2?arg2_type ...)
            Rules:
            - Preserve the order and intent but MERGE or condense obvious repetition and routine steps.
            - On first mention, write each object or argument EXACTLY as given; do not generalize (no phrases like “some object”). After that, you may use unambiguous pronouns (it/that/they), but never invent new names or paraphrases.
            - Do not omit arguments or add details not present in the plan.
            - Do not add facts not present in the plan. No lists, headings, quotes, or formatting.
            
            Output: only the paragraph.
            Plan: {plan}""",
        "parameters": ["plan"]
    },
    "p2nl-detail": {
        "template": """Convert the following plan into a SINGLE PARAGRAPH of step-by-step instructions in the SAME ORDER as the actions, making the description more explicit and clear.
        
            A plan is a sequence of actions, where each action is represented as either a verb or a verb phrase followed by its arguments in the form (action_name arg1?arg1_type arg2?arg2_type ...)
            Rules:
            - Prefer explicitness over brevity; Try to add more details, especially when some actions are not clear.
            - On first mention, write each object or argument EXACTLY as given; do not generalize (no phrases like “some object”). After that, you may use unambiguous pronouns (it/that/they), but never invent new names or paraphrases.
            - Do not omit arguments. 
            - No lists, headings, quotes, or formatting.

            Output: only the paragraph.
            Plan: {plan}""",
        "parameters": ["plan"]
    },
    "nl2p": {
        "template": """Convert the following natural language description into a plan, 
            A plan is a sequence of action signatures, each action signature is represented as an action verb (or verb phrase) and its arguments, in the format of (action_name arg1?arg1_type arg2?arg2_type ...)
            Return only the plan as a single line with actions separated by commas. Do not include explanations, steps, or any other text outside of the required output.
            NL: {nl}""",
        "parameters": ["nl"]
    },
    "nl2p_refinement": {
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
    "verbs": {
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
    "srl": {
        "template": Template("""
            Extract all predicate–argument structures (semantic role labels) from the text.
            Return STRICTLY a JSON array; each item:
            
            {
            "predicate": "lemma",
            "arguments": {"ARG0": "...", "ARG1": "...", "AM-TMP": "...", ...}
            }

            Only roles present in the text may appear; values must be exact substrings.

            Rules:
            1) Predicates: detect verbal (and light-verb) predicates; normalize to the verb lemma (e.g., "was purchased" → "purchase").
            2) Roles: Use PropBank-style roles.
                - Core: ARG0 (agent/causer), ARG1 (patient/theme), ARG2... (as appropriate).
                - Adjuncts: AM-TMP (time), AM-LOC (location), AM-MNR (manner), AM-INS (instrument), AM-CAU (cause), AM-PNC (purpose), AM-DIR (direction), AM-NEG (negation).
            3) Text fidelity: Every argument value must be an exact substring from the input text; if absent, omit that role (do NOT invent).
            4) Voice & control: Handle passive/raising; if the agent is implicit and resolvable within the sentence, use it; otherwise omit ARG0.
            5) Coordination: If two verbs share arguments, output separate lines for each verb, reusing the arguments as needed.
            6) Granularity: Arguments are contiguous phrases (no discontiguous spans). Prefer the minimal informative span (e.g., "the red box" not just "box").
            7) Output hygiene: No explanations or extra text. Only lines in the specified format. Do not output empty roles.
            
            Input: $nl
            """),
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
    },
    "v2a": {
        "description": "Verb to Action Grouping",
        "template": """
            Given the following groups of actions, and verb, determine which group of action the verb belongs to, based on...
            Return only the 0-based index of the action group. If the verb doesn't belong to any of the given groups, return -1.

            Actions: {actions}
            Verb: {verb}
            """,
        "parameters": ["actions","verb"]
    }
    
}

def v2a(task: any, parameters: Dict[str, Any], model: str, is_async: bool = False) -> List[List[str]]:
    verbs = parameters.get("verbs", [])
    if not verbs:
        raise ValueError("No verbs provided for v2a task")
    res = [[verbs[0]]]
    for verb in verbs[1:]:
        added = False
        for group in res:
            is_verb_in_group = task.get_llm_response(parameters, model, is_async)
            if is_verb_in_group:
                group.append(verb)
                added = True
                break
        if not added:
            res.append([verb])
    return res

TASK_FUNCTIONS = {
    "v2a": {
        "description": "Verb to Action Grouping",
        "function": v2a

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

def generate_prompt(prompt_name: str, prompt_args: Dict[str, Any] = {}) -> str:
    if prompt_name not in PROMPTS:
        raise ValueError(f"Prompt {prompt_name} not found in config: {PROMPTS.keys()}")
    if not all(param in prompt_args for param in PROMPTS[prompt_name]["parameters"]):
        raise ValueError(f"Missing parameters for prompt {prompt_name}. Required: {PROMPTS[prompt_name]['parameters']}, Provided: {list(prompt_args.keys())}")
    if isinstance(PROMPTS[prompt_name]["template"], Template):
        prompt = PROMPTS[prompt_name]["template"].substitute(**prompt_args)
    else:
        prompt = PROMPTS[prompt_name]["template"].format(**prompt_args)
    return prompt