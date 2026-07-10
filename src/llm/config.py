import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Dict, Any
from string import Template

load_dotenv()

# Model configurations
MODELS = {
    "gpt-4.1": {
        "provider": "openai",
        "model_name": "gpt-4.1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL")
    },
    "gpt-4.1-mini": {
        "provider": "openai",
        "model_name": "gpt-4.1-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL")
    },
    "gpt-4.1-nano": {
        "provider": "openai",
        "model_name": "gpt-4.1-nano",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL")
    },
    "gpt-5.4": {
        "provider": "openai",
        "model_name": "gpt-5.4",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "supports_custom_sampling": False,
    },
    "gpt-5.4-mini": {
        "provider": "openai",
        "model_name": "gpt-5.4-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "supports_custom_sampling": False,
    },
    "gpt-4o": {
        "provider": "openai",
        "model_name": "gpt-4o",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL")
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model_name": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL")
    },
    "ds-r1:8b": {
        "provider": "ollama",
        "model_name": "deepseek-r1:8b",
    },
    "gemma3-12b": {
        "provider": "ollama",
        "model_name": "gemma3:12b",
    },
    "gemma3-27b": {
        "provider": "ollama",
        "model_name": "gemma3:27b",
    },
    "llama3-70b": {
        "provider": "ollama",
        "model_name": "llama3.3:70b",
    }
}

# Evaluation metrics configuration
EVALUATION_METRICS = [
    "response_length",
    "response_time",
    "token_count",
    # Add more metrics as needed
] 

TASK_FUNCTIONS = {}

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
            - Do not add, drop, reorder, or reinterpret steps.
            - You may rephrase the action verb or verb phrase slightly to mimic how a human would retell the action, but keep the meaning unchanged.
            - On first mention, write each object or argument EXACTLY as given; do not generalize (no phrases like “some object”). After that, you may use unambiguous pronouns (it/that/they), but never invent new names or paraphrases.
            - Do not omit arguments or add details not present in the plan.
            - Do not add facts not present in the plan. No lists, headings, quotes, or formatting.

            Output: only the paragraph.
            Plan: {plan}""",
        "parameters": ["plan"]
    },
    "gpt3_to_plan": {
        "template": Template("""
            $egs
            TEXT: $nl                 
            ACTIONS:
        """),
        "parameters": ["nl", "egs"]
    },
    "nl2p_1": {
        "template": Template("""
            You are given a natural language paragraph that may include both actions and contextual information.

            Your task is to extract the executable actions explicitly described in the paragraph.
            
            Definition:
            - An action is a concrete, executable event that changes the state, location, configuration, possession, or availability of an entity.
            - A trigger verb is the verb or verb phrase in the text that denotes that executable action.
            - An argument is a noun phrase denoting an entity directly affected by the action.
                               
            Rules:
            - Use only actions explicitly supported by the paragraph.
            - Preserve action order.
            - Do not invent actions, arguments, object types, or hidden preconditions.
            - Extract concrete eventive actions, not modal, auxiliary, linking, reporting, advisory, or purely descriptive verbs.
            - When a phrase contains both a framing verb and a concrete event, use the concrete event as the action trigger. Avoid using verbs whose main role is to frame, check, start, continue, recommend, or enable another event unless that verb itself changes an entity.
            - Do not include agents such as "you" as arguments.
            - Arguments are core affected entities only. Do not include time, duration, purpose, manner, or instrument phrases unless the phrase denotes an entity whose state is directly changed.
            - Resolve pronouns such as it, them, this, these, those, and everything to the nearest unambiguous entity or entity set in the preceding context.
            - Keep a complete noun phrase as one argument when it denotes one object.
            - Keep coordinated entities as separate arguments when they are separate objects.
            - Use concise surface forms from the text for arguments.
            - When an action has no explicit object, infer its argument from the immediately preceding local context only if the action clearly applies to the current affected entity or working set. Otherwise, use an empty list.
            - Output valid JSON only.

            Return a JSON array. Each item must have exactly this form:
            [
                {
                    "verb": "trigger verb or verb phrase",
                    "arguments": ["argument 1", "argument 2" ...]
                }
            ]

            Paragraph:
            $nl
            """),
        "parameters": ["nl"]
    },
    "nl2p_1_ablation": {
        "template": Template("""
            You are given a natural language paragraph that may contain multiple explicit or embedded executable actions, as well as non-action context.

            Your task is to extract all and only the executable actions explicitly described in the paragraph.

            Definition:
            - An action is a concrete, executable event that changes the state, location, configuration, possession, or availability of an entity.
            - A trigger verb is the verb or verb phrase in the text that denotes that executable action.
            - An argument is a noun phrase denoting an entity directly affected by the action.

            General extraction rules:
            - Use only actions explicitly supported by the paragraph.
            - Preserve action order.
            - Process the paragraph sentence by sentence. Do not borrow actions or arguments from a different sentence unless a pronoun or ellipsis clearly refers back to it.
            - Do not invent actions, arguments, object types, or hidden preconditions.
            - Do not include agents such as "you" as arguments.
            - Output must be valid JSON only.

            Action coverage rules:
            - Extract all eventive actions, not only the main imperative verb.
            - Include explicit passive, participial, embedded, conditional, and state-change events when they denote real executable steps.
            - Include actions in before/after/while/until/if clauses when they denote real executable events.
            - Exclude modal, auxiliary, linking, reporting, advisory, and purely descriptive verbs when they do not denote an executable step.
            - Prefer the concrete state-changing event over framing, control, light, aspectual, or causative verbs.
            - For constructions such as "start to X", "continue X-ing", "try to X", "make sure X is done", "allow X to Y", or "let X Y", extract X/Y when X/Y is the real executable action.
            - Keep the trigger verb close to the wording in the paragraph, but choose the event head rather than surrounding helper words.

            Argument rules:
            - Arguments should be core affected entities, usually direct objects or entities whose state, location, configuration, possession, or availability changes.
            - Do not include time, duration, temperature, condition, purpose, manner, location, source, destination, container, or instrument phrases as arguments unless that entity itself is directly changed.
            - Do not include prepositions inside arguments. Prefer the core noun phrase over the full prepositional phrase.
            - For transfer, placement, insertion, removal, or movement actions, include the moved or changed entity. Include the source, destination, or container only if it is also directly acted on or changed.
            - Resolve pronouns such as "it", "them", "this", "these", "those", and "everything" to the most specific preceding entity or entity set when the antecedent is unambiguous. If ambiguous, keep the original pronoun text.
            - Keep a complete noun phrase as one argument when it denotes one object.
            - Keep coordinated entities as separate arguments only when they are separate affected objects.
            - Use concise surface forms from the text for arguments.
            - When an action has no explicit object, infer its argument from the immediately preceding local context only if the action clearly applies to the current affected entity or working set. Otherwise, use an empty list.

            Return a JSON array. Each item must have exactly this form:
            [
                {
                    "verb": "trigger verb or verb phrase",
                    "arguments": ["argument 1", "argument 2" ...]
                }
            ]

            Paragraph:
            $nl
            """),
        "parameters": ["nl"]
    },
    "verbs": {
        "template": """
            You are given a natural language paragraph describing a sequence of actions.
            Your task is to extract all candidate action verbs.  
            
            Definition:
            - An action verb (or verb phrase) is an eventive verb that denotes a state transition in the underlying system.
            
            Rules:
            - Maintain the order of verbs as they appear in the text.
            - Do NOT add explanations or extra text.
            
            Return the result in a STRICT JSON array: ["verb1", "verb2", ...]

            NL: {nl}
            """,
        "parameters": ["nl"]
    },
    "args": {
        "template": Template("""
            You are given a natural language paragraph describing a sequence of actions.

            Your task is to extract all candidate arguments of actions from the paragraph.

            Definition:
            - An action is an event executable by an agent that causes a state transition in the underlying system; it consists of a trigger verb and its arguments (zero or more).
            - Arguments are noun phrases denoting entities whose state is directly affected by executing the action.

            Rules:
            - Do NOT include agents, instruments, locations, manners, or temporal expressions as arguments unless their own state is directly changed by the action.
            - If an argument is a pronoun, replace it with the nearest preceding noun phrase it refers to.
            - Do NOT merge multiple different entities into one argument unless they are explicitly stated as exclusive; plurals are allowed if explicitly stated in the text.
            - Preserve the order in which the noun phrases appear in the paragraph.
            - If no candidate arguments are present, return an empty list.
            - Do NOT add explanations or extra text.

            Return the result STRICTLY as a JSON array of strings: ["arg1", "arg2", ...].

            NL: $nl

            """),
        "parameters": ["nl"]
    },
    "nl2p_2_verb_args": {
        "template": Template("""
            You are given:
            (1) a natural language paragraph describing a sequence of actions
            (2) a list of action verbs extracted from the paragraph

            Your task is to assign arguments to each verb.
            
            Definition:
                - Arguments are noun phrases denoting entities whose state is directly affected by executing the action.

            Rules:
            - Use ONLY the verbs provided in the list.
            - Do NOT include agents, instruments, locations, manners, or temporal expressions as arguments unless their own state is directly changed by the action.
            - If an argument is a pronoun, replace it with the nearest preceding noun phrase it refers to.
            - Do NOT merge multiple different entities into one argument unless they are explicitly stated as exclusive; plurals are allowed if explicitly stated in the text.
            - Maintain the order of verbs and arguments as they appear in the text.             
            - If a verb has no explicit arguments, return an empty list of arguments.
            - Do NOT add explanations or extra text.
                             
            Return the result STRICTLY as a JSON array, where each item has the form:
            {
                "verb": "verb",
                "arguments": ["arg1", "arg2", ...]
            }

            NL: $nl
            
            Verbs: $verbs
            """),
        "parameters": ["nl", "verbs"]
    },

    "nl2p_3_verb_args": {
        "template": Template("""
            You are given:
            (1) a natural language paragraph describing a sequence of actions
            (2) a list of action verbs extracted from the paragraph
            (3) a list of argument candidates extracted from the paragraph

            Your task is to assign arguments to each verb.

            Rules:
            - Use ONLY the verbs and arguments provided in the list.
            - Do NOT include agents, instruments, locations, manners, or temporal expressions as arguments unless their own state is directly changed by the action.
            - If an argument is a pronoun, replace it with the nearest preceding noun phrase it refers to.
            - Do NOT merge multiple different entities into one argument unless they are explicitly stated as exclusive; plurals are allowed if explicitly stated in the text.
            - Do NOT merge multiple candidates into one argument.
            - Maintain the order of verbs and arguments as they appear in the text.             
            - If a verb has no explicit arguments, return an empty list of arguments.
            - Do NOT infer or assume missing arguments.
            - Do NOT add explanations or extra text.

            Return the result STRICTLY as a JSON array, where each item has the form:
            {
                "verb": "verb",
                "arguments": ["arg1", "arg2", ...]
            }

            NL: $nl
            
            Verbs: $verbs
                             
            Arguments: $args
            """),
        "parameters": ["nl", "verbs", "args"]
    },
    "llm_coref_resolution": {
        "template": Template("""
            You are given an instructional text from a procedural dataset.

            Rewrite the text with entity-only coreference resolution.

            Replace a pronoun or demonstrative ONLY when all of the following are true:
            - The mention is one of: "it", "its", "itself", "they", "them", "their", "themselves", "this", "that", "these", or "those".
            - The mention clearly refers to a concrete entity or entities that already appeared in the text.
            - The antecedent can be written as a noun phrase, such as an object, ingredient, tool, file, window, button, person, place, or other named entity.

            Do NOT replace the mention if it refers to:
            - A previous action, event, process, step, instruction, state, condition, result, fact, or whole sentence.
            - A verb phrase, adjective phrase, clause, or abstract idea.
            - An ambiguous antecedent.

            Editing rules:
            - Preserve the original step order, actions, entities, and facts.
            - Do not add new steps, remove steps, or infer hidden objects.
            - Do not convert events or actions into noun phrases.
            - Make the minimum changes needed for clear entity coreference.
            - You may add commas and basic punctuation only when it does not change meaning.
            - Keep the text as one paragraph.

            Output only the rewritten paragraph. Do not output JSON, bullets, markdown, explanations, or labels.

            TEXT:
            $nl
        """),
        "parameters": ["nl"]
    },
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
