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
    "gpt-5-nano": {
        "provider": "openai",
        "model_name": "gpt-5-nano",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL")
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
    },
    "gemma3": {
        "provider": "ollama",
        "model_name": "gemma3",
    },
    "gemma3:12b": {
        "provider": "ollama",
        "model_name": "gemma3:12b",
    },
    "llama3.2": {
        "provider": "ollama",
        "model_name": "llama3.2",
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
            - Do not add, drop, reorder, or reinterpret steps.
            - You may rephrase the action verb or verb phrase slightly to mimic how a human would retell the action, but keep the meaning unchanged.
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
    "gpt3_to_plan": {
        "template": Template("""
            $egs
            TEXT: $nl                 
            ACTIONS:
        """),
        "parameters": ["nl", "egs"]
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
    "nl2p_1": {
        "template": Template("""
            You are given a natural language paragraph describing a sequence of actions.
            
            Your task is to extract actions from the following natural language plan description.  
            
            Definition:
            - An action is an event executable by an agent that causes a state transition in the underlying system; it consists of a trigger verb and its arguments (zero or more).
            - A trigger verb is the verb or verb phrase that directly denotes an executable, state-changing action. 
            - Arguments are noun phrases denoting entities whose state is directly affectedby executing the action.
                               
            Rules:
            - Use only verbs explicitly present in the paragraph; do NOT invent verbs.
            - Verbs that are non-eventive (e.g. modal, auxiliary, linking, aspectual, light, control, advisory) should be excluded when they only frame or modify another action.
            - Maintain the order of verbs and arguments as they appear in the text.
            - Do NOT include agents, instruments, locations, manners, or temporal expressions as arguments unless their own state is directly changed by the action.
            - If an action has no arguments, return an empty list for "arguments".
            - If an argument is a pronoun, replace it with the nearest preceding noun phrase it refers to.
            - Do NOT merge multiple different entities into one argument unless they are explicitly stated as exclusive; plurals are allowed if explicitly stated in the text.
            - Do NOT add explanations or extra text.
            
            Return the result in STRICTLY a JSON array; each action item: 
            {
                "verb": "verb",
                "arguments": ["arg1", "arg2" ...]
            }
            - Output only the JSON array, with no explanations or extra text.

            Input: $nl
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
    "trigger_verbs": {
        "template": Template("""
            You are given a natural language paragraph describing actions.

            Your task is to extract the trigger verbs of actions.

            Definition:
            - An action is an event executable by an agent that causes a state transition in the underlying system; it consists of a trigger verb and its arguments (zero or more).
            - A trigger verb is the verb or verb phrase that directly denotes an executable, state-changing action. 

            Constraints:
            - Use only verbs explicitly present in the paragraph; do NOT invent verbs.
            - Verbs that are non-eventive (e.g. modal, auxiliary, linking, aspectual, light, control, advisory) should be excluded when they only frame or modify another action.
            - Maintain the order of verbs and arguments as they appear in the text.
            - If no trigger verbs are present, return an empty list.
            - Do NOT add explanations or extra text.

            Return the result STRICTLY as a JSON array of strings: ['verb1', 'verb2', ...].

            Paragraph:
            $nl

            """),
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

    
    
    "remove_non_eventive_verbs": {
        "template": Template("""
            Given a list of verbs or verb phrases extracted from the following natural language plan description, identify and REMOVE non-eventive verbs that do NOT describe concrete, observable actions or events.
            Non-eventive verbs include, but are not limited to:
                - Modal verbs that express necessity, possibility, permission, or ability (e.g., "can", "must", "should", "might").
                - Auxiliary verbs used to form tenses, moods, or voices (e.g., "is", "are", "was", "were", "have", "do").
                - Linking verbs that connect the subject to a subject complement (e.g., "seem", "become").
                - Aspectual or phase verbs that mark the temporal stage of another event rather than a distinct action (e.g., start by cooking, begin to wash, continue cleaning...).
                    In such cases, keep only the main verb describing the action ("cook", "wash", "clean") and remove the aspectual verb.
                - Light or causative verbs that serve primarily to permit, cause, or help another event rather than being observable actions themselves (e.g., "let it melt", "allow the paint to dry", "make it move").
                    In such cases, keep only the complement verb ("melt", "dry", "move") and remove the light verb.
                - Control or verification verbs (e.g., "make sure", "ensure", "verify", "check", "confirm") when followed by a clause expressing a desired state or condition.  
                    In such cases, extract the underlying action that achieves the state (e.g., "make sure the door is closed" → keep "close the door") and remove the control verb itself.
                - Advisory or reporting verbs that express recommendation, suggestion, or communication without representing an actual step (e.g., "recommend", "suggest", "advise", "say", "tell", "report").
                    In such cases, keep only the embedded or complement action (e.g., "it is recommended to add" → keep "add") and remove the advisory verb.
            
            Rules:
            - Remove only non-eventive verbs; retain all eventive verbs that describe concrete actions or events.
            - If a verb is removed, all its arguments must also be removed. If all verbs are removed, return an empty list.
            - Maintain the order of the remaining verbs as they appear in the input list.
            - Keep the input format the same as the output format, in STRICTLY a JSON array; each item:
            {
                "verb": "verb or verb phrase",
                "arguments": ["arg1", "arg2" ...]
            }
            - Output only the JSON array, with no explanations or extra text.
            
            Original Natural Language: $nl
            Input Verbs: $verbs                 
            """),
        "parameters": ["nl", "verbs"]
    },
    "identify_verb_types": {
        "template": Template("""
            Given a list of verbs or verb phrases extracted from a natural language text, identify the types of these verbs:
                - Essential verbs: core actions that are crucial to the main task or plan described in the text, without which the task cannot be completed.
                - Exclusive verbs: actions that are alternatives to other actions, where only one from a set of exclusive actions can be chosen or performed.
                - Optional verbs: supplementary actions that are not critical to the main task but may provide additional context or details, and can be omitted without affecting the overall plan.
            Rules:
            - Classify each verb into one of the three categories: "essential", "optional", or "exclusive".
            - Maintain the order of verbs as they appear in the input list; do not remove the verbs.
            - Return the result in STRICTLY a JSON array; each item:
            {
                "verb": "verb or verb phrase", // keep exactly as in the input
                "arguments": ["arg1", "arg2" ...], // keep exactly as in the input
                "type": "essential" | "optional" | "exclusive", // the verb type identified
                "related_verbs": ["index1", "index2" ...]  //list of verb indices that are mutually exclusive with this verb; for exclusive verbs, empty list for essential and optional verbs;
            }
            - Output only the JSON array, with no explanations or extra text.
                             
            Original Natural Language: $nl
            Input Verbs: $verbs  
        """),
        "parameters": ["nl", "verbs"]
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
        "template": Template("""
            Given a list of action groups and a target verb, determine which action group the verb belongs to.  

            - The action groups are provided as a two-dimensional list: [[verb or verb phrase with arguments]].  
            * Each inner list represents one action group.  
            * Items inside an inner list are alternative verb forms or phrases (with arguments) that describe the same underlying action.  
            - Verbs in the same group share the same action ontology:  
            * They may differ in surface wording, but  
            * They correspond to the same core action, having the same preconditions for execution and the same effects once executed.  

            Instructions:  
            1. Compare the target verb to the verbs in each group.  
            2. If the target verb matches the ontology of one group, return the **0-based index** of that action group.  
            3. If the verb does not belong to any group, return **-1**.  

            Return only the action_name (string), with no explanations or extra text.  

            Action groups: $actions  
            Verb: $verb  

            """),
        "parameters": ["actions","verb"]
    }
    
}

def v2a(task: any, parameters: Dict[str, Any], model: str, is_async: bool = False) -> List[List[str]]:
    verbs = parameters.get("verbs")
    if not verbs:
        raise ValueError("No verbs provided for v2a task")
    res = [[verbs[0]]]
    for verb in verbs[1:]:
        group_idx = task.get_llm_response({"actions": res, "verb": verb}, model, is_async)
        if group_idx != "-1":
            res[int(group_idx)].append(verb)
        else:
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