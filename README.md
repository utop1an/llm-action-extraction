# LLM-based Action (State) Extraction

# Input
A natural language description of a sequence of actions (possibly no state descriptions)

# Methodologies

## Baseline approach
One prompt (zero or few shots) to get a structured output


## Advanced approach
Break the extraction into smaller steps, to ask the LLM to do different small tasks, roughly

- identify and merge objects and object types
- identify and merge action names, determine action arity
- infer predicate names, determine predicate arity
- identify and infer missing vals
- identify and correct noisy vals
- ...
- integration and get the final output


## Self-validation
Using input + output to ask LLM to validate and improve its answer for each step.

## Constraint/Rules/Expert knowledge
Adding constraint or rules or other expert knowledge in the prompt...

## Zero shot / One shot / Few shots

# Output
 
- Object: obj_name?obj_type
    - noun?noun
- Action Signature: action_name(obj, obj...)
    - verb(noun, noun...)
- Predicate: predicate_name(obj, obj...)
    -  noun/adj(noun, noun...)

## Pure action extraction
Extract only actions, expected output:
```
Input: Pick up object A, place it on top of B to form a stack, separate C from D (if stacked), and then put down object C.

Output: pickup(A) stack(A, B) unstack(C,D) putdown(C)
```

## Action and State extraction
Extract actions and possible intermediate states,  expected output:
```
Input: Pick up object A, place it on top of B to form a stack, separate C from D (if stacked), and then put down object C.

Output: 
s0: **None**, handempty(), on(C, D)
s1: **pickup(A)**, holding(A), on(C, D)
s2: **stack(A, B)**, on(A, B), on(C, D)
s3:  **unstack(C,D)**, holding(C) 
s4: **putdown(C)**, handempty()
```

## Models

### chatgpt

#### gpt-4

### deepseek
locally deployed via [ollama](https://ollama.com/)

```shell
ollama run deepseek-r1:8b
```
 
ollama python library ([github](https://github.com/ollama/ollama-python))

#### r1:8b


# Experiment settings

## MEMO

### GPT3-TO-PLAN
Replication of the gpt3-to-plan paper, with 2-shot prompting (which has overall good performance with less tokens, the default setting in the paper?), using examples with the highest propotion of OP and EX actions (as mentioned in the paper).

### NL2P
3 step extraction:
1. extract verbs and args
2. remove non-eventive verbs
3. label verb types (ES, EX, OP)

### NL2P-1
1 step extraction of eventive verbs and their args (with simple explanation of eventive verbs)

### VerbArgs
1 step extraction of verbs and args only 


### Possible Improvements
- Traditional NLP/LLM based POS first
- Extract verbs only (no args)
- Better definition of eventive verbs
- Experiment on different Temperature settings

# Report

## Possible problems

### GT Labelled actions?
- Some are not really an action, or a step? "after mixed", "until well grinded", ... (and sometimes they are not extracted, labelled as JJ in the the dataset)

### GT Labelled args?
- No comma, when multiple args... (also for verbs, "add something... top with something...")
- Some are not labelled, "mix a b and c (in a bowl)", "do something (with something)"
- it should be an arg, but labelled as a verb, "using the opener to open the can"


### Predicated actions
- do something before doing something, "doing something" should be a step after, often be omitted