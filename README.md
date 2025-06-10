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

