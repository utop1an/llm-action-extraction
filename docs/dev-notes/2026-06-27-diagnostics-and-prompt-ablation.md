# Diagnostics And Prompt Ablation

Date: 2026-06-27

## Context

This work focused on interpreting `results/experiment/nl2p_1/gpt-4o/evaluation_mismatch_diagnostics.csv` against manual samples in `annotation_analysis/samples/sampled_A.csv` and `annotation_analysis/samples/sampled_B.csv`.

The goal was to distinguish likely LLM errors from likely dataset annotation issues, especially argument-level issues that make strict evaluation understate model behavior.

## Changes

- Added stricter dataset-side argument diagnostics in `src/evaluation_helpers.py`:
  - `wrong_arguments:gold_pronoun_or_generic_reference`
  - `missing_arguments:gold_missing_valid_argument`
  - `extra_arguments:preposition_object`
  - `missing_arguments:preposition_object`
  - stricter gold split/head-word handling for `extra_arguments:unnecessary_head_or_modifier_split`
- Added and expanded tests in `tests/test_evaluation.py` for:
  - pronoun/generic-reference cases
  - gold missing-valid-argument cases
  - non-entity, adjectival, verbal, and not-in-text negative cases
  - preposition-object priority
  - gold split/head-word boundary cases
- Added `src/examples/diagnostics_overview.ipynb` to summarize diagnostics CSV characteristics:
  - mismatch type distribution
  - dataset-side and LLM-side issue counts
  - domain-specific coverage for gold split and preposition object issues
  - argument count distributions and review samples
- Added a prompt ablation variant:
  - prompt key: `nl2p_1_ablation` in `src/llm/config.py`
  - solver class: `NL2P_1_Ablation` in `src/solvers/nl2p_1.py`
  - experiment entry: `experiment.py -s nl2p_1_ablation`
  - filename parsing support in `src/evaluation_helpers.py`
  - prompt tests in `tests/test_prompts.py`

## Findings

Current diagnostics are useful as a candidate review layer, but not yet precise enough to directly support strong claims that apparent LLM errors are dataset issues.

The strongest observed candidate dataset issue families are:

- GT noun-phrase split/head-word granularity, for example `lemon, juice` vs `lemon juice`.
- Inconsistent treatment of prepositional objects.
- Gold arguments that are pronouns or generic references while the model outputs a concrete entity.
- Gold actions with empty/unknown arguments while the model extracts an entity-like direct object.

However, `preposition_object` is currently the main precision risk. It can over-fire because the diagnostic uses full `original_text` rather than the action-local sentence. This lets an argument be marked as a preposition object because it appears in a prepositional phrase somewhere else in the same document.

Examples of risky current behavior:

- `Preheat oven to 325oF` can be marked as `extra_arguments:preposition_object` because `oven` appears elsewhere as `out of the oven`.
- `Start the blender again` can be marked as `extra_arguments:preposition_object` because `blender` appears elsewhere as `in the blender`.
- Prediction spans such as `fruit chunks in the blender` can be marked as `missing_arguments:preposition_object`, even though the stronger interpretation is an LLM overlong argument span.

The LLM-side issue distribution suggests the prompt ablation should focus on general extraction behavior rather than domain-specific rules:

- missing actions, especially passive, participial, embedded, and state-change events
- wrong action heads caused by control/light verbs such as `let`, `allow`, `make sure`, and `try`
- argument boundary problems, especially overly long spans, prepositional phrases, and unresolved pronouns
- sentence-locality problems when repeated verbs occur in a paragraph

## Recommended Next Diagnostics Work

To support a high-precision dataset-issue argument, add a separate strong-evidence layer instead of treating all candidate labels as strong evidence.

Recommended fields:

- `strong_dataset_issue`
- `dataset_issue_confidence`
- `gold_sent_id`
- `gold_sentence`
- `gold_token_index`

Recommended rule changes:

- Make preposition-object diagnostics sentence-local, not paragraph-wide.
- Treat unmatched predicted arguments containing whole prepositional spans, such as `chunks in the blender`, as `extra_arguments:overlong_argument_span` rather than GT missing preposition objects.
- For strong gold split diagnostics, exclude coordinated entities connected by `and` or `or`; keep only same-NP modifier/head splits.
- Keep `missing_actions` and `wrong_action` as candidate review labels, not strong dataset-issue evidence.

## 2026-06-27 Strong Argument Diagnostics Update

Argument-level dataset diagnostics now keep only strong evidence families:

- GT noun-phrase split/head-word granularity, for example `lemon, juice` vs `lemon juice`.
- Inconsistent treatment of prepositional objects.
- Gold pronoun or generic references where the model outputs a concrete entity.

`missing_arguments:gold_missing_valid_argument` is no longer emitted as a dataset-side argument issue because it is useful for review but too weak for the strong-evidence layer. Diagnostic rows now include `strong_dataset_issue` and `dataset_issue_confidence`; argument mismatch rows populate these only when the dataset-side label is one of the strong families above.

Preposition-object evidence is now action-sentence-local. The evaluator chooses the sentence containing the matched gold action and passes that sentence to `classify_argument_mismatch`, so arguments such as `oven` in `Preheat oven` are not marked as preposition objects merely because `out of the oven` appears elsewhere in the same document. The preposition-object helper also avoids treating overlong predicted spans such as `fruit chunks in the blender` as dataset-side preposition-object omissions.

## Prompt Ablation

The new `nl2p_1_ablation` prompt keeps the original output contract but adds general rules:

- process text sentence by sentence
- extract all eventive actions, including passive, participial, embedded, and state-change events
- prefer concrete events over control/light/causative verbs
- extract only core arguments, usually direct objects or entities whose state changes
- avoid prepositions inside arguments
- resolve clear pronouns
- keep complete noun phrases as one argument, but split coordinated separate objects

Run example:

```powershell
C:\Users\Apexmod\miniforge3\envs\llm\python.exe experiment.py -s nl2p_1_ablation -m gpt-4o -d cooking
```

## Validation

Commands run during this work:

```powershell
C:\Users\Apexmod\miniforge3\envs\llm\python.exe -m pytest tests\test_evaluation.py tests\test_prompts.py -q
```

Result:

```text
52 passed
```

```powershell
C:\Users\Apexmod\miniforge3\envs\llm\python.exe -m compileall experiment.py src\llm\config.py src\solvers\nl2p_1.py src\solvers\__init__.py src\evaluation_helpers.py tests\test_evaluation.py tests\test_prompts.py
```

Result: compile succeeded.

## Current Worktree Notes

At the time this note was written, relevant edited or new files included:

- `experiment.py`
- `src/evaluation_helpers.py`
- `src/llm/config.py`
- `src/solvers/__init__.py`
- `src/solvers/nl2p_1.py`
- `tests/test_evaluation.py`
- `tests/test_prompts.py`
- `src/examples/diagnostics_overview.ipynb`
