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

## 2026-06-28 Conversation Summary

### Argument Dataset-Issue Scope

The current high-precision argument diagnosis policy is to keep only these strong dataset issue families:

- GT noun-phrase split/head-word granularity, for example `lemon, juice` vs `lemon juice`.
- Inconsistent treatment of prepositional objects.
- Gold arguments that are pronouns or generic references while the model outputs a concrete entity.

Weak review signals such as gold actions with empty or unknown arguments while the model extracts an entity-like direct object should not populate `strong_dataset_issue`. They can be useful for manual review, but they are not strong enough to support dataset-error claims.

### Preposition-Object Label Direction

`extra_arguments:preposition_object` and `missing_arguments:preposition_object` are named from the gold annotation side:

- `extra_arguments:preposition_object`: gold has a prepositional-object argument that the model does not output. Example: text `put food into the red bowl`, gold `put(food, bowl)`, prediction `put(food)`. The dataset-side interpretation is that gold may have over-included a prepositional object.
- `missing_arguments:preposition_object`: the model outputs a prepositional-object argument that gold does not include. Example: text `put food into the red bowl`, gold `put(food)`, prediction `put(food, bowl)`. The dataset-side interpretation is that gold may have omitted a prepositional object.

This distinction matters because both labels are about inconsistent dataset treatment of prepositional objects, but they point in opposite annotation directions.

### Action-Diagnosis Precision

Action-level diagnostics can precisely describe evaluation outcomes, but most cannot precisely prove the underlying cause:

- `unmatched_gold_action` with `candidate_llm_issue=missing_actions` is precise as an evaluation outcome: a gold action was not matched by any prediction. It should be read as "the model missed this gold action under the current evaluator", not as proof that the text semantics are unambiguous.
- `unmatched_prediction` with `candidate_llm_issue=extra_actions` is relatively strong when the predicted verb does not appear in the source text; this is a good hallucination or text-external extra-action signal.
- `unmatched_prediction` with `candidate_dataset_issue=missing_actions` is weak under the current rule because it only checks whether the predicted verb lemma appears in `original_text`. It should remain a candidate review label unless tightened to action-local, eventive, argument-supported evidence.
- `wrong_action` is also a candidate review label. Lexical or argument overlap between an unmatched gold action and an unused prediction can indicate a wrong action head, but it does not reliably identify the true cause.

A future strong action-side dataset issue should be limited to a stricter `missing_action` rule: the unused predicted action should appear in the action-local sentence, use an eventive lexical verb rather than a control/light/auxiliary verb, have argument support in that sentence, and not duplicate or paraphrase an existing gold action.

## 2026-06-28 Optional And Exclusive Matching Update

The evaluator now treats optional and exclusive annotations as extraction-tolerance rules rather than ordinary required labels:

- Optional actions (`act_type == 2`) may be omitted without creating a false negative or an unmatched-gold diagnostic. If an optional action is predicted, it is counted as one gold denominator item and one true positive.
- Exclusive action groups (`act_type == 3` plus `related_acts`) count as one gold action for the whole group. A prediction is expected to match one member of the group. If the model predicts multiple valid alternatives from the same group, the first/best match contributes to action/object metrics and the additional group matches are neutral: they do not increase `total_tagged`, do not add true positives, and do not produce unmatched-prediction diagnostics.
- Exclusive object alternatives in `obj_idxs[1]` can satisfy the required object slot represented by `obj_idxs[0]`. When the required object is already matched, additional predicted exclusive alternatives are neutral and do not increase `obj_tagged`.
- Non-exclusive extra predicted actions or objects still count normally. The neutral handling applies only when the extra prediction matches the same exclusive action group or the action's exclusive object alternatives.

This keeps the F1 denominator aligned with the annotation intent: `A or B` expects one extracted action, while extracting both valid alternatives should not be punished or rewarded. The same policy applies to object alternatives such as required object slot 0 plus exclusive object alternatives in slot 1.

Implementation points:

- `evaluation.py` groups exclusive actions by `act_idx + related_acts`, picks the best unused prediction for the group, and consumes any additional matching group predictions as neutral.
- `src/evaluation_helpers.py::match_objs` still uses strict `argument_match_type(...)` matching, but ranks direct essential-object matches ahead of exclusive-object alternatives and subtracts unmatched exclusive alternatives from `obj_tagged`.
- `tests/test_evaluation.py` covers optional omission, one-of-many exclusive action extraction, multiple valid exclusive action predictions, and exclusive object alternatives.

Focused validation after this update:

```powershell
C:\Users\Apexmod\miniforge3\envs\llm\python.exe -m pytest tests\test_evaluation.py -q
```

Result:

```text
56 passed
```

Compile validation:

```powershell
C:\Users\Apexmod\miniforge3\envs\llm\python.exe -m compileall evaluation.py src\evaluation_helpers.py tests\test_evaluation.py
```

Result: compile succeeded.

## 2026-06-28 Validation

After the strong argument diagnostics update, the focused validation command was:

```powershell
C:\Users\Apexmod\miniforge3\envs\llm\python.exe -m pytest tests\test_evaluation.py tests\test_prompts.py -q
```

Result:

```text
55 passed
```

Compile validation was:

```powershell
C:\Users\Apexmod\miniforge3\envs\llm\python.exe -m compileall evaluation.py src\evaluation_helpers.py tests\test_evaluation.py
```

Result: compile succeeded.

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

## 2026-06-28 OpenAI Batch Experiment Workflow

OpenAI full-run experiments now have a separate Batch API workflow in `scripts/openai_batch_experiment.py`. The synchronous `experiment.py` path remains available for smoke tests and local/Ollama runs.

The Batch workflow supports one-call solvers:

- `nl2p_1`
- `nl2p_1_ablation`

It intentionally does not support `nl2p_2` or `nl2p_3` yet because those solvers have multi-step dependencies and would need step-wise batch collection.

The script uses one manifest per `(solver, model, run_id)` and allows all three EASDRL datasets in the same batch input when `-d` is omitted. Do not mix different solver/model combinations into one manifest; submit them as separate batch jobs.

Typical full-run commands:

```powershell
C:\Users\Apexmod\miniforge3\envs\llm\python.exe scripts\openai_batch_experiment.py prepare -s nl2p_1 -m gpt-5.4-mini --run-id full
C:\Users\Apexmod\miniforge3\envs\llm\python.exe scripts\openai_batch_experiment.py submit results\batch\nl2p_1\gpt-5.4-mini\full\manifest.json
C:\Users\Apexmod\miniforge3\envs\llm\python.exe scripts\openai_batch_experiment.py status results\batch\nl2p_1\gpt-5.4-mini\full\manifest.json
C:\Users\Apexmod\miniforge3\envs\llm\python.exe scripts\openai_batch_experiment.py collect results\batch\nl2p_1\gpt-5.4-mini\full\manifest.json
C:\Users\Apexmod\miniforge3\envs\llm\python.exe evaluation.py -d results\nl2p_1\gpt-5.4-mini --diagnostics
```

For `nl2p_1_ablation`, replace `nl2p_1` with `nl2p_1_ablation` in the commands.

Batch `prepare` writes OpenAI request JSONL to `results/batch/<solver>/<model>/<run-id>/input.jsonl`. `submit` uploads that file and stores `input_file_id` and `batch_id` in `manifest.json`. `status` refreshes status, output file id, error file id, and request counts. `collect` downloads or reads `output.jsonl`, parses each response by `custom_id`, and writes evaluator-compatible result files under `results/<solver>/<model>/`.

Important output contract:

- OpenAI raw batch output is `results/batch/.../output.jsonl` and is not read directly by `evaluation.py`.
- `collect` writes compact human-readable JSON files with only:
  - `dataset`
  - `doc_id`
  - `sentences`
  - `prediction`
  - `gold_actions`
- `collect` also writes `.pkl` files containing the original dataset samples with `pred` populated. Current `evaluation.py` reads these `.pkl` files, so compact JSON does not affect evaluation.

## 2026-06-28 Batch Smoke Test

A successful `test10` smoke run used:

```powershell
C:\Users\Apexmod\miniforge3\envs\llm\python.exe scripts\openai_batch_experiment.py prepare -s nl2p_1 -m gpt-5.4-mini -d cooking -l 10 --run-id test10
C:\Users\Apexmod\miniforge3\envs\llm\python.exe scripts\openai_batch_experiment.py submit results\batch\nl2p_1\gpt-5.4-mini\test10\manifest.json
C:\Users\Apexmod\miniforge3\envs\llm\python.exe scripts\openai_batch_experiment.py status results\batch\nl2p_1\gpt-5.4-mini\test10\manifest.json
C:\Users\Apexmod\miniforge3\envs\llm\python.exe scripts\openai_batch_experiment.py collect results\batch\nl2p_1\gpt-5.4-mini\test10\manifest.json
C:\Users\Apexmod\miniforge3\envs\llm\python.exe evaluation.py -d results\nl2p_1\gpt-5.4-mini --diagnostics
```

The Batch manifest reported 10 completed requests and 0 failed requests. The collected evaluation result for `cooking` first 10 documents was:

```text
Precision:        0.9682539682539683
Recall:           0.7991266375545851
F1:               0.875598086124402
Object Precision: 0.6308243727598566
Object Recall:    0.6929133858267716
Object F1:        0.6604127579737337
```

The diagnostics for this smoke test had 175 rows:

```text
argument_mismatch        107
unmatched_gold_action     41
wrong_action              21
unmatched_prediction       6
```

This smoke-test diagnosis suggested `nl2p_1_ablation` should focus on general improvements, not domain-specific rules:

- increase action recall for explicit embedded, passive, participial, conditional, and state-change events
- prefer concrete state-changing event heads over framing/control/light/aspectual verbs
- tighten argument boundaries to core affected entities
- resolve clear pronouns conservatively
- filter non-action background facts, warnings, explanations, preferences, and expected outcomes
- keep complete noun phrases together instead of imitating dataset head-word splits

## 2026-06-28 Model Selection Notes

For the main OpenAI comparison, prefer `gpt-5.4` plus `gpt-5.4-mini` over a wider set of five OpenAI models. This gives a clearer strong-vs-compact closed-source comparison without turning the work into an OpenAI model leaderboard.

Recommended main experiment matrix:

- `gpt-5.4` with `nl2p_1` and `nl2p_1_ablation`
- `gpt-5.4-mini` with `nl2p_1` and `nl2p_1_ablation`
- an open-source Llama model with both prompts
- an open-source Gemma model with both prompts

Use `gpt-5.5` only as an optional upper-bound or supplementary run. The estimated cost for `gpt-5.5` on all three datasets with both prompts was around the low tens of USD at standard pricing, and lower with Batch, but `gpt-5.4` plus `gpt-5.4-mini` provides more useful comparison coverage for similar or lower total cost.

## 2026-06-28 Latest `nl2p_1_ablation` Prompt Direction

The latest `nl2p_1_ablation` prompt remains general and avoids dataset-domain examples. It strengthens:

- sentence-local processing, with cross-sentence arguments only for clear pronoun or ellipsis references
- filtering of general background facts, preferences, warnings, explanations, and expected outcomes
- coverage of explicit passive, participial, embedded, conditional, and state-change events
- concrete event-head selection over framing/control/light/aspectual/causative verbs
- argument boundaries around core affected entities
- exclusion of time, duration, temperature, condition, purpose, manner, location, source, destination, container, and instrument phrases unless the entity itself is changed
- movement/placement handling: include the moved or changed entity, and include source/destination/container only when directly acted on or changed
- conservative pronoun resolution for `it`, `them`, `this`, `these`, `those`, and `everything`
- complete noun phrase preservation and coordinated-entity splitting only for separate affected objects

The baseline `nl2p_1` should remain weaker than `nl2p_1_ablation` so the experiment can show whether diagnostics-informed prompt refinement improves general action extraction behavior.

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

## 2026-06-28 Batch/Prompt Validation

Focused validation after the Batch script and latest ablation prompt updates:

```powershell
C:\Users\Apexmod\miniforge3\envs\llm\python.exe scripts\openai_batch_experiment.py --help
C:\Users\Apexmod\miniforge3\envs\llm\python.exe -m pytest tests\test_openai_batch_experiment.py tests\test_prompts.py -q
C:\Users\Apexmod\miniforge3\envs\llm\python.exe -m compileall src\llm\config.py scripts\openai_batch_experiment.py tests\test_prompts.py
```

Result:

```text
5 passed
compileall succeeded
```

Known pitfalls fixed during this work:

- `src.llm.task.task` imports `TASK_FUNCTIONS`; `src/llm/config.py` now defines `TASK_FUNCTIONS = {}` to keep the existing import chain working.
- `scripts/openai_batch_experiment.py` has a raw top-level docstring because Windows paths such as `C:\Users\...` otherwise trigger Python `unicodeescape` parsing errors.
