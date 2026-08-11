# Contents and Reproducibility Scope

## Experimental matrix

The included predictions cover:

- Methods: `gpt3_to_plan`, `nl2p_1`, `nl2p_1_ablation`, `nl2p_1_coref`
- Models: `gemma3-12b`, `gemma3-27b`, `llama3-70b`, `gpt-5.4`,
  `gpt-5.4-mini`
- Domains: `cooking`, `wikihow`, `win2k`

This yields 60 expected metric rows in `expected_metrics.csv`.

## Result representation

Each result JSON contains only the dataset identifier, integer document ID,
input sentences, parsed model prediction, and gold action records. These files
are human-readable and do not contain API responses, request identifiers, or
credentials.

The evaluator historically reads PKL files. To avoid duplicating the labeled
data in every method/model directory, `scripts/rebuild_result_pickles.py`
combines each compact prediction JSON with the corresponding labeled dataset.
For the coreference condition it also restores the exact resolved input text
from `data/coref_llm/`.

## Metrics

`evaluation.py` computes:

- action precision, recall, and F1;
- strict object/argument precision, recall, and F1;
- conservative adjusted object/argument precision, recall, and F1; and
- counts of perfect and mismatched action-argument matches.

The adjusted metrics only discount high-confidence annotation inconsistencies
recognized by the evaluator. Candidate diagnostics are not automatically
treated as annotation errors.

## Excluded material

The following material is intentionally not included:

- `.git`, local environment files, credentials, IDE settings, and caches;
- external repository links for the submitted code or data;
- large derived mismatch-diagnostic CSV files;
- duplicated result PKL files that can be rebuilt from included inputs;
- unrelated datasets, exploratory notebooks, figures, temporary renders, and
  cluster-specific job scripts.
