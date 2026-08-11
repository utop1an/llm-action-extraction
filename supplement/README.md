# Code and Data Supplement

This anonymous archive contains the implementation, evaluation data, saved
model predictions, and scripts needed to reproduce the reported action and
argument extraction metrics.

## What is included

- `src/`: extraction methods, prompts, model adapters, and evaluation helpers.
- `evaluation.py`: action- and argument-level evaluation, including the
  conservative adjusted argument metrics.
- `experiment.py`: synchronous inference entry point.
- `scripts/openai_batch_experiment.py`: batch inference entry point.
- `scripts/rebuild_result_pickles.py`: reconstructs evaluator-ready PKL files
  from the compact JSON predictions included in this archive.
- `scripts/reproduce_supplement.py`: reproduces every included metric and
  checks it against `expected_metrics.csv`.
- `data/easdrl/`: the three labeled evaluation datasets used by the paper.
- `data/coref_llm/`: the resolved inputs used by the coreference condition.
- `results/`: saved JSON predictions for four methods, five models, and three
  evaluation domains.
- `tests/`: focused evaluator, batch, and model-adapter tests.
- `MANIFEST.sha256`: SHA-256 digest and byte size of every payload file.

Large mismatch-diagnostic CSV files and derived result PKL files are omitted
to stay within the submission limit. They can be regenerated locally.

## Environment

Python 3.12 is recommended. From the extracted archive root:

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies and English spaCy
model:

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Reproduce the reported metrics

Run the complete deterministic re-evaluation:

```bash
python scripts/reproduce_supplement.py
```

This command reconstructs the derived PKL files, evaluates all 20
method/model configurations over the three domains, and compares all 60 rows
with `expected_metrics.csv`. It exits with a non-zero status if any metric or
count differs.

For a quick installation check:

```bash
python scripts/reproduce_supplement.py --quick
```

To also regenerate the detailed mismatch diagnostics for a selected result
directory:

```bash
python scripts/rebuild_result_pickles.py --results-dir results/nl2p_1/gpt-5.4-mini
python evaluation.py -d results/nl2p_1/gpt-5.4-mini --diagnostics
```

## Re-run inference

Saved predictions are included so reviewers can reproduce evaluation without
paid APIs or local model servers. Fresh inference is optional and may vary
with model service updates.

Example synchronous smoke run:

```bash
python experiment.py -s nl2p_1 -m gpt-5.4-mini -d cooking -l 10
```

Example batch preparation:

```bash
python scripts/openai_batch_experiment.py prepare -s nl2p_1 -m gpt-5.4-mini -d cooking -l 10 --run-id smoke
```

No credentials are included. Fresh hosted-model inference requires the
reviewer to supply their own API credentials through environment variables.
Local-model inference requires a compatible Ollama installation and model.

## Anonymity and archive hygiene

The archive intentionally excludes version-control metadata, environment
files, credentials, IDE settings, caches, temporary files, machine-specific
paths, and submission-author metadata. `ANONYMITY_CHECK.txt` records the
automated checks performed when the archive was built.
