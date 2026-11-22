# LLM action and argument extraction

Given procedural text such as *Pick up the apple and place it in the basket*, the extraction target is an ordered sequence such as `pickup(apple)`, `place(apple, basket)`. The repository contains the extraction methods, saved predictions and offline evaluation code.

## Start with the results

[Supplementary experiments: NL2P prompt ablation and random two-shot comparisons](results/rebuttal_20261027/README.md)

| Experiment | Where to start | What it contains |
| --- | --- | --- |
| NL2P four-condition ablation, November 22 | [Predictions by run](results/nl2p_four_conditions_20261122/) · [Local summary](results/batch/four_arms_20261122/mean_sd_percent.csv) | Five repeats of the full, minimal, definitions-removed and rules-removed prompts on two models. |
| GPT3-to-Plan example selection, November 22 | [Predictions by run](results/gpt3toplan_trainpool_20261122/) · [GPT-5.4 scores](results/batch/gpt3toplan_trainpool_20261122/gpt-5.4_mean_sd.csv) · [Mini scores](results/batch/gpt3toplan_trainpool_20261122/gpt-5.4-mini_mean_sd.csv) | Ranked versus random two-shot selection from the training pool, five repeats each. |
| Main EASDRL benchmark | [Frozen results](supplement/expected_metrics.csv) | 4 methods × 5 models × 3 domains. Each row gives action F1, strict argument F1, adjusted argument F1 and match counts. |
| Random two-shot examples and minimal prompt | [Comparison](results/rebuttal_20261027/gpt-5.4-mini/comparison.json) · [Random-seed adjusted scores](results/rebuttal_20261027/gpt-5.4-mini/rand2_adjusted_20261028.json) | Fixed versus random demonstrations and the effect of shortening the extraction prompt. |
| Repeated full and minimal prompts | [GPT-5.4-mini](results/nl2p_repeats_20261027_v1/gpt-5.4-mini/comparison_adjusted.json) · [GPT-5.4 completed runs](results/nl2p_repeats_20261027_v1/repaired/gpt-5.4/comparison_adjusted.json) | Run-level results, including the repaired GPT-5.4 runs. The original partial runs have separate directories. |
| cEASDRL adaptation to RISeC | [Training and selection](results/ceasdrl_risec_20261118/spans_v2/SUMMARY.md) · [Audit](results/ceasdrl_risec_20261118/spans_v2/audit_review/AUDIT.md) | End-to-end extraction of full action names and linked argument spans. The audit records the split, scoring scope and limitations. |
| Short-story extraction | [Saved outputs](results/short_stories/) · [PDDL comparison script](scripts/compare_short_story_pddl.py) | Ordered event mentions and their correspondence with hand-authored domain actions. |

The October [random-seed predictions](results/rebuttal_20261027/random_two_shot/) and [full/minimal repeats](results/nl2p_repeats_20261027_v1/runs/) are grouped next to their comparisons. Folder names retain the original run IDs, including the separately collected repairs. Each run contains one directory per model.

November summary tables and raw batch receipts live in the local `results/batch/` workspace. To regenerate the November summary CSVs from the saved scores and predictions, run `python scripts/historical/rebuild_20261122_reports.py NEW_OUTPUT_DIR`. This reads the original reporting scripts and translates the receipt paths in a temporary copy.

For the main benchmark, the methods are GPT3-to-Plan, NL2P-1, prompt ablation and coreference-resolved NL2P-1. The domains are Cooking, WikiHow and Win2K. Per-document JSON predictions and per-model CSV scores live in `results/<method>/<model>/`. The main evaluator scores arguments on matched actions. The RISeC experiment uses its own linked-span metric.

## Reproduce the main benchmark

Use Python 3.12. From the repository root, install the evaluation dependencies and the English spaCy model, then verify one saved result.

```sh
python -m pip install -r supplement/requirements.txt
python -m spacy download en_core_web_sm
python scripts/build_code_data_supplement.py --force
python dist/code_data_supplement/scripts/reproduce_supplement.py --quick --clean
```

The builder compares all 60 saved CSV rows with the frozen table and creates `dist/code_and_data_supplement.zip`. To recompute all 60 rows from saved predictions, omit `--quick`. Both commands run offline after dependency installation. The [paper supplement guide](supplement/README.md) also covers fresh inference and detailed diagnostics.

To regenerate the mismatch report for one condition in the working tree, run:

```sh
python evaluation.py -d results/nl2p_1/gpt-5.4-mini --diagnostics
```

## Analysis notebooks and adapters

| Purpose | Entry point | Inputs and outputs |
| --- | --- | --- |
| Benchmark diagnostics (11 notebooks) | [Model comparisons](analysis/diagnostics/all_models_comparison.ipynb) · [Mismatch overview](analysis/diagnostics/diagnostics_overview.ipynb) | Saved evaluation and mismatch CSVs in `results/`. |
| Manual annotations (2) | [Read annotated samples](analysis/annotations/read_sample_results.ipynb) | The two [sample CSVs](analysis/annotations/samples/) accompany the notebooks. |
| Short stories (3) | [PDDL comparison](analysis/short_stories/short_story_pddl_comparison.ipynb) · [Export comparisons](analysis/short_stories/read_short_story_results_into_csv.ipynb) | Saved story predictions in `results/short_stories/` and [comparison CSVs](analysis/short_stories/output/). |
| Earlier examples (7) | [Examples and accompanying files](analysis/examples/) | Includes a transport text fixture and the action/argument F1 workbook. Some examples require historical data files or a configured model. |

The [Naruto evaluator](analysis/naruto/evaluation_naruto.py) aligns prediction CSVs with the [saved labels](analysis/naruto/extracted_labels.csv). Read the [saved scores](results/naruto/output/evaluation_result.csv) or see the [development notes index](docs/dev-notes/README.md) for rerun inputs.

Launch Jupyter inside the repository and open the desired notebook. The Codex short-story notebook and several early examples run inference. Start with the comparison notebooks to inspect existing results.

For serial evaluation from the repository root, run `bash hpc/evaluation.sh`. It reads `PREFIX`, `SOLVERS`, `MODELS`, `DIAGNOSTICS` and `PYTHON`, skips missing result directories and stops on the first evaluator failure. Its legacy default includes the Llama 3B directory when present. For parallel evaluation and dry runs, use `python scripts/batch_evaluate.py --help`.

## Work on the project

- [experiment.py](experiment.py) runs synchronous inference. [openai_batch_experiment.py](scripts/openai_batch_experiment.py) prepares, submits and collects batch jobs. Keep new inference outputs separate from the frozen benchmark.
- [src/solvers](src/solvers/) contains extraction methods. [src/llm](src/llm/) contains prompt templates and model adapters. [evaluation.py](evaluation.py) and [evaluation_helpers.py](src/evaluation_helpers.py) implement the main metrics.
- [RISeC instructions](docs/dev-notes/2026-11-18-risec-spans-v2.md) cover the PyTorch models, checkpoint selection and the separate legacy GloVe/ELMo environment. Historical training receipts record the paths used on the original machine. The [source relocation hashes](scripts/ceasdrl/source_relocations.json) identify the three receipt-bound scripts edited for the directory move; the audit accepts the recorded version only when the relocated source also matches its pinned digest.
- [cEASDRL scripts](scripts/ceasdrl/) cover dataset preparation, training, prediction and independent scoring for RISeC and RiCoRecA. Begin with the [complete-span workflow](docs/dev-notes/2026-11-18-risec-spans-v2.md) or the [RiCoRecA result and evaluation scope](results/ceasdrl_ricoreca_64_16_20_seed0/REPORT.md).
- [Historical rebuttal scripts](scripts/historical/rebuttal_20261027/) record preparation, submission, recovery and scoring for the October campaign. The [RiCoRecA finalizer](scripts/historical/ricoreca_20261122/finish_ceasdrl_ricoreca.py) coordinated the original training jobs. Cluster job definitions remain under [hpc](hpc/). The October campaign scripts use original batch receipts, source snapshots and local paths for historical integrity checks. Use the saved comparisons above to inspect those results on another machine.

Run `python -m src.result_paths` to verify the original [integration SHA-256 manifest](docs/dev-notes/2026-11-23-result-file-map.json) against the current layout, including local-only files.

## Historical files

The earlier IJCAI results, the raw GPT3-to-Plan pilot and the GOOSE planning corpus were archived at their original relative paths in the local `results/archive/historical-experiments.zip`. The same tracked files can be retrieved from Git revision `d7e5b95`. The original main diagnostics and the mixed coref directory's pre-cleanup records are in `results/archive/main-diagnostics-before-cleanup.zip`. Extract individual entries when needed. Local drafts, raw batch workspaces, caches and private environment files are excluded from the collaboration package.
