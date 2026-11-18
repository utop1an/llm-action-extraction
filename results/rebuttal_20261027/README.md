# Supplementary action-extraction experiments

The project extracts action names and their arguments from procedural text. The October 2026 experiments test two choices in the original benchmark: which instructions NL2P needs and how GPT3-to-Plan performs with randomly selected demonstrations. All runs cover the three benchmark datasets: cooking (CT, 116 documents), wikihow (WHG, 150), and win2k (WHS, 154).

| Experiment | Design | Results |
| --- | --- | --- |
| NL2P prompt ablation | GPT-5.4 and GPT-5.4-mini, three fresh runs per prompt per model. Minimal retains the task statement and JSON contract and removes the definitions and extraction rules. | [Per-run scores](per_run_metrics.csv), [means and sample SDs](mean_and_sd.csv) |
| Random two-shot GPT3-to-Plan (Rand2) | GPT-5.4-mini, five seeds. Each seed permutes documents within each domain. Each query takes the first two eligible examples after excluding itself. | Same two CSVs, rows labelled `rand2` |
| Historical mini controls | Single saved runs of ranked two-shot, full NL2P and minimal NL2P, alongside the five Rand2 seeds. | [Strict metrics](gpt-5.4-mini/metrics.csv) |

Each per-run row is a corpus-level score for one domain. P, R and F1 are fractions in the CSVs. Strict argument scores evaluate arguments on evaluator-matched actions. Adjusted argument scores apply the original evaluator's prediction-dependent annotation diagnostics. The SD columns use the sample standard deviation over three runs or five seeds.

### What changed with the full NL2P prompt?

Mean paired differences (Full minus Minimal), in percentage points:

| Model | Domain | Strict action F1 | Strict argument F1 |
| --- | --- | ---: | ---: |
| GPT-5.4 | WHS | −0.31 | +0.14 |
| GPT-5.4 | CT | +0.32 | +11.47 |
| GPT-5.4 | WHG | −1.57 | +8.61 |
| GPT-5.4-mini | WHS | −0.21 | +3.19 |
| GPT-5.4-mini | CT | +0.65 | +9.30 |
| GPT-5.4-mini | WHG | −1.89 | +4.66 |

Full NL2P improved strict argument F1 in every paired run on CT and WHG. The GPT-5.4 WHS argument difference changed sign across runs. For GPT-5.4-mini, mean strict action F1 for Full NL2P versus Rand2 was 85.17 versus 81.89 on CT, 77.73 versus 70.37 on WHG, and 92.08 versus 92.13 on WHS.

### Coverage and code

Each model contributed 2,520 predictions to the repeated prompt ablation. The final GPT-5.4 set retains the original 1,951 successes and the first successful retry for each of 569 quota failures. Its final coverage is 2,520/2,520 with zero parse failures. The five-seed mini comparison covers another 2,100 predictions.

- [Prompt templates](../../src/llm/config.py) and [demonstration selection](../../src/solvers/gpt3_to_plan.py)
- [Batch preparation](../../scripts/openai_batch_experiment.py), [repeat runner](../../scripts/run_nl2p_repeats_batch.py), and [failed-request recovery](../../scripts/repair_nl2p_repeats_batch.py)
- [Control scoring](../../scripts/evaluate_rebuttal_batch.py), [repeat scoring](../../scripts/evaluate_nl2p_repeats.py), and [case analysis](../../scripts/analyze_minimal_ablation.py)
- [Open-weight HPC runner](../../scripts/run_hpc_rebuttal.py) and [Slurm jobs](../../hpc). The local Gemma preflight checked 36 baseline metrics. The tables above report the completed hosted-model experiments.

The historical batch scripts expect the original campaign manifests and saved predictions. The CSVs provide the complete per-domain metrics for the comparisons above and can be read without these artifacts or model API access.

### Saved predictions and evaluation reports

The [archives](archives) preserve the original relative paths and file bytes. Extract the archives you need from the repository root. Each prediction group contains document-level JSON, its evaluator-ready PKL and per-domain metadata.

| Archive prefix | Contents |
| --- | --- |
| `repeats_final` | Three Full and three Minimal runs for each model. Use these for the reported prompt ablation. |
| `repeats_before_repair` | Original GPT-5.4 outputs, including the 569 quota failures. These overlap with the final runs. |
| `rand2` | Saved random two-shot runs. The complete five-seed comparison is for mini. The GPT-5.4 archive contains the available outputs for seeds 0–2. |
| `minimal_control` | Historical single-run minimal controls, separate from the three fresh repeats. |
| `minimal_case_analysis` | Complete case-level analysis for the historical mini control. |

For example, to replay the strict scores of the first GPT-5.4 Full run on Cooking with the project's evaluation dependencies installed:

```sh
python -m tarfile -e results/rebuttal_20261027/archives/repeats_final_gpt-5.4.tar.xz .
python -c "from pathlib import Path; from scripts.evaluate_nl2p_repeats import score; s='nl2p_full_repeat1_20261027_v1_server_repaired'; m='gpt-5.4'; p=Path('results')/s/m/f'cooking_{s}_{m}'; r,_=score(p,'cooking',s,m,116); print(r['action_f1'],r['argument_f1'])"
```

Expected action and argument F1 are 0.8649345072 and 0.6934724092. Historical JSON reports retain the original machine paths and hashes. When locating files in another checkout, use the path starting at `results/`.

- [GPT-5.4 final report and repair audit](../nl2p_repeats_20261027_v1/repaired/gpt-5.4/completion_summary_20261028.json)
- [GPT-5.4-mini repeat report](../nl2p_repeats_20261027_v1/gpt-5.4-mini/comparison_adjusted.json)
- [Mini five-seed report](gpt-5.4-mini/rand2_adjusted_20261028.json) and [selected cases](gpt-5.4-mini/minimal_selected_cases.json)
