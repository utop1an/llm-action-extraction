# Development notes

## Current reproduction entry points

| Task | Entry point |
| --- | --- |
| Reproduce the main EASDRL benchmark | The [supplement guide](../../supplement/README.md) gives the offline evaluation commands and the scope of the quick and full checks. |
| Inspect the November prompt and example-selection experiments | The [project results index](../../README.md#start-with-the-results) links the predictions and score tables. The [report builder](../../scripts/historical/rebuild_20261122_reports.py) regenerates the tables from local batch receipts in a new output directory. |
| Reproduce the RISeC complete-span analysis | Use the [complete-span protocol](2026-11-18-risec-spans-v2.md) for the target definition, environment and commands. The [audit](../../results/ceasdrl_risec_20261118/spans_v2/audit_review/AUDIT.md) documents how the scoring policy affects the reported results. |
| Inspect the Naruto baseline | The [saved evaluation](../../results/naruto/output/evaluation_result.csv) covers Cooking, WikiHow and Win2K. To rerun it, use the [adapter](../../analysis/naruto/evaluation_naruto.py) with the [label CSV](../../analysis/naruto/extracted_labels.csv) and per-dataset prediction CSVs in `results/naruto/`, or supply `miglani_data/miglani_all_events.csv` for splitting. Run `python analysis/naruto/evaluation_naruto.py` from the repository root. It writes to `results/naruto/output/`. |
| Verify the integrated artifacts | Run `python -m src.result_paths` from the repository root. It resolves the original paths in the [November 23 SHA-256 manifest](2026-11-23-result-file-map.json) against the current layout, including local batch artifacts. |

## Historical records

- [June 27 · Evaluation diagnostics and prompt ablation](2026-06-27-diagnostics-and-prompt-ablation.md) records mismatch analysis, manual sampling, evaluator changes and early prompt-ablation work.
- [November 18 · Dataset selection and cEASDRL feasibility](2026-11-18-planning-datasets-ceasdrl.md) records the evidence used to select datasets and the limits of the feasibility runs.
- [November 18 · RISeC first-token training](2026-11-18-risec-training.md) records the earlier binary action/argument target. Use the complete-span protocol above for full action names and linked argument mentions.
