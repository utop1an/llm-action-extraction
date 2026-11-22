# OpenPI-A release snapshot

Source: https://github.com/brandeis-llc/poem-OPIA

Revision inspected: `0c1bc8a184d02b42dd8d1a4db92191232b475431`.
Paper: https://aclanthology.org/2026.lrec-1.939/
License: MIT (included; source text ultimately comes from OpenPI/WikiHow).

The official silver release contains 515 selected trigger spans from 515 steps
in 327 source documents. It is not an exhaustive action list: for example the
released span for "Allow the ink to dry." is "dry". This task must not be equated
with selecting every executable planning action or extracting all action arguments.

The separate human validation inputs link entities and before/after changes to
trigger candidates. Per-annotator responses are included unchanged; disagreements
are not automatically promoted to a gold standard. No complete PDDL action model,
ES/OP/EX labels, or argument-position links are supplied by this snapshot.

The converter creates a custom 80/10/10 grouped split by original document, merging
groups that share the same normalized sentence before splitting. This is not an
official split. Use it only for a transparently labelled silver-data pilot until
the target annotation scope and independently adjudicated evaluation are agreed.

```powershell
python scripts/prepare_procedural_data.py --format openpi-a --input data/procedural/openpi_a/data/silver_standard.jsonl --output data/procedural/openpi_a/converted
```
