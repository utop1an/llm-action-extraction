# RISeC feasibility subset

Source: https://github.com/YiweiJiang2015/RISeC

Revision: `70b9a0b4c4eb656375a83bc1c0ca47c29ec2da2c`.
License: MIT (included). Original corpus: 260 recipes, split 126/56/78 documents.

This is the first 8 train, 2 dev and 2 test .txt files in lexicographic path order,
with corresponding BRAT .ann files. It is a small nonrepresentative smoke subset,
not a replacement benchmark. No synthetic labels or LLM-generated annotations.
Text files may have a final newline normalized by source retrieval; annotated
character spans are checked against their exact original surface text.

The converter preserves spans, semantic roles, explicit links, implicit links,
and annotation notes. The cEASDRL probe projects multiword action/argument spans
onto their first annotated token; this is a declared projection, not a syntactic
head detector. Implicit arguments are retained but excluded from extractive scoring.
There are no ES/OP/EX labels in RISeC; no such planning labels are invented.

Convert:

```powershell
python scripts/prepare_procedural_data.py --format risec --input data/procedural/risec_smoke/data --output data/procedural/risec_smoke/converted
```
