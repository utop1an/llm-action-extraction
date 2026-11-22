# Planning-oriented action datasets and cEASDRL feasibility

Date: 2026-11-18 (Australia/Sydney). Scope: current code, primary-source literature,
public release inspection, and local execution. This is not a completed paper-score
reproduction or a finished cEASDRL-vs-NL2P benchmark.

## Decision

There is no verified drop-in modern dataset preserving the current combination of
text-token action heads, argument heads, essential/optional/exclusive annotations,
and executable planning semantics. Dataset age alone does not imply better labels.

Recommended experimental design:

1. **Direct extraction comparison:** ReciFineGold (2026) for chef-action spans;
   RISeC for action-to-argument links and explicit/implicit argument distinctions.
   RISeC is from 2020, so it is a schema-quality option, not a claim of novelty.
2. **Planning relevance:** GOLD (2025) / Proc2PDDL (2024) for reusable action schemas,
   parameters, preconditions/effects and downstream planning. A shared text-to-span
   annotation layer is needed before cEASDRL can be compared on these inputs.
3. **Control flow extension:** PAGED (2024) if the desired contribution is conditions,
   branching, ordering and parallel execution. Do not turn XOR gateways into the
   old EX labels mechanically: branches can each contain multiple actions.
4. **Recent action-state pilot:** OpenPI-A (2026), explicitly labelled silver and
   incomplete in action coverage. Suitable for selected state-changing trigger
   detection; not a clean replacement for exhaustive planning-action extraction.

## What the current repository actually measures

- `experiment.py` hard-codes cooking (116), wikihow (150), win2k (154).
- Gold data uses `words`, `sents`, `acts`, `act_idx`, `obj_idxs`, `act_type`, and
  `related_acts`; action/argument references are token positions.
- `evaluation.py` has special ES/OP/EX accounting: omitting an OP action is neutral;
  one EX group is one gold unit. Ordinary BIO or action-schema annotations do not
  supply these labels.
- `README.md` already notes inconsistent state/action distinctions and argument
  omissions. A newly published dataset still needs a sample-level annotation audit.
- `data/ceasdrl/weights` contains pretrained `.h5` files; no training code was
  present in this repository at the start of this task.
- The current LLM output is surface verb/argument text. For the new experiments,
  preserve character spans, multiword units and entity identity rather than relying
  exclusively on lemma matching. Report action-span F1, argument-link F1, and
  end-to-end action+argument F1 separately. Gold-action-conditioned argument F1 is
  not end-to-end performance.

## Dataset comparison

| Dataset | Released annotation / scale | Fit for this project | Main qualification |
|---|---|---|---|
| ReciFineGold, 2026 | 500 manually annotated recipes; character spans for 10 types, including chef/food/tool actions and states | Good new action-detection gold candidate | No action-to-argument links or ES/OP/EX; Ac2 discontinuous components need an explicit grouping policy |
| GOLD / DIGGER, Findings EMNLP 2025 | Paper: 103 domains, 1,100 tasks, 1,086 actions | Strongest candidate for formal planning action models | Schemas are not token occurrences; mixed source datasets, and current repository contains newer generation/revision scripts |
| Proc2PDDL, ACL workshop 2024 | 27 domains; article, action descriptions, step mapping, domain and problem PDDL | Compact downstream planning benchmark | Too small as sole supervised extraction corpus; step-to-action map is not a token-span map |
| PAGED, ACL 2024 | 3,394 document–graph pairs with actions, actors, constraints, gateways and flows | Good for conditional/branching procedural structure | Documents generated from graphs and paraphrased/refined, not wholly independent human span annotations; not PDDL |
| OpenPI-A, LREC 2026 | Released silver file: 515 steps from 327 original documents across 5 topics | Modern bridge from language triggers to state changes | One selected trigger per step; partial, silver labels. Human responses must be adjudicated before being called gold |
| RISeC, AACL 2020 | 260 recipes; action/argument spans, semantic links and zero anaphora | Closest readily usable action–argument supervision | Not recent, cooking only, no planning ES/OP/EX or full PDDL |
| PET, 2022 | Business-process activities, actors, gateways and flow | Small cross-domain process extraction supplement | Small corpus; activities/roles are not automatically grounded planning operators |
| OpenPI2.0, EACL 2024 | Canonicalized entities/attributes, before/after state changes and salience | Optional state/effect evaluation | Does not directly replace action-name/argument annotations |

ReciFine's **2.2 million** recipes are the separate automatically labelled silver
resource. Do not confuse them with the **500-document ReciFineGold** release or
with ReciFineGen, which evaluates generated recipes. The public ReciFineGold
viewer and author documentation were inspected; its complete JSONL was not
downloaded in this run. Its adapter is covered by synthetic format/integrity tests,
not a claim to have validated all 500 released annotations.

GOLD includes Proc2PDDL and OpenPI-derived data. They cannot be treated as fully
independent train/test benchmarks without overlap removal. The currently published
DIGGER dataset README mentions revised PDDL and a GPT-5.4 generation pipeline;
freeze a release/commit and distinguish original-paper data from later expansions.

OpenPI-A illustrates a practical mismatch: `Allow the ink to dry.` is labelled
`dry`, and `Re-fold your bandanna and sew the sides shut.` supplies `fold`.
Neither is a complete gold sequence of all executable actions. Its release can
support a clearly stated selected-trigger task, but treating every other verb as
a confidently annotated negative would misrepresent exhaustive action extraction.

## Local cEASDRL verification

Source: `Shivam-Miglani/contextual_drl`, commit
`b6b2acb8e5cc8009adb2dacfb363e0157a08f922`, Apache-2.0.
Pinned source is in `third_party/contextual_drl/`; provenance and compatibility
changes are listed in `UPSTREAM.md`.

The current project requires Python >=3.12, but the original cEASDRL stack targets
Python 3.6. An existing local environment was usable:

- `C:\Users\Apexmod\miniforge3\envs\py36\python.exe`
- Python 3.6.13, TensorFlow 1.14.0, Keras 2.3.1, PyTorch 1.5.0+cpu.
- Actual cached GloVe-100 + ELMo-small-768 embeddings (868 dimensions).
- Local RTX 4070 detected, but these legacy verification runs deliberately used
  CPU; GPU training has not been validated.

The first attempts blocked during an unused matplotlib/IPython GUI initialization.
Moving plotting imports inside the plotting function resolved the block. Removing
an unused ipdb import avoids an unrelated readonly IPython history write at exit.
No installed Python environment or existing experiment files were modified.

Important upstream issues found in code:

1. `EADQN.py` builds online and target `Model` objects from the same tensors. The
   two objects share layer weights. This was also observed at runtime. The harness
   defaults to an independent clone; `--target upstream-shared` retains the old
   behavior for an explicitly labelled diagnostic. Do not silently claim the fix
   is an exact original implementation.
2. `cEnvironment.py` splits `arg_sents` after expanding documents into per-action
   argument instances. Related contexts from the same document can cross splits.
   The new adapter splits documents before expansion.
3. Default replay storage copies 50,000 embedding matrices. Action BERT buffers
   alone require about **147.8 GiB**; ELMo action buffers about **40.5 GiB**; ELMo
   argument buffers about **16.2 GiB**, before other objects. Compact replay stores
   document references plus tag vectors instead.
4. Argument reward code has `action == 2` branches although actions are 0/1.
   The tested datasets do not supply exclusive argument groups, so that path was
   not exercised. It must be tested/repaired before a historical full reproduction.
5. `cmain.py` does not define `gui_mode2` even though the environment reads it.
   The explicit harness avoids the brittle original constructor.

These changes establish an **adapted cEASDRL baseline**, not full published-score
reproduction. The original CNN, Q update, embeddings and environment reward
methods remain in use. Full legacy reproduction still requires original split/
tokenization/embedding parity and original metric verification.

### RISeC smoke runs completed

Official split preserved for a deterministic subset: 8 train / 2 dev / 2 test
documents, 171 actions, 305 explicit argument links, 89 implicit links.
Implicit arguments remain in canonical data but are excluded from extractive
argument scoring. Multitoken spans are projected to their first annotated token,
not a claimed syntactic head. Full spans remain available for later evaluation.

| Run | Training steps | Optimizer updates | Weight load | Changed weights + checkpoint round trip | Test F1 |
|---|---:|---:|---|---|---:|
| Action, ELMo | 128 | 124 | Existing cooking action ELMo weights loaded | Passed | 0.000 |
| Explicit arguments, ELMo | 128 | 124 | Existing cooking argument ELMo weights loaded | Passed | 0.000 |

The pretrained weights were loaded for compatibility, then reset before new-data
training. These are **from-scratch training-path checks**, not fine-tuning of the
old weights. Both short runs predict no positives on their held-out subset. This
does not reproduce paper performance and cannot support a model-quality claim.
Runtime was approximately 92 s and 104 s, including embeddings and evaluation.
Receipts contain dataset hashes, split IDs, losses, skipped examples and metrics.

An additional OpenPI-A pilot uses all 515 released steps for a custom grouped split:
412 train / 51 dev / 52 test instances. Original document IDs and identical text
are grouped before splitting. The fixed 512-step budget produced 508 optimizer
updates; this does not mean every training instance was visited. Runtime was
264.5 seconds on CPU, including embeddings and evaluation.

| OpenPI-A selected-trigger pilot | Dev F1 | Test precision | Test recall | Test F1 |
|---|---:|---:|---:|---:|
| Adapted cEASDRL, seed 0 | 0.7059 | 0.5769 | 0.5769 | 0.5769 |
| Train-only lexical baseline | Not used for selection | 0.5682 | 0.4808 | 0.5208 |

Losses were finite, weights changed, and checkpoint reload preserved predictions.
The test set contains 52 annotated triggers: cEASDRL matched 30 of 52 predictions;
the lexical baseline matched 25 of 44 predictions. These are first-token F1
scores against selected silver triggers, not exact-span scores or exhaustive
planning-action evaluation. One short run on 52 test steps cannot establish a
reliable superiority claim. No NL2P/LLM comparison has yet been run on this split.

The released human-validation files were also inspected without adjudication.
They contain 452 entity/state-change units; an annotator's `confirm` response can
retain an edited trigger, so it must not be counted as acceptance of the original
silver label. See `data/procedural/openpi_a/release_audit.json` for the counts.

## Reproduce the local checks

Run from the repository root. Use a fresh output directory for each run:

```powershell
python scripts/ceasdrl/prepare_procedural_data.py --format risec --input data/procedural/risec_smoke/data --output data/procedural/risec_smoke/converted
& C:\Users\Apexmod\miniforge3\envs\py36\python.exe -u scripts/ceasdrl/ceasdrl_local.py --data data/procedural/risec_smoke/converted --output results/ceasdrl_new_run/act --mode act --max-steps 128 --check-pretrained
& C:\Users\Apexmod\miniforge3\envs\py36\python.exe -u scripts/ceasdrl/ceasdrl_local.py --data data/procedural/risec_smoke/converted --output results/ceasdrl_new_run/arg --mode arg --max-steps 128 --check-pretrained
python -m pytest tests/test_procedural_data.py -q -p no:cacheprovider
```

To reproduce the OpenPI-A pilot with the downloaded release:

```powershell
python scripts/ceasdrl/prepare_procedural_data.py --format openpi-a --input data/procedural/openpi_a/data/silver_standard.jsonl --output data/procedural/openpi_a/converted
python scripts/ceasdrl/audit_openpi_a.py
& C:\Users\Apexmod\miniforge3\envs\py36\python.exe -u scripts/ceasdrl/ceasdrl_local.py --data data/procedural/openpi_a/converted --output results/ceasdrl_new_run/openpi_a_act --mode act --max-steps 512 --check-pretrained
```

After independently downloading the official `recifinegold.jsonl`:

```powershell
python scripts/ceasdrl/prepare_procedural_data.py --format recifinegold --input data/procedural/recifinegold.jsonl --output data/procedural/recifinegold_converted
& C:\Users\Apexmod\miniforge3\envs\py36\python.exe -u scripts/ceasdrl/ceasdrl_local.py --data data/procedural/recifinegold_converted --output results/recifinegold_pilot --mode act --epochs 20 --max-steps 0
```

ReciFineGold argument mode deliberately fails: no argument links are annotated.
The 20-epoch command is a starting experiment configuration, not an established
convergence setting or a completed run.

Before paper comparisons: freeze annotations and document/domain splits; inspect
50–100 examples for state/action distinctions, negation, optionality and omitted
arguments; tune only on dev; run at least three seeds; compare cEASDRL, a modern
supervised token/link baseline, and NL2P under the same labels and test split.
Use separate planning executability/goal-success evaluation for GOLD/Proc2PDDL.
No paid LLM calls or full multi-seed experiments were launched in this task.

## Primary sources

- ReciFine paper: https://aclanthology.org/2026.findings-eacl.210/
- ReciFineGold release: https://huggingface.co/datasets/nuhuibrahim/recifinegold
- Author schema: https://github.com/nuhu-ibrahim/ReciFine/blob/main/DATASETS.md
- GOLD paper: https://aclanthology.org/2025.findings-emnlp.138/
- GOLD current data organization: https://github.com/fip-lab/DIGGER/tree/main/DIGGER_demo/train_comet/dataset
- Proc2PDDL paper: https://aclanthology.org/2024.nlrse-1.2/
- Proc2PDDL release: https://github.com/zharry29/proc2pddl
- PAGED paper: https://aclanthology.org/2024.acl-long.583/
- PAGED release: https://github.com/SCUNLP/PAGED
- OpenPI-A paper: https://aclanthology.org/2026.lrec-1.939/
- OpenPI-A release: https://github.com/brandeis-llc/poem-OPIA
- RISeC paper: https://aclanthology.org/2020.aacl-main.82/
- RISeC release: https://github.com/YiweiJiang2015/RISeC
- PET: https://pdi.fbk.eu/pet-dataset/
- OpenPI2.0: https://aclanthology.org/2024.eacl-long.10/
- cEASDRL paper: https://pure.tudelft.nl/ws/portalfiles/portal/85314362/w42.pdf
- cEASDRL source: https://github.com/Shivam-Miglani/contextual_drl
