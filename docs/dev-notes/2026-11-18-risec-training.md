# RISeC binary action/argument training

Historical first-token experiment. The corrected complete-span protocol is in
[`2026-11-18-risec-spans-v2.md`](2026-11-18-risec-spans-v2.md).

Date: 2026-11-18 (Australia/Sydney).

## Correction: user-facing names versus token anchors

The user clarified that the desired task is full text -> action names with their
corresponding argument names. The earlier first-token projection is NOT a
complete-name extractor. In particular, projecting `a large bowl` to `a` can
score a correct anchor while producing an unusable argument name. Therefore the
97.23% action-anchor F1 and gold-action-conditioned argument dev scores must not
be reported as the requested complete-name extraction performance.

After access was restored, the saved files showed three completed argument
epochs and an interrupted fourth epoch; there was no completed argument run
receipt. The best available argument checkpoint is epoch 3. No retraining or
test-set tuning was performed for the following evaluation.

The existing action checkpoint (epoch 5) was chained with that argument
checkpoint. Argument input windows and masks use predicted action locations,
not gold action annotations. Results on all 78 test documents:

| Metric | Precision | Recall | F1 |
|---|---:|---:|---:|
| Action first-token anchors | 96.84% | 97.61% | 97.23% |
| Action-argument first-token pairs, end to end | 87.58% | 77.23% | 82.08% |
| Actual complete action names, strict | 88.50% | 89.20% | 88.85% |
| Actual complete linked argument names, strict | 28.21% | 24.88% | 26.44% |

Strict name evaluation normalizes case and whitespace only and distinguishes
occurrences by their source positions. An argument is correct only when both its
full name and its parent action name are correct. Missing actions retain their
argument links in the recall denominator. Predictions are not expanded using
gold spans. Implicit arguments remain excluded. These are diagnostic results for
the current anchor-trained checkpoints, not a completed full-name baseline.

Receipts: `results/ceasdrl_risec_20261118/pipeline_seed0_epoch3/receipt.json`
and `name_evaluation.json`. Reproducible strict-name scoring is implemented in
`scripts/ceasdrl/evaluate_risec_names.py`; its three tests cover partial names, incorrect
argument ownership, missed actions and repeated action names.

To meet the clarified task, future training must predict complete names/spans
(or use an explicitly agreed content-head normalization), instead of arbitrarily
treating the first annotated token as the argument name. The official document
split remains 126 train / 56 validation / 78 test.

## Scope and fixed protocol

The user requested cEASDRL training on RISeC and explicitly excluded ES/EX/OP.
This experiment treats every annotated action occurrence as a positive extraction
target. It trains action detection and explicit argument extraction separately.
It does not classify essential/optional/exclusive actions or semantic roles.

The source is YiweiJiang2015/RISeC revision
`70b9a0b4c4eb656375a83bc1c0ca47c29ec2da2c`. All 523 downloaded file contents match
their original Git blob hashes. The official document split is preserved:
126 train, 56 dev, 78 test. An upstream orphan annotation,
`data/train/165_Coq_au_Vin_alla_Italiana.ann`, has no matching training text and
is excluded; no labels have been invented or silently repaired.

The corpus has 3,061 actions, 5,520 explicit argument relations and 1,695 implicit
relations under the documented conversion. Original spans, roles and implicit
relations remain in canonical JSONL; training projects spans to their first
annotated token and excludes implicit arguments and condition-clause arguments.
This projection is not a syntactic-head detector or exact-span evaluation.

All 260 texts passed character-offset checks. The longest document is 545 tokens,
so the action model uses 600 positions instead of the older harness's 500.
Argument contexts are the same centered 100-token windows used by the previous
local harness. All annotated explicit arguments fit; no action/argument example
is discarded for an out-of-window argument.

## Implementation and verification

Original GloVe-100 and ELMo-small-768 contextual features are exported in the
existing Python 3.6 environment; 1,430 unique contexts took approximately 201 s.
Embeddings are fixed. No feature fitting uses dev or test labels.

To make full training practical on the RTX 4070, the original four-branch CNN
(2/3/4/5-gram, 32 filters, global max pool, Dense-256, two Q outputs) is implemented
in PyTorch. Repeated tag/distance channels are contracted algebraically; model
capacity is unchanged. Keras Conv2D initializer fan-in/fan-out conventions are
preserved. No dropout was added: the original class stores a dropout option but
does not use a dropout layer.

The actual pretrained Keras weights were loaded into both implementations.
Forward Q values, the original two-output MSE DQN loss, and all 12 parameter
gradients were compared. Relative loss errors were about 1e-7; maximum relative
gradient L2 error was 1.28e-6. All 48 binary-reward test cases passed. The report
is `results/ceasdrl_risec_20261118/parity/parity_report.json`.

This is an **adapted cEASDRL implementation**, not original paper-score
reproduction. It retains the original binary-label rewards, action-rate shaping,
terminal bonus, contextual embeddings, CNN, Q target and reward-sign replay
sampling. Recorded differences include an independent target network, a compact
replay buffer, vectorized episodes, official document splits, the span projection
and longer action capacity. Backend optimizer numerics are not claimed bit-identical.

## Training configuration

- From scratch, seed 0, maximum 5 complete epochs, patience 3 on dev F1.
- 16 concurrent document/action-context episodes; replay batch 32.
- Adam learning rate 0.001, epsilon 1e-8; gamma 0.9.
- One update every four vectorized environment steps; target sync every 100 updates.
- Replay capacity 50,000; floor(0.9 * 32) positive-reward samples per update.
- Exploration decreases from 1.0 to 0.1 over the first epoch's token count.
- Best checkpoint selected only by dev micro F1, then test evaluated.
- No paid API calls, no NL2P run, no tuning against test results.

Argument evaluation uses gold action locations, so it is explicitly not the
end-to-end pipeline score. Role-agnostic links and omitted arguments need separate
interpretation. A pipeline evaluation must propagate predicted actions and count
missed-action argument links as false negatives.

The pipeline implemented here uses only predicted action locations to construct
argument windows and exclusion masks. Its scoring denominator includes all gold
explicit links, including links attached to actions missed by the action model.
Complete-frame matching additionally requires the action location and its entire
set of explicit argument anchors to match exactly. No gold spans are used to
expand predictions at inference time.

## Label interpretation

First-token scores can penalize an otherwise related verb choice. For example,
the test recipe `191_Dads_Flat_Bread` annotates `let rise` as one action span;
its anchor is `let`, while predicting `rise` counts as one FP and one FN. Another
recipe describes `cabbage will cook down` without labelling that occurrence as an
action, while labelling the imperative `cook` later in the recipe. These are
examples of span and instruction/state distinctions, not automatic evidence that
the human annotation is wrong. The complete action error list is retained in
`act_seed0/error_examples.json`; no test labels were changed after seeing errors.

New GPU-trained checkpoints are also exported to the original Keras `.h5`
layout. `check_ceasdrl_trained_keras.py` loads them in the actual upstream Keras
network and checks real test-context Q outputs against the GPU checkpoint. This
checks artifact compatibility; it does not turn this adapted protocol into an
exact historical-paper reproduction.

## Commands

Run from the repository root:

```powershell
python scripts/ceasdrl/materialize_risec.py
python scripts/ceasdrl/prepare_procedural_data.py --format risec --input data/procedural/risec/upstream/data --output data/procedural/risec/converted
& C:\Users\Apexmod\miniforge3\envs\py36\python.exe -u scripts/ceasdrl/export_ceasdrl_features.py --data data/procedural/risec/converted --output data/procedural/risec/features_v1
& C:\Users\Apexmod\miniforge3\envs\py36\python.exe -u scripts/ceasdrl/export_ceasdrl_parity.py --output results/ceasdrl_risec_20261118/parity
python scripts/ceasdrl/check_ceasdrl_parity.py --fixtures results/ceasdrl_risec_20261118/parity
python -u scripts/ceasdrl/train_ceasdrl_risec.py --features data/procedural/risec/features_v1 --output results/ceasdrl_risec_new/act_seed0 --parity-report results/ceasdrl_risec_20261118/parity/parity_report.json --mode act --epochs 5 --patience 3
python -u scripts/ceasdrl/train_ceasdrl_risec.py --features data/procedural/risec/features_v1 --output results/ceasdrl_risec_new/arg_seed0 --parity-report results/ceasdrl_risec_20261118/parity/parity_report.json --mode arg --epochs 5 --patience 3
python -m pytest tests/test_procedural_data.py tests/test_ceasdrl_torch.py -q -p no:cacheprovider
```

Run outputs contain configuration, epoch history, best/last checkpoints, dataset
and feature hashes, source hashes, raw test predictions, and final receipts.

After both training jobs complete, evaluate the actual two-stage pipeline:

```powershell
& C:\Users\Apexmod\miniforge3\envs\py36\python.exe -u scripts/ceasdrl/prepare_ceasdrl_pipeline.py --data data/procedural/risec/converted/test.jsonl --action-predictions results/ceasdrl_risec_20261118/act_seed0/test_predictions.json --existing-features data/procedural/risec/features_v1 --output data/procedural/risec/pipeline_features_seed0
python scripts/ceasdrl/evaluate_ceasdrl_pipeline.py --data data/procedural/risec/converted/test.jsonl --features data/procedural/risec/pipeline_features_seed0 --argument-run results/ceasdrl_risec_20261118/arg_seed0 --output results/ceasdrl_risec_20261118/pipeline_seed0
python scripts/ceasdrl/summarize_risec_training.py
```

Use a fresh destination for evaluations and model exports. Export a selected
checkpoint for the original Keras network with:

```powershell
python scripts/ceasdrl/export_ceasdrl_checkpoint.py --run results/ceasdrl_risec_20261118/act_seed0 --features data/procedural/risec/features_v1
& C:\Users\Apexmod\miniforge3\envs\py36\python.exe -u scripts/ceasdrl/check_ceasdrl_trained_keras.py --run results/ceasdrl_risec_20261118/act_seed0
```

Replace `act_seed0` with `arg_seed0` for the argument model. The `.pt` and `.h5`
weights and regenerable embedding/fixture arrays exist locally but are excluded
from Git. Receipts, source code, canonical annotations and training histories
remain reviewable independently of those binary caches.
