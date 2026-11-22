# RISeC complete-name extraction with adapted cEASDRL

This experiment repairs the earlier first-token training target. Its task is:
full procedural text -> complete action mentions and their linked complete
argument mentions. It excludes ES/EX/OP and semantic-role classification.
It is an adaptation of cEASDRL, not reproduction of the paper's reported scores.

**Audit correction (2026-11-18):** the frozen target set includes 45 `TODO`
relations (29 train / 6 dev / 10 test), whose role is not defined in the paper.
The reported 81.82% is reproducible under that policy. Excluding the 10 test
relations with predictions fixed gives 81.71%; this is a reference-policy
sensitivity check, not retraining on cleaned labels. The old count of 1,695
"implicit" links also includes temporal and other action-to-action relations.
See `results/ceasdrl_risec_20261118/spans_v2/audit_review/AUDIT.md` for details.

## Data and annotation coverage

Use the existing pinned RISeC source and official document split: **126 train,
56 dev, 78 test**. Seed 0 affects model training, not the data split. Documents
are never split into train/test sentences or action examples independently.

Canonical data retain 3,061 action mentions and 5,520 legacy extraction links,
including the unresolved `TODO` links noted above. All retained spans round-trip through BIO with
no loss. There are 110 adjacent argument-span pairs and one adjacent action-span
pair; binary inside/outside tags cannot distinguish their boundaries. There are
no discontinuous or overlapping spans within these extraction targets.

Implicit links and condition-clause links remain outside the evaluated task,
as in the preceding conversion. The orphan upstream training annotation with
no matching text remains excluded. This repair fixes our training projection;
it does not claim to manually correct all RISeC annotation errors.

The action context is the full document padded to 600 tokens (corpus max 545).
Each argument context is a deterministic 100-token window around the action's
first token. **Every complete explicit argument fits**, including its final
token. The first action token is only an input location cue, not the output
name. Any target outside the window raises an error rather than being dropped.

## Model and fixed training protocol

The verified CNN, frozen GloVe-100 + ELMo-small-768 embeddings,
repeated distance/tag features, DQN Bellman targets, and independent target
network are retained. Final balanced replay is described below. The output has three decisions:
`O`, `B`, `I`. This is a necessary change from the original binary extractor.
State tags are 0 for unvisited, and 1/2/3 for visited O/B/I decisions.

Rewards retain the earlier magnitudes and shaping: correct O gives 50, correct
B/I gives 100, incorrect choices negate the gold class's reward. B versus I
must be correct. Initial candidates counted non-O tokens for rate shaping;
the selected balanced candidates count decoded mention starts. The terminal
bonus is unchanged. The MSE is over three Q outputs rather than two. The binary
Keras/GPU numerical parity check supports the reused core, not an assertion
that BIO extension exists in the original paper.

Both stages start from scratch, seed 0, maximum 10 epochs, patience 3. Adam
learning rate 0.001, epsilon 1 -> 0.1 over the first training epoch, gamma 0.9,
16 simultaneous environments, replay batch 32, one optimization per four
vector steps, and target synchronization every 100 optimization steps.
Best checkpoints are selected **only by exact-span micro F1 on dev**. Argument
training and checkpoint selection condition on annotated actions; the final
reported pipeline conditions on predicted actions.

Checkpoints now include optimizer, target network, replay, counters and RNG
states for resuming from the last completed epoch. Previous first-token runs
are preserved separately.

Development-only action-stage comparison: the initial learning-rate 0.001 run
showed dev F1 0.7808, 0.8411, 0.8473, 0.8231, 0.7838, 0.8532 in epochs 1-6.
One additional action run was started with learning rate 0.0003 based on dev
diagnosis, using the same seed, 10-epoch cap and patience 3.
Select the action candidate by its best dev exact-span F1; keep both receipts.
The selection function reads no test score. However, each completed training
run automatically computes its test metrics before cross-candidate selection;
the procedure must not be described as a fully sealed test set. The
argument-stage schedule is unchanged.

### Development diagnosis and final candidate

The initial BIO argument run completed three epochs with dev full-span F1
10.61%, 14.89%, 14.54%. A 64-context dev diagnostic showed only 17/121 inside
tokens correctly predicted. Training labels are 133,146 O, 2,692 B, 3,276 I:
95.7% background. The upstream sign-only replay heavily favors correct O.
This run was deliberately stopped during epoch four and retained as a dev
pilot (`arg_seed0/pilot_status.json`); it is not the final test result.

A separately recorded candidate keeps the CNN/DQN and changes two BIO-specific
training details. The rate reward counts decoded mention starts (B or I after
O), so a multiword name is one extraction rather than several. The nominal 90%
positive reward replay quota is `int(0.9 * 32) = 28` samples (87.5% of a batch).
It is balanced among correct O/B/I, with uniform sampling
inside each group. The remaining quota still samples nonpositive rewards.
These are explicit adaptations, not claims about the upstream implementation.

Train `act_balanced_seed0` and `arg_balanced_seed0` from scratch with the same
seed, learning rate 0.001, 10-epoch maximum and patience 3. Their extra arguments
are `--span-replay balanced --span-rate mentions`. Select the action stage by
best dev full-span F1 across the three completed candidates. Select the final
argument checkpoint by dev full-span F1 within the completed balanced run.
The recorded decision uses train/dev diagnostics, and the selection function
uses dev alone; the harness nevertheless precomputes test scores as noted
above. Earlier training code is retained in `source_history` where
needed to audit historical receipt hashes.

## Evaluation definition

The action model first runs on each full test text. Only its predicted action
spans create argument contexts. The argument model predicts BIO spans in those
contexts. Names are copied directly from the original text using predicted
boundaries, without looking up gold spans, names, action masks, or relations.
An invalid I after O is deterministically decoded as the beginning of a span.

* **Action-name F1:** exact complete action name and occurrence boundaries.
* **Linked-argument-name F1:** both the complete parent action and the complete
  argument must match, including occurrence boundaries. An argument attached
  to the wrong action is incorrect. Every gold explicit link remains in the
  denominator, including arguments of actions the model misses.

Names normalize only case and whitespace. Counts are micro-averaged over the
78 test documents; repeated occurrences are distinct. No role label is needed
for a correct argument. Gold-action-conditioned argument scores in the stage
receipt are diagnostic, not the main linked-argument result.

## Reproduction commands

The following commands rerun the selected balanced configuration into fresh
output directories, reusing the existing verified BIO feature manifest. For a
fresh feature preparation, first run `prepare_ceasdrl_spans.py` as below only
if its output directory does not already exist:

```powershell
python scripts/ceasdrl/prepare_ceasdrl_spans.py --data data/procedural/risec/converted --existing-features data/procedural/risec/features_v1 --output data/procedural/risec/features_spans_v2
```

```powershell
$run = "results/ceasdrl_risec_rerun_seed0"

python scripts/ceasdrl/train_ceasdrl_risec.py --features data/procedural/risec/features_spans_v2 --output "$run/act_balanced_seed0" --parity-report results/ceasdrl_risec_20261118/parity/parity_report.json --mode act --span-labels --span-replay balanced --span-rate mentions --epochs 10 --patience 3 --seed 0

python scripts/ceasdrl/train_ceasdrl_risec.py --features data/procedural/risec/features_spans_v2 --output "$run/arg_balanced_seed0" --parity-report results/ceasdrl_risec_20261118/parity/parity_report.json --mode arg --span-labels --span-replay balanced --span-rate mentions --epochs 10 --patience 3 --seed 0

& C:\Users\Apexmod\miniforge3\envs\py36\python.exe scripts/ceasdrl/prepare_ceasdrl_spans.py --data data/procedural/risec/converted/test.jsonl --action-predictions "$run/act_balanced_seed0/test_predictions.json" --existing-features data/procedural/risec/features_v1 --output "$run/pipeline_features"

python scripts/ceasdrl/evaluate_ceasdrl_spans.py --data data/procedural/risec/converted/test.jsonl --features "$run/pipeline_features" --action-run "$run/act_balanced_seed0" --argument-run "$run/arg_balanced_seed0" --output "$run/pipeline_seed0"
```

Existing output directories are protected from overwrite. For an interrupted
training run, repeat its command with `--resume`; otherwise use a fresh output
directory. Cached contextual embeddings can be shared, but target manifests
are regenerated from complete spans. Missing ELMo contexts require the legacy
Python environment; training/evaluation use the main CUDA environment.

These rerun commands preserve the frozen legacy annotation policy, including
TODO links. They do not implement a cleaned-label benchmark or repair the
training harness's early computation of candidate test scores. The historical
initial BIO candidates remain available in `act_seed0`, `act_lr3e4_seed0`, and
`arg_seed0`; they are not the selected configuration shown above.

For a new unannotated text file, use the trained checkpoint paths recorded in
the final report:

```powershell
python scripts/ceasdrl/predict_ceasdrl_text.py --text-file recipe.txt --action-checkpoint ACTION_RUN/best.pt --argument-checkpoint ARGUMENT_RUN/best.pt --legacy-python C:\Users\Apexmod\miniforge3\envs\py36\python.exe --embedding-cache data/procedural/risec/features_v1 --output prediction_example
```

This entry point reads only the supplied text, predicts both stages, and writes
`predictions.json` with full names and their source positions. It supports up to
600 tokens per document, matching this model's training capacity. It does not
compute F1 for unlabeled text.

## Results

Completed. Both selected runs trained for all 10 epochs. Selection by dev F1
chose `act_balanced_seed0` epoch 8 and `arg_balanced_seed0` epoch 10.
Their dev full-span F1 scores are 95.8845% and 80.6667%, respectively.

| Final end-to-end test metric | Precision | Recall | F1 | Correct / predicted / gold |
|---|---:|---:|---:|---:|
| Complete action names | 95.57% | 97.95% | **96.75%** | 862 / 902 / 880 |
| Complete linked argument names | 75.33% | 89.54% | **81.82%** | 1447 / 1921 / 1616 |

The diagnostic argument test with gold actions is 83.57% F1. It is not the
reported end-to-end result. The previous first-token checkpoints, rescored with
the same full-name evaluator, remain 88.85% action / 26.44% linked-argument F1.
That old argument run only completed three epochs, so the before/after table
is a repair comparison, not a controlled ablation of one change.

All 28 relevant tests pass. Both trained checkpoints pass original Keras CNN
forward checks with `num_actions=3`: maximum absolute Q differences are
0.00030518 (action) and 0.00012207 (argument). Prediction from the unannotated
`182_BBQ_Chuck_Roast.txt` exactly matches the corresponding batch output.
The example preserves mistakes, including spurious parameters, for inspection.

Final artifacts are under `results/ceasdrl_risec_20261118/spans_v2/`:
`SUMMARY.md`, `summary.json`, `selection.json`, `learning_curves.csv`,
`pipeline_seed0/receipt.json`, `pipeline_seed0/predictions.json`,
`pipeline_seed0/errors.json`, and `prediction_example.md`.

Both selected runs retain `best.pt` checkpoints and exported
`best.keras.weights.h5` files.
