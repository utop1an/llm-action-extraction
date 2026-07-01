#!/usr/bin/env bash
set -euo pipefail

PREFIX="${PREFIX:-./results}"
SOLVERS="${SOLVERS:-nl2p_1 nl2p_1_ablation}"
MODELS="${MODELS:-gpt-5.4 gpt-5.4-mini gemma3-12b gemma3-27b llama3-3b llama3-70b}"
DIAGNOSTICS="${DIAGNOSTICS:---diagnostics}"
PYTHON="${PYTHON:-python3}"

for solver in $SOLVERS; do
    for model in $MODELS; do
        result_dir="$PREFIX/${solver}/${model}"
        if [ ! -d "$result_dir" ]; then
            echo "Skipping missing results dir: $result_dir"
            continue
        fi

        echo "Evaluating solver: $solver with model: $model"
        "$PYTHON" evaluation.py -d "$result_dir" $DIAGNOSTICS
        echo "Completed evaluation for solver: $solver with model: $model"
    done
done
