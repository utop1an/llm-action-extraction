PREFIX="./results"
SOLVER=nl2p_1

for model in gemma3 gemma3_12b llama3.2 gpt-4.1 gpt-4.1-mini gpt-4.1-nano gpt-4o gpt-4o-mini; do
    echo "Evaluating solver: $solver with model: $model"

    python3 evaluation.py \
        -d "$PREFIX/${solver}/${model}/"

    echo "Completed evaluation for solver: $solver with model: $model"
done
