PREFIX="./results"
SOLVER=gpt3_to_plan

for model in gemma3 gemma3_12b llama3.2 gpt-4.1 gpt-4.1-mini gpt-4.1-nano; do
    echo "Evaluating solver: $solver with model: $model"

    python3 evaluation.py \
        -d "$PREFIX/${SOLVER}/${model}/"

    echo "Completed evaluation for solver: $solver with model: $model"
done
