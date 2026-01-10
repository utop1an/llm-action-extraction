PREFIX="./results"

for solver in nl2p_1, gpt3_to_plan; do
    for model in gemma3, gemma3_12b, llama3.2; do
        echo "Evaluating solver: $solver with model: $model"

        python3 evaluation.py \
            -d "$PREFIX/${solver}/${model}/"

        echo "Completed evaluation for solver: $solver with model: $model"
    done
done
