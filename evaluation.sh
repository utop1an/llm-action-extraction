PREFIX="./results/experiment"
SOLVERS="gpt3_to_plan nl2p_1 nl2p_2 nl2p_3"
MODELS="gemma3 gemma3_12b llama3.2 gpt-4.1 gpt-4.1-mini gpt-4.1-nano"

for solver in $SOLVERS; do
    for model in $MODELS; do
        echo "Evaluating solver: $solver with model: $model"

        python3 evaluation_first_match.py \
            -d "$PREFIX/${solver}/${model}/"
        echo "Completed evaluation for solver: $solver with model: $model"
    done
done