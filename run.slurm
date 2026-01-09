#! /bin/bash
#SBATCH --job-name=act-extraction
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=u6797221@anu.edu.au     # Where to send mail
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --output=%j_act-extraction.log
# Standard output and error log
set -euo pipefail

pwd; hostname; date
echo "--- running ---"

MODEL_DIR=~/ollama_models
SIF=~/ollama.sif
MODEL="gemma3"

mkdir -p "$MODEL_DIR"

# 1. Start Ollama server (background, SAME JOB)
singularity exec --nv \
  -B $MODEL_DIR:/ollama \
  $SIF \
  ollama serve &

OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT

# 2. Wait for server to be ready
echo "Waiting for Ollama..."
until curl -s http://127.0.0.1:11434/api/tags >/dev/null; do
    sleep 2
done

# 3. Pull Gemma 3 (cached after first run)
singularity exec \
  -B $MODEL_DIR:/ollama \
  $SIF \
  ollama pull $model

# 4. Run experiments
for SOLVER in nl2p_1 gp3_to_plan; do
    echo "Running task solver: $solver"
    singularity exec -B ~/llm-action-extraction/src/:/src \
        -B ~/llm-action-extraction/data/:/data \
        -B ~/llm-action-extraction/logs:/logs \
        -B ~/llm-action-extraction/results/:/results \
        $SIF python3 /src/experiment.py \
        -s $SOLVER -m $MODEL

    echo "Completed task solver: $SOLVER, model: $MODEL"

    singularity exec -B ~/llm-action-extraction/src/:/src \
        -B ~/llm-action-extraction/data/:/data \
        -B ~/llm-action-extraction/logs:/logs \
        -B ~/llm-action-extraction/results/:/results \
        $SIF python3 /src/evaluation.py \
        -d results/$SOLVER/$MODEL/ \

    echo "Completed task solver: $SOLVER, model: $MODEL"

    sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,MaxVMSize,State,Elapsed >> $SOLVER_$MODEL_$SLURM_JOB_ID.log
done

echo "--- done ---"
date
