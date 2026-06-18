#! /bin/bash
#SBATCH --job-name=act-extraction
#SBATCH --mail-type=END
#SBATCH --mail-user=u6797221@anu.edu.au
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=%j_act-extraction.log

set -euo pipefail

pwd; hostname; date
echo "--- running ---"

BASE=$HOME/llm-action-extraction
MODEL_DIR=$HOME/ollama_models
SIF=$HOME/ollama.sif
MODEL="llama3.2"

mkdir -p "$MODEL_DIR"

# 1. Start Ollama
singularity exec --nv \
  -B "$MODEL_DIR:/ollama" \
  "$SIF" \
  ollama serve &

OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT

# 2. Wait for Ollama
echo "Waiting for Ollama..."
until curl -s http://127.0.0.1:11434/api/tags >/dev/null; do
    sleep 2
done

# 3. Pull model (cached after first run)
singularity exec \
  -B "$MODEL_DIR:/ollama" \
  "$SIF" \
  ollama pull "$MODEL"

# 4. Run experiments
for SOLVER in nl2p_1 ; do
    echo "Running solver: $SOLVER"

    singularity exec --nv \
        -B "$BASE:/work" \
        "$SIF" \
        python3 -u /work/experiment.py \
            -s "$SOLVER" \
            -m "$MODEL"

    sacct -j "$SLURM_JOB_ID" \
      --format=JobID,JobName,MaxRSS,MaxVMSize,State,Elapsed \
      >> "${SOLVER}_${MODEL}_${SLURM_JOB_ID}.log"

    echo "Completed solver: $SOLVER"
done

echo "--- done ---"
date
