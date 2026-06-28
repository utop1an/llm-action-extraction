#! /bin/bash
#SBATCH --job-name=act-llama3-70b
#SBATCH --mail-type=END
#SBATCH --mail-user=u6797221@anu.edu.au
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.log

set -euo pipefail

pwd; hostname; date
echo "--- running ---"

BASE=$HOME/llm-action-extraction
MODEL_DIR=$HOME/ollama_models
SIF=$HOME/ollama.sif
OLLAMA_MODEL="llama3.3:70b"
MODEL_KEY="llama3-70b"
SOLVERS="${SOLVERS:-nl2p_1 nl2p_1_ablation}"
DATASET="${DATASET:-}"
LIMIT="${LIMIT:-}"
COREF="${COREF:-none}"
COREF_DIR="${COREF_DIR:-}"

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
  ollama pull "$OLLAMA_MODEL"

# 4. Run experiments
for SOLVER in $SOLVERS; do
    echo "Running solver: $SOLVER model: $MODEL_KEY"

    ARGS=(-s "$SOLVER" -m "$MODEL_KEY")
    if [ -n "$DATASET" ]; then
        ARGS+=(-d "$DATASET")
    fi
    if [ -n "$LIMIT" ]; then
        ARGS+=(-l "$LIMIT")
    fi
    if [ "$COREF" != "none" ]; then
        ARGS+=(--coref "$COREF")
    fi
    if [ -n "$COREF_DIR" ]; then
        ARGS+=(--coref-dir "$COREF_DIR")
    fi

    singularity exec --nv \
        -B "$MODEL_DIR:/ollama" \
        -B "$BASE:/work" \
        "$SIF" \
        bash -lc "cd /work && python3 -u experiment.py ${ARGS[*]}"

    sacct -j "$SLURM_JOB_ID" \
      --format=JobID,JobName,MaxRSS,MaxVMSize,State,Elapsed \
      >> "${SOLVER}_${MODEL_KEY}_${SLURM_JOB_ID}.log"

    echo "Completed solver: $SOLVER"
done

echo "--- done ---"
date
