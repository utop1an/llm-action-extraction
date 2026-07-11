#! /bin/bash
#SBATCH --job-name=act-gemma3-27b
#SBATCH --mail-type=END
#SBATCH --mail-user=u6797221@anu.edu.au
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.log

set -euo pipefail

pwd; hostname; date
echo "--- running ---"

BASE=$HOME/llm-action-extraction
MODEL_DIR=$HOME/ollama_models
SIF=$HOME/ollama.sif
OLLAMA_MODEL="gemma3:27b"
MODEL_KEY="gemma3-27b"
SOLVERS="${SOLVERS:-gpt3_to_plan nl2p_1 nl2p_1_ablation nl2p_1_coref}"
DATASET="${DATASET:-}"
LIMIT="${LIMIT:-}"
COREF="${COREF:-none}"
COREF_DIR="${COREF_DIR:-}"
RESUME="${RESUME:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
JOB_ID_FOR_PORT="${SLURM_JOB_ID:-$$}"
OLLAMA_PORT="${OLLAMA_PORT:-$((20000 + JOB_ID_FOR_PORT % 20000))}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:${OLLAMA_PORT}}"
export SINGULARITYENV_OLLAMA_HOST="$OLLAMA_HOST"
export APPTAINERENV_OLLAMA_HOST="$OLLAMA_HOST"

mkdir -p "$MODEL_DIR"

# 1. Start Ollama
echo "Starting Ollama on $OLLAMA_HOST"
singularity exec --nv \
  -B "$MODEL_DIR:/ollama" \
  "$SIF" \
  ollama serve &

OLLAMA_PID=$!
trap "kill $OLLAMA_PID 2>/dev/null || true" EXIT

# 2. Wait for Ollama
echo "Waiting for Ollama..."
until curl -s "http://${OLLAMA_HOST}/api/tags" >/dev/null; do
    sleep 2
done

# 3. Pull model (cached after first run)
singularity exec \
  -B "$MODEL_DIR:/ollama" \
  "$SIF" \
  ollama pull "$OLLAMA_MODEL"

# 4. Run experiments
for SOLVER in $SOLVERS; do
    SOLVER_COREF="$COREF"
    case "$SOLVER" in
        nl2p_coref|nl2p_1_coref)
            SOLVER_COREF="llm"
            ;;
    esac

    echo "Running solver: $SOLVER model: $MODEL_KEY coref: $SOLVER_COREF"

    ARGS=(-s "$SOLVER" -m "$MODEL_KEY")
    if [ -n "$DATASET" ]; then
        ARGS+=(-d "$DATASET")
    fi
    if [ -n "$LIMIT" ]; then
        ARGS+=(-l "$LIMIT")
    fi
    if [ "$SOLVER_COREF" != "none" ]; then
        ARGS+=(--coref "$SOLVER_COREF")
    fi
    if [ -n "$COREF_DIR" ]; then
        ARGS+=(--coref-dir "$COREF_DIR")
    fi
    if [ "$RESUME" != "0" ]; then
        ARGS+=(--resume)
    fi
    if [ -n "$CHECKPOINT_EVERY" ]; then
        ARGS+=(--checkpoint-every "$CHECKPOINT_EVERY")
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
