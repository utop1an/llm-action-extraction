#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${CAMPAIGN_ROOT:?}" "${MODEL_KEY:?}" "${SIF:?}" "${MODEL_DIR:?}"
cd "$CAMPAIGN_ROOT"
export OLLAMA_HOST="127.0.0.1:$((20000 + SLURM_JOB_ID % 20000))"
export SINGULARITYENV_OLLAMA_HOST="$OLLAMA_HOST"
export APPTAINERENV_OLLAMA_HOST="$OLLAMA_HOST"
export REBUTTAL_SIF_SHA256="$(sha256sum "$SIF" | cut -d ' ' -f1)"
export SINGULARITYENV_REBUTTAL_SIF_SHA256="$REBUTTAL_SIF_SHA256"
export APPTAINERENV_REBUTTAL_SIF_SHA256="$REBUTTAL_SIF_SHA256"
nvidia-smi
singularity exec --nv -B "$MODEL_DIR:/ollama" "$SIF" ollama serve &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT
READY=0
for attempt in $(seq 1 90); do
    kill -0 "$OLLAMA_PID"
    if curl -fsS "http://$OLLAMA_HOST/api/tags" >/dev/null; then READY=1; break; fi
    sleep 2
done
[ "$READY" = 1 ] || { echo 'Ollama startup timed out'; exit 1; }
for CELL_INDEX in $(seq "${SLURM_ARRAY_TASK_ID:-0}" "${SHARDS:-1}" 29); do
    singularity exec --nv -B "$MODEL_DIR:/ollama" -B "$CAMPAIGN_ROOT:/work" "$SIF" \
        bash -c 'cd /work && python3 -u scripts/run_hpc_rebuttal.py run --directory /work --model "$1" --index "$2"' \
        bash "$MODEL_KEY" "$CELL_INDEX"
done
