#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${CAMPAIGN_ROOT:?}" "${MODEL_KEY:?}" "${SIF:?}"
export SINGULARITYENV_PYTHONPATH=/work/eval_deps
export APPTAINERENV_PYTHONPATH=/work/eval_deps
singularity exec -B "$CAMPAIGN_ROOT:/work" "$SIF" \
    bash -c 'cd /work && python3 -u scripts/run_hpc_rebuttal.py evaluate --directory /work --model "$1"' \
    bash "$MODEL_KEY"
