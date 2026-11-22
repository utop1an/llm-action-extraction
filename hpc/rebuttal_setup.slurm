#!/bin/bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -euo pipefail
: "${CAMPAIGN_ROOT:?}" "${SIF:?}"
singularity exec -B "$CAMPAIGN_ROOT:/work" "$SIF" \
    python3 -m pip install --disable-pip-version-check --no-cache-dir --target /work/eval_deps \
    'spacy==3.8.7' 'click==8.1.8' 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
export SINGULARITYENV_PYTHONPATH=/work/eval_deps
export APPTAINERENV_PYTHONPATH=/work/eval_deps
singularity exec -B "$CAMPAIGN_ROOT:/work" "$SIF" \
    bash -c 'cd /work && python3 scripts/run_hpc_rebuttal.py verify && python3 scripts/run_hpc_rebuttal.py validate-evaluator --model gemma3-27b'
