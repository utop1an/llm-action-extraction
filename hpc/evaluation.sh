#!/usr/bin/env bash
set -euo pipefail

driver="$(dirname -- "${BASH_SOURCE[0]}")/../scripts/batch_evaluate.py"
exec "${PYTHON:-python3}" "$driver" --hpc-serial
