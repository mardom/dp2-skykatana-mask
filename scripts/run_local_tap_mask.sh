#!/usr/bin/env bash
# Memory-bounded local runner via External RSP TAP + Gaia DR3 TAP (`lsdb_env-py313`).
# Mirrors `./scripts/run_rsp_mask.sh` for running outside the Rubin Science Platform.
#
# Usage:
#   ./scripts/run_local_tap_mask.sh [configs/dp2_wfd_default.yaml] [optional extra dp2-mask args...]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${REPO_ROOT}/configs/dp2_wfd_default.yaml}"
if [[ $# -gt 0 ]]; then
    shift
fi

# Auto-detect `lsdb_env-py313` if PYTHON_BIN is not set explicitly
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${HOME}/miniconda3/envs/lsdb_env-py313/bin/python" ]]; then
        PYTHON_BIN="${HOME}/miniconda3/envs/lsdb_env-py313/bin/python"
    elif [[ -x "${HOME}/anaconda3/envs/lsdb_env-py313/bin/python" ]]; then
        PYTHON_BIN="${HOME}/anaconda3/envs/lsdb_env-py313/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-65536}"

echo "=== Pre-flight 32 GiB Memory Estimate (Local TAP Mode) ==="
"${PYTHON_BIN}" -m dp2_skykatana_mask.cli estimate-ram --config "${CONFIG_PATH}"

echo "=== Running DP2 SkyKatana Mask Pipeline Locally via IVOA TAP ==="
exec "${PYTHON_BIN}" -m dp2_skykatana_mask.cli build-tap --config "${CONFIG_PATH}" "$@"
