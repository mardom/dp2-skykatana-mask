#!/usr/bin/env bash
# Memory-hardened runner for 32 GiB RAM RSP Nublado terminals / batch shells.
# Usage:
#   ./scripts/run_rsp_mask.sh [configs/dp2_wfd_default.yaml] [optional extra dp2-mask args...]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${REPO_ROOT}/configs/dp2_wfd_default.yaml}"
if [[ $# -gt 0 ]]; then
    shift
fi

export PATH="${HOME}/.local/bin:${PATH}"

# Prevent OpenMP/BLAS thread explosion on 64-core shared Kubernetes nodes
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

# Return freed glibc arenas promptly to the OS so pod cgroup RSS stays low
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-65536}"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "=== Pre-flight 32 GiB RSP Memory Estimate ==="
"${PYTHON_BIN}" -m dp2_skykatana_mask.cli estimate-ram --config "${CONFIG_PATH}"

echo "=== Running DP2 SkyKatana Mask Pipeline ==="
exec "${PYTHON_BIN}" -m dp2_skykatana_mask.cli build --config "${CONFIG_PATH}" "$@"
