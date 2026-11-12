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

# Automatically initialize EUPS `lsst_distrib` in fresh RSP terminals / tmux sessions
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if ! command -v python >/dev/null 2>&1 || ! python -c "import lsst.daf.butler" >/dev/null 2>&1; then
        for lsst_loader in \
            "/opt/lsst/software/stack/loadLSST.bash" \
            "/opt/lsst/software/stack/loadLSST.sh" \
            "${LSST_HOME:-}/loadLSST.bash"; do
            if [[ -n "${lsst_loader}" && -f "${lsst_loader}" ]]; then
                echo "=== Initializing RSP LSST Science Pipelines (${lsst_loader} + setup lsst_distrib) ==="
                set +u
                # shellcheck disable=SC1090
                source "${lsst_loader}"
                setup lsst_distrib
                set -u
                break
            fi
        done
    fi
    PYTHON_BIN="python"
fi

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
