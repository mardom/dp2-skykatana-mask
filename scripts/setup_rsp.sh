#!/usr/bin/env bash
# Safe user-space installer for Rubin Science Platform (RSP) Nublado pods (`rubin-env`).
#
# Why `--user --no-deps`?
# On RSP, the `LSST` kernel (`rubin-env`) already ships binary-matched builds of
# `lsst.daf.butler`, `lsst.sphgeom`, `healsparse`, `hpgeom`, `numpy`, `pyarrow`,
# `astropy`, `pandas`, and `matplotlib`. Running an unconstrained `pip install`
# can overwrite `numpy` or `pyarrow` in `~/.local` and break `lsst.daf.butler`!
# This script installs ONLY missing lightweight packages + latest `skykatana@main`
# + `dp2-skykatana-mask` without touching `rubin-env` core libraries.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# In RSP JupyterLab terminals (unlike the `LSST` notebook kernel), EUPS `lsst_distrib`
# is not initialized automatically at shell startup. Bootstrap it BEFORE resolving
# `python` / `pip` so user-space packages and `~/.local/bin/dp2-mask` bind to `rubin-env`.
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if ! command -v python >/dev/null 2>&1 || ! python -c "import lsst.daf.butler" >/dev/null 2>&1; then
        for lsst_loader in \
            "/opt/lsst/software/stack/loadLSST.bash" \
            "/opt/lsst/software/stack/loadLSST.sh" \
            "${LSST_HOME:-}/loadLSST.bash"; do
            if [[ -n "${lsst_loader}" && -f "${lsst_loader}" ]]; then
                echo "=== [0/4] Bootstrapping RSP LSST Science Pipelines (${lsst_loader} + setup lsst_distrib) ==="
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

echo "=== [1/4] Checking active Python environment ==="
"${PYTHON_BIN}" -c "import sys; print('Python:', sys.executable, sys.version)"

PIP_USER_FLAG=()
# Use `--user` when inside a read-only system/conda prefix like RSP `/opt/lsst/software/stack/...`
if ! "${PYTHON_BIN}" -c "import os, sysconfig; p = sysconfig.get_paths()['purelib']; raise SystemExit(0 if os.access(p, os.W_OK) else 1)"; then
    echo "Read-only base environment detected (RSP rubin-env); installing to ~/.local (--user)."
    PIP_USER_FLAG=(--user)
fi

echo "=== [2/4] Ensuring lightweight SkyKatana runtime dependencies are present ==="
for pkg_spec in "mocpy:mocpy>=0.15" "tqdm:tqdm" "yaml:pyyaml"; do
    mod_name="${pkg_spec%%:*}"
    pip_req="${pkg_spec#*:}"
    if ! "${PYTHON_BIN}" -c "import ${mod_name}" >/dev/null 2>&1; then
        echo "Installing missing dependency: ${pip_req}"
        "${PYTHON_BIN}" -m pip install "${PIP_USER_FLAG[@]}" "${pip_req}"
    else
        echo "Found ${mod_name} in environment."
    fi
done

echo "=== [3/4] Installing latest SkyKatana from GitHub main (--no-deps) ==="
"${PYTHON_BIN}" -m pip install "${PIP_USER_FLAG[@]}" --upgrade --no-deps \
    "git+https://github.com/samotracio/skykatana.git@main"

echo "=== [4/4] Installing dp2-skykatana-mask editable (--no-deps) ==="
# `--force-reinstall` rewrites `~/.local/bin/dp2-mask` shebang to the active `rubin-env` Python
"${PYTHON_BIN}" -m pip install "${PIP_USER_FLAG[@]}" --force-reinstall --no-deps -e "${REPO_ROOT}"

echo "=== Setup complete! Verifying imports ==="
"${PYTHON_BIN}" -c "import skykatana, dp2_skykatana_mask; print('skykatana:', getattr(skykatana, '__version__', 'installed'), '| dp2_skykatana_mask:', dp2_skykatana_mask.__version__)"
if [[ -f "/opt/lsst/software/stack/loadLSST.bash" ]]; then
    "${PYTHON_BIN}" -c "import lsst.daf.butler as b; print('lsst.daf.butler OK:', b.__file__)"
fi
echo "Tip: In any new RSP terminal tab, ensure lsst_distrib and ~/.local/bin are active:"
echo "  source /opt/lsst/software/stack/loadLSST.bash && setup lsst_distrib"
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
