#!/usr/bin/env bash
# =============================================================================
# greatlakes_setup.sh — install ShEPhERD-2 on U-M Great Lakes
#
# Run ONCE, on a login node (not via sbatch):
#   bash greatlakes_setup.sh            # default CUDA build
#   bash greatlakes_setup.sh cu124      # pick another
#
# Idempotent: pulls instead of re-cloning, skips env creation if the prefix
# exists, so it is safe to re-run after a failure.
#
# Paths live in greatlakes_config.sh; edit them there, not here.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/greatlakes_config.sh"

CUDA_EXTRA="${1:-cu126}"   # cu118 | cu124 | cu126 | cu128 | cu130

echo "=== ShEPhERD-2 setup on Great Lakes ==="
echo "    turbo   : $TURBO"
echo "    scratch : $SCRATCH"
echo "    repo    : $REPO_DIR"
echo "    tgcpl   : $TGCPL_DIR"
echo "    env     : $ENV_PREFIX"
echo "    hf_home : $HF_HOME"
echo "    torch   : $CUDA_EXTRA"
echo

# --- 0. preconditions ------------------------------------------------------
# --pre-install skips the ENV_PREFIX check: creating it is this script's job.
if ! tgcpl_check_paths --pre-install; then
    echo
    echo "Fix the paths in greatlakes_config.sh before continuing." >&2
    exit 1
fi

mkdir -p "$SCRATCH" "$(dirname "$ENV_PREFIX")" "$HF_HOME" "$SCRATCH/logs"

# --- 1. conda --------------------------------------------------------------
# Sourced from CONDA_SH rather than `module load` + `conda info --base`: the
# inherited CONDA_EXE points at a mount path that no longer exists, so the
# conda shell function fails on activate. See greatlakes_config.sh.
source "$CONDA_SH"

# --- 2. repo ---------------------------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
    echo "[1/4] Repo exists — pulling..."
    git -C "$REPO_DIR" pull --ff-only
else
    echo "[1/4] Cloning shepherd2..."
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone https://github.com/coleygroup/shepherd2.git "$REPO_DIR"
fi

# --- 3. environment --------------------------------------------------------
# Built at an explicit --prefix rather than a named env, so it lands exactly
# where greatlakes_config.sh says. python<3.12 is a hard requirement of the
# shepherd package; 3.12 fails at metadata resolution with
# "requires a different Python".
if [ -d "$ENV_PREFIX" ]; then
    echo "[2/4] Env exists at $ENV_PREFIX — skipping create."
else
    echo "[2/4] Creating env (python 3.11 + xtb)..."
    conda create -y --prefix "$ENV_PREFIX" -c conda-forge python=3.11 xtb
fi

conda activate "$ENV_PREFIX"

# Guard against installing into the wrong interpreter. If activation silently
# failed, pip would fall back to a user install under ~/.local and scatter
# torch across the home quota.
_py_prefix="$(python -c 'import sys; print(sys.prefix)')"
if [ "$_py_prefix" != "$ENV_PREFIX" ]; then
    echo "ERROR: activate did not take." >&2
    echo "  expected sys.prefix $ENV_PREFIX" >&2
    echo "  got               $_py_prefix" >&2
    exit 1
fi
echo "      active python: $(python -V 2>&1) at $_py_prefix"

# --- 4. shepherd2 ----------------------------------------------------------
echo "[3/4] Installing shepherd2 [$CUDA_EXTRA]..."
cd "$REPO_DIR"
pip install --quiet uv
# No --system: install into the active conda prefix.
uv pip install -e ".[${CUDA_EXTRA}]"

# --- 5. verify -------------------------------------------------------------
echo "[4/4] Verifying..."
python - <<'PY'
import shutil, sys, torch
from shepherd import load_model
from shepherd_score.conformer_generation import embed_conformer_from_smiles
print("python       :", sys.version.split()[0])
print("torch        :", torch.__version__)
print("torch.cuda   :", torch.version.cuda)
print("xtb on PATH  :", shutil.which("xtb") or "*** MISSING ***")
print("shepherd     : import OK")
print()
print("NOTE: torch.cuda.is_available() is False on a login node (no GPU).")
print("      The smoke-test job checks the GPU.")
PY

cat <<EOF

=== Done ===

Job scripts activate this themselves via greatlakes_env.sh. To activate by
hand:
    source $SCRIPT_DIR/greatlakes_env.sh

Next:
    python $TGCPL_DIR/tests/test_redesign.py     # fast, no GPU
    sbatch $SCRIPT_DIR/greatlakes_smoke_test.sh  # checks the GPU
EOF
