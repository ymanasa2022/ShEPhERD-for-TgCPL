#!/usr/bin/env bash
# =============================================================================
# setup_shepherd2.sh — install ShEPhERD-2 on a generic HPC or workstation.
#
# ON GREAT LAKES USE setup/greatlakes_setup.sh INSTEAD. That one reads every
# path from setup/greatlakes_config.sh, so there is nothing to edit here and
# nothing to keep in sync. This script is the portable fallback for a machine
# that has no config file: edit the four paths below, then run it.
#
#   bash setup/setup_shepherd2.sh            # default CUDA build
#   bash setup/setup_shepherd2.sh cu124      # pick another
#
# Idempotent: pulls instead of re-cloning, skips env creation if the prefix
# exists, so it is safe to re-run after a failure.
# =============================================================================

set -euo pipefail

# --- edit these ------------------------------------------------------------
REPO_DIR="$HOME/shepherd2"                    # upstream coleygroup/shepherd2
ENV_PREFIX="$HOME/envs/shepherd2"             # conda env, by prefix
CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
# ---------------------------------------------------------------------------

TGCPL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_EXTRA="${1:-cu126}"   # cu118 | cu124 | cu126 | cu128 | cu130

case "$CUDA_EXTRA" in
    cu118|cu124|cu126|cu128|cu130) ;;
    *) echo "ERROR: unknown CUDA extra '$CUDA_EXTRA'." >&2
       echo "       Choose from: cu118 cu124 cu126 cu128 cu130" >&2
       exit 1 ;;
esac

echo "=== ShEPhERD-2 setup ==="
echo "    repo    : $REPO_DIR"
echo "    tgcpl   : $TGCPL_DIR"
echo "    env     : $ENV_PREFIX"
echo "    torch   : $CUDA_EXTRA"
echo

if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: conda hook not found at $CONDA_SH" >&2
    echo "       Set CONDA_SH at the top of this script to the right path." >&2
    exit 1
fi

# Source the hook explicitly rather than relying on the shell already having
# conda as a function: a non-interactive shell does not, and an inherited
# CONDA_EXE can point at a path that no longer exists.
source "$CONDA_SH"

# --- 1. repo ---------------------------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
    echo "[1/4] Repo exists — pulling..."
    git -C "$REPO_DIR" pull --ff-only
else
    echo "[1/4] Cloning shepherd2..."
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone https://github.com/coleygroup/shepherd2.git "$REPO_DIR"
fi

# --- 2. environment --------------------------------------------------------
# Built at an explicit --prefix rather than a named env, so it lands exactly
# where this script says. python<3.12 is a hard requirement of the shepherd
# package; 3.12 fails at metadata resolution with "requires a different
# Python". xtb must come from conda-forge -- it is not pip-installable, and
# generation dies partway with FileNotFoundError: 'xtb' without it.
if [ -d "$ENV_PREFIX" ]; then
    echo "[2/4] Env exists at $ENV_PREFIX — skipping create."
else
    echo "[2/4] Creating env (python 3.11 + xtb)..."
    mkdir -p "$(dirname "$ENV_PREFIX")"
    conda create -y --prefix "$ENV_PREFIX" -c conda-forge python=3.11 xtb
fi

conda activate "$ENV_PREFIX"

# Guard against installing into the wrong interpreter. If activation silently
# failed, pip falls back to a user install under ~/.local and scatters torch
# across the home quota.
_py_prefix="$(python -c 'import sys; print(sys.prefix)')"
if [ "$_py_prefix" != "$ENV_PREFIX" ]; then
    echo "ERROR: activate did not take." >&2
    echo "  expected sys.prefix $ENV_PREFIX" >&2
    echo "  got               $_py_prefix" >&2
    exit 1
fi
echo "      active python: $(python -V 2>&1) at $_py_prefix"

# --- 3. shepherd2 ----------------------------------------------------------
echo "[3/4] Installing shepherd2 [$CUDA_EXTRA]..."
cd "$REPO_DIR"
pip install --quiet uv
# No --system: install into the active conda prefix, not the base interpreter.
uv pip install -e ".[${CUDA_EXTRA}]"

# --- 4. verify -------------------------------------------------------------
echo "[4/4] Verifying..."
python - <<'PY'
import shutil, sys, torch
from shepherd import load_model
from shepherd_score.conformer_generation import embed_conformer_from_smiles
print("python       :", sys.version.split()[0])
print("torch        :", torch.__version__)
print("torch.cuda   :", torch.version.cuda)
print("cuda avail   :", torch.cuda.is_available())
print("xtb on PATH  :", shutil.which("xtb") or "*** MISSING ***")
print("shepherd     : import OK")
print()
print("NOTE: cuda avail is False on a login node (no GPU). Check it in a job.")
PY

cat <<EOF

=== Done ===

Activate with:
    source $CONDA_SH
    conda activate $ENV_PREFIX

Next:
    python $TGCPL_DIR/tests/test_redesign.py     # fast, no GPU
    python $TGCPL_DIR/scripts/tgcpl/tgcpl_generate.py --help
EOF
