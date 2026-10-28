#!/usr/bin/env bash
# =============================================================================
# greatlakes_setup.sh — install ShEPhERD-2 on U-M Great Lakes
#
# Run ONCE, on a login node (not via sbatch):
#   bash greatlakes_setup.sh
#
# Installs into /scratch so the ~80 GB home quota is not consumed by the
# conda env, the PyTorch wheels, or the downloaded model checkpoints.
# =============================================================================

set -euo pipefail

ACCOUNT="maom99"
UNIQNAME="$(whoami)"
SCRATCH="/scratch/${ACCOUNT}_root/${ACCOUNT}/${UNIQNAME}"
REPO_DIR="$SCRATCH/shepherd2"
ENV_PREFIX="$SCRATCH/envs/shepherd2"
CUDA_EXTRA="${1:-cu126}"   # cu118 | cu124 | cu126 | cu128 | cu130

echo "=== ShEPhERD-2 setup on Great Lakes ==="
echo "    scratch : $SCRATCH"
echo "    repo    : $REPO_DIR"
echo "    env     : $ENV_PREFIX"
echo "    torch   : $CUDA_EXTRA"
echo

if [ ! -d "$SCRATCH" ]; then
    echo "ERROR: $SCRATCH does not exist."
    echo "Check your scratch path with:  ls /scratch/${ACCOUNT}_root/${ACCOUNT}/"
    exit 1
fi

mkdir -p "$SCRATCH/envs" "$SCRATCH/hf_cache" "$SCRATCH/logs"

# --- 1. conda --------------------------------------------------------------
# Great Lakes provides Anaconda via Lmod. `module load` is how you get conda
# onto PATH; `conda activate` additionally needs the shell hook sourced.
module load python3.11-anaconda 2>/dev/null \
  || module load anaconda 2>/dev/null \
  || { echo "ERROR: no anaconda module found. Run 'module avail anaconda' and edit this line."; exit 1; }

source "$(conda info --base)/etc/profile.d/conda.sh"

# --- 2. repo ---------------------------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
    echo "[1/4] Repo exists — pulling..."
    git -C "$REPO_DIR" pull --ff-only
else
    echo "[1/4] Cloning shepherd2..."
    git clone https://github.com/coleygroup/shepherd2.git "$REPO_DIR"
fi

# --- 3. environment --------------------------------------------------------
# Built at an explicit --prefix in scratch rather than a named env in $HOME.
if [ -d "$ENV_PREFIX" ]; then
    echo "[2/4] Env exists at $ENV_PREFIX — skipping create."
else
    echo "[2/4] Creating env (python 3.11 + xtb)..."
    conda create -y --prefix "$ENV_PREFIX" -c conda-forge python=3.11 xtb
fi

conda activate "$ENV_PREFIX"

# --- 4. shepherd2 ----------------------------------------------------------
echo "[3/4] Installing shepherd2 [$CUDA_EXTRA]..."
cd "$REPO_DIR"
pip install --quiet uv
# No --system: install into the active conda prefix.
uv pip install -e ".[${CUDA_EXTRA}]"

# --- 5. verify -------------------------------------------------------------
echo "[4/4] Verifying..."
python - <<'PY'
import shutil, torch
from shepherd import load_model
from shepherd_score.conformer_generation import embed_conformer_from_smiles
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

Activate in a job script with:
    module load python3.11-anaconda
    source \$(conda info --base)/etc/profile.d/conda.sh
    conda activate $ENV_PREFIX

Next:  sbatch greatlakes_smoke_test.sh
EOF
