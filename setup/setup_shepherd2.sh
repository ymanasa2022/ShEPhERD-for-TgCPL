#!/usr/bin/env bash
# =============================================================================
# setup_shepherd2.sh
# One-shot setup script for ShEPhERD-2 on an HPC (SLURM/PBS, CUDA ≥ 12.4)
#
# Usage: bash setup_shepherd2.sh [cuda_version]
#   cuda_version: 118 | 124 | 126 | 128 | 130  (default: 126)
#
# After this runs, activate with:
#   conda activate shepherd2
# =============================================================================

set -e

CUDA_VER="${1:-126}"
VALID_VERS="118 124 126 128 130"
if ! echo "$VALID_VERS" | grep -qw "$CUDA_VER"; then
    echo "ERROR: CUDA version '$CUDA_VER' not recognised. Choose from: $VALID_VERS"
    exit 1
fi

echo "========================================================"
echo " Setting up ShEPhERD-2 with CUDA $CUDA_VER"
echo "========================================================"

# 1. Clone (or update) the repo
REPO_DIR="$HOME/shepherd2"
if [ -d "$REPO_DIR/.git" ]; then
    echo "[1/5] Repo already exists — pulling latest..."
    git -C "$REPO_DIR" pull --ff-only
else
    echo "[1/5] Cloning ShEPhERD-2..."
    git clone https://github.com/coleygroup/shepherd2.git "$REPO_DIR"
fi

# 2. Create conda environment with Python 3.11 + xTB (required, not pip-installable)
echo "[2/5] Creating conda environment 'shepherd2' with xTB..."
conda create -n shepherd2 python=3.11 conda-forge::xtb -y

# 3. Install ShEPhERD with the correct CUDA extra via uv (bundled in conda env)
echo "[3/5] Installing ShEPhERD (CUDA $CUDA_VER)..."
conda run -n shepherd2 pip install uv
conda run -n shepherd2 bash -c "
    cd $REPO_DIR && \
    uv pip install -e '.[cu${CUDA_VER}]' --python \$(which python) --system
"

echo "[4/5] Verifying install..."
conda run -n shepherd2 python -c "
import torch
from shepherd import load_model
from shepherd_score.conformer_generation import embed_conformer_from_smiles
print('torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
print('ShEPhERD import OK')
"

echo "[5/5] Done!"
echo ""
echo "  ✓ Activate with:  conda activate shepherd2"
echo "  ✓ Repo at:        $REPO_DIR"
echo "  ✓ Test with:      python $REPO_DIR/examples/basic_inference/basic_inference_test.py --device cpu"
echo ""
echo "  ✓ TgCPL run:      python $REPO_DIR/tgcpl_generate.py"
