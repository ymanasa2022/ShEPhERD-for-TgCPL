#!/usr/bin/env bash
# =============================================================================
# greatlakes_smoke_test.sh — short GPU job proving the install works.
# Submit:  sbatch greatlakes_smoke_test.sh
# Run this BEFORE the big jobs; it costs ~10 min of one GPU.
# =============================================================================
#SBATCH --job-name=tgcpl_smoke
#SBATCH --account=maom99
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/greatlakes_env.sh"

echo "=== Smoke test ==="
echo "node    : $HOSTNAME"
echo "started : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

echo "--- 1. GPU visible to torch? ---"
python - <<'PY'
import shutil, sys, torch
ok = torch.cuda.is_available()
print("torch           :", torch.__version__)
print("cuda available  :", ok)
print("device          :", torch.cuda.get_device_name(0) if ok else "NONE")
print("xtb on PATH     :", shutil.which("xtb") or "*** MISSING ***")
if not ok:
    sys.exit("FAIL: no GPU visible to torch — check the cu* extra matches the driver")
if not shutil.which("xtb"):
    sys.exit("FAIL: xtb missing from PATH")
PY

echo
echo "--- 2. shepherd2 end-to-end (benzene, 3 samples) ---"
# Uses the repo's own test: embeds a SMILES, extracts the interaction profile,
# generates, xTB-relaxes, and scores. Exercises every component the real runs need.
cd "$REPO_DIR"
python examples/basic_inference/basic_inference_test.py \
    --smiles "c1ccccc1" \
    --output-dir "$SCRATCH/smoke_test_output" \
    --batch-size 3

echo
echo "--- 3. real TgCPL lead, 2 samples ---"
cd "$TGCPL_DIR"
python tgcpl_generate.py \
    --ref-smiles Cpd18 \
    --n-samples 2 \
    --batch-size 2 \
    --output-dir "$SCRATCH/smoke_test_tgcpl" \
    --seed 42

echo
echo "=== Smoke test finished: $(date) ==="
echo "If all three sections passed, submit:  sbatch greatlakes_tgcpl.sh"
