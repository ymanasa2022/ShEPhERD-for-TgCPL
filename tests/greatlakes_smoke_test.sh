#!/usr/bin/env bash
# Short GPU job proving the install works. Run before the big jobs.
#SBATCH --job-name=tgcpl_smoke
#SBATCH --account=maom0          # must match your Slurm account: run `my_accounts`
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
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/setup/greatlakes_env.sh"

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
# The repo's own test: embed, profile, generate, xTB-relax, score -- every
# component the real runs need.
cd "$REPO_DIR"
python examples/basic_inference/basic_inference_test.py \
    --smiles "c1ccccc1" \
    --output-dir "$SCRATCH/smoke_test_output" \
    --batch-size 3

echo
echo "--- 3. real TgCPL lead, 2 samples ---"
cd "$TGCPL_DIR"
python "$TGCPL_PKG_DIR/tgcpl_generate.py" \
    --ref-smiles Cpd18 \
    --n-samples 2 \
    --batch-size 2 \
    --output-dir "$SCRATCH/smoke_test_tgcpl" \
    --seed 42

echo
echo "=== Smoke test finished: $(date) ==="
echo "If all three sections passed, submit:  sbatch slurm_scripts/greatlakes_tgcpl.sbatch"
