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

# sbatch copies this script into Slurm's spool dir, so BASH_SOURCE
# points there, not at the repo. Use the submit directory under Slurm;
# TGCPL_REPO overrides when submitting from outside the repo.
REPO_ROOT="${TGCPL_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [ ! -f "$REPO_ROOT/setup/greatlakes_config.sh" ]; then
    echo "ERROR: no repo at $REPO_ROOT" >&2
    echo "  submit from the repo root, or export TGCPL_REPO=/path/to/repo" >&2
    exit 1
fi
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
echo "--- 3. both campaign arms, 2 samples each ---"
# Same flags as greatlakes_redesign.sbatch tasks 2 and 4, only smaller: a lead
# that embeds unconstrained is no evidence the other one does, and neither
# exercises scaffold matching or the negative atom offset.
cd "$TGCPL_DIR"
for REF in Cpd12 Cpd18; do
    echo
    echo ">>> $REF"
    python "$TGCPL_PKG_DIR/tgcpl_generate.py" \
        --ref-smiles "$REF" \
        --n-samples 2 \
        --batch-size 2 \
        --condition all \
        --scaffold-constraint \
        --scaffold-core warhead_morpholine_p2n \
        --add-atoms-range -6 -2 \
        --output-dir "$SCRATCH/smoke_test_tgcpl" \
        --seed 42 || { echo "SMOKE TEST FAILED: $REF" >&2; exit 1; }
    # Exit 0 with nothing generated is the failure mode that reached production.
    SDF="$SCRATCH/smoke_test_tgcpl/$REF/generated_molecules.sdf"
    if [ ! -s "$SDF" ]; then
        echo "SMOKE TEST FAILED: $REF produced no molecules ($SDF)" >&2
        exit 1
    fi
done

echo
echo "=== Smoke test finished: $(date) ==="
echo "If all three sections passed, submit:"
echo "  RUN_STAMP=\$(date +%Y%m%d_%H%M%S) \\"
echo "      sbatch --export=ALL --array=2,4 slurm_scripts/greatlakes_redesign.sbatch"
