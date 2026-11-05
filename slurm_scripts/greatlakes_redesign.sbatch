#!/usr/bin/env bash
# =============================================================================
# Three-point redesign, 2x2: {Cpd12, Cpd18} x {P1 replaced, P1 kept}.
# 100 samples each, filtered, written as a 3D SDF for docking.
#
#   sbatch --array=1-4 slurm_scripts/greatlakes_redesign.sbatch
#
#   task  ref    P1         core
#     1   Cpd12  REPLACED   warhead_p2n
#     2   Cpd12  KEPT       warhead_morpholine_p2n
#     3   Cpd18  REPLACED   warhead_p2n
#     4   Cpd18  KEPT       warhead_morpholine_p2n
#
# Both cores pin the P2 nitrogen, so its two groups regrow in every run; only
# P1 differs, which is what makes 1-vs-2 and 3-vs-4 clean comparisons.
#
# Tasks 1 and 3 free P1, which tunes the nitrile's electrophilicity: those need
# a covalent-reactivity check, not just a docking score.
#
# Output: $OUTPUT_ROOT/redesign_<registry-id>_<arm>_<YYYYmmdd_HHMMSS>/
# =============================================================================
#SBATCH --job-name=tgcpl_redesign
#SBATCH --account=maom0          # must match your Slurm account: run `my_accounts`
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=1-4
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/setup/greatlakes_env.sh"
cd "$TGCPL_DIR"

# --- shared settings ---------------------------------------------------------
N_SAMPLES=100
BATCH_SIZE=25
SEED=42

MAX_MW=500
MAX_ROTB=10        # Cpd12's own value, so no lead fails its own criterion
MAX_RING=7         # keeps cycloheptyl; prior runs made 11-14-membered rings

# N_x1 offset: at the reference atom count generation centres on the 500
# ceiling and ~half fails on MW. Shaving 2-6 atoms recentres below it.
# Must be space-separated; the "=-6" form does not parse.
ADD_ATOMS_MIN=-6
ADD_ATOMS_MAX=-2

# --- per-task configuration --------------------------------------------------
# Share one stamp across the array, else tasks stamp themselves minutes apart:
#   RUN_STAMP=$(date +%Y%m%d_%H%M%S) sbatch --export=ALL --array=1-4 ...
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

# Named by registry id so output joins back to the compound records.
case "${SLURM_ARRAY_TASK_ID:?set --array, e.g. sbatch --array=1-4}" in
  1) REF=Cpd12; VMCC=VMCC-0005199; CORE=warhead_p2n;            ARM=p1replaced ;;
  2) REF=Cpd12; VMCC=VMCC-0005199; CORE=warhead_morpholine_p2n; ARM=p1kept     ;;
  3) REF=Cpd18; VMCC=VMCC-0005222; CORE=warhead_p2n;            ARM=p1replaced ;;
  4) REF=Cpd18; VMCC=VMCC-0005222; CORE=warhead_morpholine_p2n; ARM=p1kept     ;;
  *) echo "ERROR: unknown array index ${SLURM_ARRAY_TASK_ID} (expected 1-4)" >&2
     exit 1 ;;
esac

TAG="${VMCC}_${ARM}"
OUT_DIR="$OUTPUT_ROOT/redesign_${TAG}_${RUN_STAMP}"

echo "=== task ${SLURM_ARRAY_TASK_ID} (${TAG}) on $HOSTNAME ==="
echo "started   : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "reference : $REF"
echo "core      : $CORE"
echo "samples   : $N_SAMPLES"
echo "output    : $OUT_DIR"
echo

# --- generate ----------------------------------------------------------------
python "$TGCPL_PKG_DIR/tgcpl_generate.py" \
    --ref-smiles "$REF" \
    --n-samples "$N_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --condition all \
    --scaffold-constraint \
    --scaffold-core "$CORE" \
    --add-atoms-range "$ADD_ATOMS_MIN" "$ADD_ATOMS_MAX" \
    --output-dir "$OUT_DIR" \
    --seed "$SEED"

# --- filter ------------------------------------------------------------------
# Writes passing.smi and passing.sdf (xTB-relaxed coords, ready for docking).
echo
echo ">>> filtering ${TAG}"
python "$GENERATE_DIR/filter_generated.py" "$OUT_DIR" \
    --max-mw "$MAX_MW" \
    --max-rotb "$MAX_ROTB" \
    --max-ring-size "$MAX_RING"

echo
echo "=== task ${SLURM_ARRAY_TASK_ID} (${TAG}) finished: $(date) ==="
echo "Boltz-2 input: $OUT_DIR/passing.sdf"
