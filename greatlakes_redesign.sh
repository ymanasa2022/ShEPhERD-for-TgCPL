#!/usr/bin/env bash
# =============================================================================
# greatlakes_redesign.sh — VMCC-0005199 / Cpd18 three-point redesign, 2x2.
#
#   sbatch --array=1-4 greatlakes_redesign.sh      # all four
#   sbatch --array=2   greatlakes_redesign.sh      # just one
#
# Four runs = {Cpd12, Cpd18} x {morpholine replaced, morpholine kept}.
# 100 samples each. Output is filtered and written as a 3D SDF for Boltz-2.
#
#   task  reference              P1 morpholine  scaffold core
#   ----  ---------------------  -------------  -----------------------
#     1   VMCC-0005199 (Cpd12)   REPLACED       warhead_p2n
#     2   VMCC-0005199 (Cpd12)   KEPT           warhead_morpholine_p2n
#     3   VMCC-0005222 (Cpd18)   REPLACED       warhead_p2n
#     4   VMCC-0005222 (Cpd18)   KEPT           warhead_morpholine_p2n
#
# Output: $OUTPUT_ROOT/redesign_<VMCC>_<arm>_<YYYYmmdd_HHMMSS>/
#
# Both cores pin the P2 nitrogen itself, so in every run the two groups on that
# nitrogen — the benzylic aryl/cycloalkyl and the basic chain — are regenerated.
# The only difference across the pair is whether P1 is held or explored, which
# is what makes 1-vs-2 and 3-vs-4 clean comparisons.
#
# Reactivity caveat for tasks 1 and 3: the nitrile's electrophilicity depends on
# how electron-poor the triazine is, and the P1 substituent tunes exactly that.
# Molecules from the "replaced" arm need a covalent-reactivity check, not just a
# docking score — Boltz-2 will happily pose a warhead that would not react.
# =============================================================================
#SBATCH --job-name=tgcpl_redesign
#SBATCH --account=maom99
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
source "$SCRIPT_DIR/greatlakes_env.sh"
cd "$TGCPL_DIR"

# --- shared settings ---------------------------------------------------------
N_SAMPLES=100
BATCH_SIZE=25
SEED=42

# MW ceiling for this campaign.
MAX_MW=500
# Rotatable-bond ceiling: Cpd12's own value, so nothing in the lead series is
# excluded by its own criterion.
MAX_ROTB=10
# Largest ring allowed. 7 keeps cycloheptyl (Cpd13, the 39.6x compound) and
# rejects macrocycles — prior unconstrained runs produced 11-, 12- and
# 14-membered rings, which are not useful starting points for this series.
MAX_RING=7

# N_x1 offset. Cpd12 embeds to 74 atoms (incl. H) at 495.6 Da and Cpd18 to 75 at
# 507.7 Da, i.e. ~6.7-6.8 Da/atom. Left at the reference count, generation
# centres right on the 500 ceiling and roughly half the output fails the filter
# on MW alone. Shaving 2-6 atoms recentres the distribution on ~460-485 Da.
#
# Pass space-separated: "--add-atoms-range=-6 -2" does NOT parse under argparse,
# "--add-atoms-range -6 -2" does.
ADD_ATOMS_MIN=-6
ADD_ATOMS_MAX=-2

# --- per-task configuration --------------------------------------------------
# Run stamp. All four tasks should share ONE stamp so the campaign groups in a
# listing; array tasks start at slightly different times, so taking `date`
# per-task would scatter them across several minutes. Export it at submission:
#
#   RUN_STAMP=$(date +%Y%m%d_%H%M%S) sbatch --export=ALL --array=1-4 greatlakes_redesign.sh
#
# The fallback keeps a bare `sbatch` working, just with per-task stamps.
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

# Runs are named by registry id rather than the repo's internal CpdN labels, so
# output directories join back to the compound records without a lookup table.
# The ids come from LEADS[<name>]["vmcc"] in tgcpl_generate.py.
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
python tgcpl_generate.py \
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
# Writes passing.smi and passing.sdf. The SDF carries the xTB-relaxed 3D
# coordinates, so it goes straight to Boltz-2 without re-embedding.
echo
echo ">>> filtering ${TAG}"
python filter_generated.py "$OUT_DIR" \
    --max-mw "$MAX_MW" \
    --max-rotb "$MAX_ROTB" \
    --max-ring-size "$MAX_RING"

echo
echo "=== task ${SLURM_ARRAY_TASK_ID} (${TAG}) finished: $(date) ==="
echo "Boltz-2 input: $OUT_DIR/passing.sdf"
