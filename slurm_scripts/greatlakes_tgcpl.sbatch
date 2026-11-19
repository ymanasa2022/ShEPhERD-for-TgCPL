#!/usr/bin/env bash
# The five TgCPL generation runs, as a job array (one failure costs only its
# own task, and each task gets the full wall clock).
#
#   task 1  all leads, shape+ESP+pharm
#   task 2  Cpd18, shape+ESP only
#   task 3  Cpd18, grown 2-6 atoms
#   task 4  Cpd12, scaffold-constrained
#   task 5  dual-target Cpd18 vs Cpd1
#   task 6  SUPERSEDED by greatlakes_redesign.sbatch
#   task 99 Cpd12 redesign under a 400 Da ceiling
#SBATCH --job-name=tgcpl_gen
#SBATCH --account=maom0          # must match your Slurm account: run `my_accounts`
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=1-5
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=END,FAIL

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

cd "$TGCPL_DIR"

echo "=== array task ${SLURM_ARRAY_TASK_ID} on $HOSTNAME ==="
echo "started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

case "$SLURM_ARRAY_TASK_ID" in

  1)  echo ">>> run1: all leads, shape+ESP+pharm, 50 each (400 total)"
      python "$TGCPL_PKG_DIR/tgcpl_generate.py" \
          --ref-smiles all \
          --n-samples 50 \
          --batch-size 25 \
          --condition all \
          --output-dir "$OUTPUT_ROOT/run1_all_leads" \
          --seed 42
      ;;

  2)  # No pharmacophore constraint -> more structural diversity.
      echo ">>> run2: Cpd18, shape+ESP only, 100 samples"
      python "$TGCPL_PKG_DIR/tgcpl_generate.py" \
          --ref-smiles Cpd18 \
          --n-samples 100 \
          --batch-size 25 \
          --condition shape esp \
          --output-dir "$OUTPUT_ROOT/run2_shape_esp_Cpd18" \
          --seed 42
      ;;

  3)  echo ">>> run3: Cpd18, grown by 2-6 atoms, 100 samples"
      python "$TGCPL_PKG_DIR/tgcpl_generate.py" \
          --ref-smiles Cpd18 \
          --n-samples 100 \
          --batch-size 25 \
          --condition all \
          --add-atoms-range 2 6 \
          --add-pharms-range 0 2 \
          --output-dir "$OUTPUT_ROOT/run3_grow_Cpd18" \
          --seed 42
      ;;

  4)  # Keeps the triazine-CN-morpholine warhead fixed.
      echo ">>> run4: Cpd12, scaffold-constrained, 100 samples"
      python "$TGCPL_PKG_DIR/tgcpl_generate.py" \
          --ref-smiles Cpd12 \
          --n-samples 100 \
          --batch-size 25 \
          --condition all \
          --scaffold-constraint \
          --output-dir "$OUTPUT_ROOT/run4_scaffold_Cpd12" \
          --seed 42
      ;;

  5)  # Selectivity at the model level: ON Cpd18, NOT Cpd1.
      echo ">>> run5: dual-target Cpd18 vs Cpd1, scaffold-constrained, 100 samples"
      python "$TGCPL_PKG_DIR/tgcpl_generate.py" \
          --dual-target \
          --dual-tgcpl-ref Cpd18 \
          --dual-hscpl-ref Cpd1 \
          --scaffold-constraint \
          --n-samples 100 \
          --batch-size 25 \
          --output-dir "$OUTPUT_ROOT/run5_dual_target" \
          --seed 42
      ;;

  6)  # Kept so an old `sbatch --array=6` fails loudly instead of vanishing.
      echo "run6 moved: use  sbatch --array=1-4 greatlakes_redesign.sh"; exit 1
      ;;

  99) # VMCC-0005199 (= Cpd12), 400 Da ceiling. Pins the warhead + P2 nitrogen
      # only, so P1 and both groups on that nitrogen regenerate.
      #
      # The negative --add-atoms-range is the MW lever: ~6.7 Da/atom here, so
      # N_x1 57-63 targets 380-420 Da; at the reference 74 atoms generation
      # lands near 495 Da. The flag was written for GROWING, so negatives are
      # unexercised -- try --n-samples 2 first. Must be space-separated;
      # "--add-atoms-range=-17 -11" does not parse.
      echo ">>> run6: VMCC-0005199/Cpd12, P1+P2+P3 free, N_x1 57-63, 150 samples"
      python "$TGCPL_PKG_DIR/tgcpl_generate.py" \
          --ref-smiles Cpd12 \
          --n-samples 150 \
          --batch-size 25 \
          --condition all \
          --scaffold-constraint \
          --scaffold-core warhead_p2n \
          --add-atoms-range -17 -11 \
          --output-dir "$OUTPUT_ROOT/run6_vmcc5199_mw400" \
          --seed 42
      python "$GENERATE_DIR/filter_generated.py" "$OUTPUT_ROOT/run6_vmcc5199_mw400" \
          --max-mw 400 --max-rotb 7 --max-ring-size 7
      ;;

  *)  echo "ERROR: unknown array index $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo
echo "=== task ${SLURM_ARRAY_TASK_ID} finished: $(date) ==="
