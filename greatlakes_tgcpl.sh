#!/usr/bin/env bash
# =============================================================================
# greatlakes_tgcpl.sh — the five TgCPL generation runs, as a job array.
#
# Submit all:        sbatch greatlakes_tgcpl.sh
# Submit a subset:   sbatch --array=1,3 greatlakes_tgcpl.sh
#
# An array runs the five independently and in parallel, so one failure does not
# cost the others, and each gets the full wall clock rather than sharing it.
# =============================================================================
#SBATCH --job-name=tgcpl_gen
#SBATCH --account=maom99
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/greatlakes_env.sh"

cd "$TGCPL_DIR"

echo "=== array task ${SLURM_ARRAY_TASK_ID} on $HOSTNAME ==="
echo "started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

case "$SLURM_ARRAY_TASK_ID" in

  1)  # All 8 leads, full conditioning. The broad baseline sweep.
      echo ">>> run1: all leads, shape+ESP+pharm, 50 each (400 total)"
      python tgcpl_generate.py \
          --ref-smiles all \
          --n-samples 50 \
          --batch-size 25 \
          --condition all \
          --output-dir "$OUTPUT_ROOT/run1_all_leads" \
          --seed 42
      ;;

  2)  # Drop the pharmacophore constraint -> more structural diversity.
      echo ">>> run2: Cpd18, shape+ESP only, 100 samples"
      python tgcpl_generate.py \
          --ref-smiles Cpd18 \
          --n-samples 100 \
          --batch-size 25 \
          --condition shape esp \
          --output-dir "$OUTPUT_ROOT/run2_shape_esp_Cpd18" \
          --seed 42
      ;;

  3)  # Grow 2-6 heavy atoms: explore extended P2 / larger P3.
      echo ">>> run3: Cpd18, grown by 2-6 atoms, 100 samples"
      python tgcpl_generate.py \
          --ref-smiles Cpd18 \
          --n-samples 100 \
          --batch-size 25 \
          --condition all \
          --add-atoms-range 2 6 \
          --add-pharms-range 0 2 \
          --output-dir "$OUTPUT_ROOT/run3_grow_Cpd18" \
          --seed 42
      ;;

  4)  # Keep the triazine-CN-morpholine warhead fixed.
      echo ">>> run4: Cpd12, scaffold-constrained, 100 samples"
      python tgcpl_generate.py \
          --ref-smiles Cpd12 \
          --n-samples 100 \
          --batch-size 25 \
          --condition all \
          --scaffold-constraint \
          --output-dir "$OUTPUT_ROOT/run4_scaffold_Cpd12" \
          --seed 42
      ;;

  5)  # Selectivity encoded at the model level: ON Cpd18, NOT Cpd1.
      echo ">>> run5: dual-target Cpd18 vs Cpd1, scaffold-constrained, 100 samples"
      python tgcpl_generate.py \
          --dual-target \
          --dual-tgcpl-ref Cpd18 \
          --dual-hscpl-ref Cpd1 \
          --scaffold-constraint \
          --n-samples 100 \
          --batch-size 25 \
          --output-dir "$OUTPUT_ROOT/run5_dual_target" \
          --seed 42
      ;;

  6)  # SUPERSEDED by greatlakes_redesign.sh (4-run 2x2 at MW<=500). Kept only
      # so an old `sbatch --array=6` does not silently vanish.
      echo "run6 moved: use  sbatch --array=1-4 greatlakes_redesign.sh"; exit 1
      ;;

  99) # VMCC-0005199 (= Cpd12) three-point redesign, under a 400 Da ceiling.
      #
      # Pins the warhead + the P2 nitrogen only, so the P1 morpholine and BOTH
      # groups on that nitrogen (benzylic aryl/cycloalkyl and the basic chain)
      # are regenerated.
      #
      # The negative --add-atoms-range is the MW lever: Cpd12 embeds to 74 atoms
      # incl. H, and this series runs ~6.7 Da/atom, so N_x1 57-63 targets the
      # 380-420 Da band. Without it N_x1 defaults to the reference's 74 and
      # generation lands near 495 Da, nowhere near the ceiling. Note the flag
      # was written for GROWING; negatives are arithmetically fine but
      # unexercised, so run this once at --n-samples 2 before the full job.
      #
      # Pass the two values space-separated. "--add-atoms-range=-17 -11" does
      # NOT parse; argparse needs "--add-atoms-range -17 -11".
      echo ">>> run6: VMCC-0005199/Cpd12, P1+P2+P3 free, N_x1 57-63, 150 samples"
      python tgcpl_generate.py \
          --ref-smiles Cpd12 \
          --n-samples 150 \
          --batch-size 25 \
          --condition all \
          --scaffold-constraint \
          --scaffold-core warhead_p2n \
          --add-atoms-range -17 -11 \
          --output-dir "$OUTPUT_ROOT/run6_vmcc5199_mw400" \
          --seed 42
      python filter_generated.py "$OUTPUT_ROOT/run6_vmcc5199_mw400" \
          --max-mw 400 --max-rotb 7 --max-ring-size 7
      ;;

  *)  echo "ERROR: unknown array index $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo
echo "=== task ${SLURM_ARRAY_TASK_ID} finished: $(date) ==="
