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

  *)  echo "ERROR: unknown array index $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo
echo "=== task ${SLURM_ARRAY_TASK_ID} finished: $(date) ==="
