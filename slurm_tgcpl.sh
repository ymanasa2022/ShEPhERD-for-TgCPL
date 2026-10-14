#!/usr/bin/env bash
#SBATCH --job-name=tgcpl_shepherd2
#SBATCH --output=logs/shepherd2_%j.out
#SBATCH --error=logs/shepherd2_%j.err
#SBATCH --partition=gpu         # <-- change to your GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8       # for xTB workers
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# =============================================================================
# SLURM job script for TgCPL ShEPhERD-2 generation
#
# Edit the paths below, then submit with:
#   mkdir -p logs
#   sbatch slurm_tgcpl.sh
# =============================================================================

SHEPHERD_DIR="$HOME/shepherd2"           # path to cloned repo
SCRIPT="$SHEPHERD_DIR/tgcpl_generate.py"
OUTPUT_DIR="$SHEPHERD_DIR/tgcpl_outputs"

module load cuda/12.6 2>/dev/null || true   # adjust to your HPC module name
conda activate shepherd2

cd "$SHEPHERD_DIR"
mkdir -p logs

echo "Job started: $(date)"
echo "Node: $HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# -------------------------------------------------------------------
# Run 1: Generate from all 4 lead molecules
#         Condition on shape + ESP + pharmacophores
#         50 samples per lead  (200 total)
# -------------------------------------------------------------------
python "$SCRIPT" \
    --ref-smiles all \
    --n-samples 50 \
    --batch-size 25 \
    --condition all \
    --output-dir "$OUTPUT_DIR/run1_all_leads" \
    --seed 42

# -------------------------------------------------------------------
# Run 2: Shape + ESP only (no pharmacophore) from best leads
#         Allows more structural diversity
# -------------------------------------------------------------------
python "$SCRIPT" \
    --ref-smiles Cpd18 \
    --n-samples 100 \
    --batch-size 25 \
    --condition shape esp \
    --output-dir "$OUTPUT_DIR/run2_shape_esp_Cpd18" \
    --seed 42

# -------------------------------------------------------------------
# Run 3: Grow molecules (add 2-6 heavy atoms) from Cpd18
#         Goal: explore extended P2 or larger P3 groups
# -------------------------------------------------------------------
python "$SCRIPT" \
    --ref-smiles Cpd18 \
    --n-samples 100 \
    --batch-size 25 \
    --condition all \
    --add-atoms-range 2 6 \
    --add-pharms-range 0 2 \
    --output-dir "$OUTPUT_DIR/run3_grow_Cpd18" \
    --seed 42

# -------------------------------------------------------------------
# Run 4: Scaffold-constrained (keep triazine-CN-morpholine warhead)
# -------------------------------------------------------------------
python "$SCRIPT" \
    --ref-smiles Cpd12 \
    --n-samples 100 \
    --batch-size 25 \
    --condition all \
    --scaffold-constraint \
    --output-dir "$OUTPUT_DIR/run4_scaffold_Cpd12" \
    --seed 42

echo ""
echo "Job finished: $(date)"
echo "Output: $OUTPUT_DIR"
