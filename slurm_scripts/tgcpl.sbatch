#!/usr/bin/env bash
#SBATCH --job-name=tgcpl_shepherd2
#SBATCH --output=logs/shepherd2_%j.out
#SBATCH --error=logs/shepherd2_%j.err
#SBATCH --partition=gpu         # <-- change to your GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8       # for xTB workers
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Generic SLURM job for TgCPL generation: hardcoded paths, conda by name.
# On Great Lakes use greatlakes_tgcpl.sbatch instead. Edit the paths below.

TGCPL_DIR="$HOME/ShEPhERD-for-TgCPL"     # this repo
SCRIPT="$TGCPL_DIR/scripts/tgcpl/tgcpl_generate.py"
OUTPUT_DIR="$TGCPL_DIR/tgcpl_outputs"

module load cuda/12.6 2>/dev/null || true   # adjust to your HPC module name
conda activate shepherd2

cd "$TGCPL_DIR"
mkdir -p logs

echo "Job started: $(date)"
echo "Node: $HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# Run 1: all leads, shape + ESP + pharmacophores, 50 samples each.
python "$SCRIPT" \
    --ref-smiles all \
    --n-samples 50 \
    --batch-size 25 \
    --condition all \
    --output-dir "$OUTPUT_DIR/run1_all_leads" \
    --seed 42

# Run 2: no pharmacophore constraint -> more structural diversity.
python "$SCRIPT" \
    --ref-smiles Cpd18 \
    --n-samples 100 \
    --batch-size 25 \
    --condition shape esp \
    --output-dir "$OUTPUT_DIR/run2_shape_esp_Cpd18" \
    --seed 42

# Run 3: grow by 2-6 heavy atoms to explore extended P2 / larger P3.
python "$SCRIPT" \
    --ref-smiles Cpd18 \
    --n-samples 100 \
    --batch-size 25 \
    --condition all \
    --add-atoms-range 2 6 \
    --add-pharms-range 0 2 \
    --output-dir "$OUTPUT_DIR/run3_grow_Cpd18" \
    --seed 42

# Run 4: scaffold-constrained, keeping the triazine-CN-morpholine warhead.
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
