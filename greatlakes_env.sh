#!/usr/bin/env bash
# =============================================================================
# greatlakes_env.sh — shared environment setup, sourced by every job script.
#   source greatlakes_env.sh
# Edit paths here once; all job scripts follow.
# =============================================================================

ACCOUNT="maom99"
UNIQNAME="$(whoami)"
SCRATCH="/scratch/${ACCOUNT}_root/${ACCOUNT}/${UNIQNAME}"

REPO_DIR="$SCRATCH/shepherd2"
ENV_PREFIX="$SCRATCH/envs/shepherd2"
TGCPL_DIR="$SCRATCH/ShEPhERD-for-TgCPL"
OUTPUT_ROOT="$SCRATCH/tgcpl_outputs"

# Keep the HF checkpoint download out of the home quota.
export HF_HOME="$SCRATCH/hf_cache"

module load python3.11-anaconda 2>/dev/null || module load anaconda 2>/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# xtb spawns threads per worker; let Slurm's CPU allocation govern.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$OUTPUT_ROOT" "$SCRATCH/logs"
