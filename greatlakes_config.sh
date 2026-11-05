#!/usr/bin/env bash
# =============================================================================
# greatlakes_config.sh — site paths. THE one place to edit.
#
# Sourced by greatlakes_env.sh (every job) and greatlakes_setup.sh (install).
# Previously both files carried their own copy of these paths, which meant a
# setup could succeed against one layout while the jobs looked in another.
#
# Defines paths only -- it does not activate anything, so it is safe to source
# from a login shell just to inspect the values.
# =============================================================================

ACCOUNT="maom"
UNIQNAME="$(whoami)"

# Turbo is persistent; scratch is purged (files untouched for ~60 days).
# Anything expensive to rebuild belongs on Turbo.
TURBO="/home/ymanasa/turbo/ymanasa"
SCRATCH="/scratch/${ACCOUNT}_root/${ACCOUNT}0/${UNIQNAME}/tgcpl-scratch-campaign"

# The upstream ShEPhERD-2 checkout (coleygroup/shepherd2), pip-installed -e.
REPO_DIR="$TURBO/opt/shepherd2"

# This repo: tgcpl_generate.py, filter_generated.py, the tgcpl/ package.
TGCPL_DIR="$TURBO/campaigns/tgcpl-martin-campaign/scripts/ShEPhERD-for-TgCPL"

# Conda env (--prefix, not a named env). Deliberately on Turbo rather than
# scratch: scratch is purged after ~60 days untouched, and an env is thousands
# of files that nothing "touches" in the sense that policy means, so losing it
# costs a ~15 min torch/CUDA reinstall.
#
# The enclosing directory name is incidental. conda, mamba and micromamba all
# write the same env format, and `conda activate <absolute prefix>` works on
# any of them regardless of which tool created it.
ENV_PREFIX="$TURBO/micromamba/envs/shepherd2"

# Generated molecules. Large and reproducible, so scratch is the right home.
OUTPUT_ROOT="$SCRATCH/SHEP_out"

# Keep the HF checkpoint download out of the home quota.
export HF_HOME="$SCRATCH/intermediate_data/shep_hf_cache"

# Conda hook to source before `conda activate`.
#
# This must be set explicitly rather than derived from `conda info --base`.
# The login shell inherits CONDA_EXE=/mnt/turbo/.../mambaforge/bin/conda from
# an old conda init block, but Turbo is mounted at /nfs/turbo now, so that path
# does not exist and the conda shell function dies with
# "No such file or directory" on any `conda activate` -- including inside a
# batch job, where it is a silent 0-molecule run.
CONDA_SH="$TURBO/miniconda3/etc/profile.d/conda.sh"


# Validate the paths that must already exist. Call after sourcing.
# Fails loudly at submit time instead of 20 minutes into a GPU allocation.
tgcpl_check_paths() {
    local missing=0 p
    for p in "$CONDA_SH" "$REPO_DIR" "$TGCPL_DIR"; do
        [ -e "$p" ] || { echo "MISSING: $p" >&2; missing=1; }
    done
    if [ "${1:-}" != "--pre-install" ]; then
        [ -d "$ENV_PREFIX" ] || {
            echo "MISSING: $ENV_PREFIX (run greatlakes_setup.sh)" >&2
            missing=1
        }
    fi
    [ "$missing" -eq 0 ] || return 1
}
