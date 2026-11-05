#!/usr/bin/env bash
# =============================================================================
# greatlakes_config.sh — site paths. THE one place to edit.
#
# Sourced by greatlakes_env.sh (every job) and greatlakes_setup.sh (install).
# =============================================================================

ACCOUNT="maom"
UNIQNAME="$(whoami)"
TURBO="/home/ymanasa/turbo/ymanasa"
SCRATCH="/scratch/${ACCOUNT}_root/${ACCOUNT}0/${UNIQNAME}/tgcpl-scratch-campaign"

# The upstream ShEPhERD-2 checkout (coleygroup/shepherd2), pip-installed -e.
REPO_DIR="$TURBO/opt/shepherd2"

# This repo (its root, not its scripts/ subdirectory).
TGCPL_DIR="$TURBO/campaigns/tgcpl-martin-campaign/scripts/ShEPhERD-for-TgCPL"
ENV_PREFIX="$TURBO/micromamba/envs/shepherd2"

# Repo layout, stated once. The Python entry points fix sys.path themselves.
SCRIPTS_DIR="$TGCPL_DIR/scripts"          # holds the importable tgcpl package
GENERATE_DIR="$SCRIPTS_DIR/generate"      # filter_generated.py, summarize_run.py
TGCPL_PKG_DIR="$SCRIPTS_DIR/tgcpl"        # tgcpl_generate.py + the package
TESTS_DIR="$TGCPL_DIR/tests"

OUTPUT_ROOT="$SCRATCH/SHEP_out"

export HF_HOME="$SCRATCH/intermediate_data/shep_hf_cache"

CONDA_SH="$TURBO/miniconda3/etc/profile.d/conda.sh"

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
