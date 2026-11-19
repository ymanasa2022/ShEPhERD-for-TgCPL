#!/usr/bin/env bash
# Site paths. The one place to edit.

ACCOUNT="maom"
UNIQNAME="$(whoami)"
TURBO="/home/ymanasa/turbo/ymanasa"
SCRATCH="/scratch/${ACCOUNT}_root/${ACCOUNT}0/${UNIQNAME}/tgcpl-scratch-campaign"

# Upstream coleygroup/shepherd2, pip-installed -e.
REPO_DIR="$TURBO/opt/shepherd2"

# This repo's root, not its scripts/ subdirectory.
TGCPL_DIR="$TURBO/campaigns/tgcpl-martin-campaign/scripts/ShEPhERD-for-TgCPL"
ENV_PREFIX="$TURBO/micromamba/envs/shepherd2"

SCRIPTS_DIR="$TGCPL_DIR/scripts"
GENERATE_DIR="$SCRIPTS_DIR/generate"
TGCPL_PKG_DIR="$SCRIPTS_DIR/tgcpl"
TESTS_DIR="$TGCPL_DIR/tests"

OUTPUT_ROOT="$SCRATCH/SHEP_out"

export HF_HOME="$SCRATCH/intermediate_data/shep_hf_cache"

# Boltz-2 gets a separate env on purpose: it pins its own torch, which would
# break torch_cluster/torch_scatter in the shepherd env.
BOLTZ_ENV="$TURBO/micromamba/envs/boltz2"
BOLTZ_ROOT="$SCRATCH/boltz"          # boltz_in/, boltz_out/, msa/
export BOLTZ_CACHE="$SCRATCH/intermediate_data/boltz_cache"   # model weights

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
