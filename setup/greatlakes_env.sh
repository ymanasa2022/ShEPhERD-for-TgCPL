#!/usr/bin/env bash
# Activate the environment. Sourced by every job script; paths live in
# greatlakes_config.sh.

_GL_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_GL_ENV_DIR/greatlakes_config.sh"

if ! tgcpl_check_paths; then
    echo "greatlakes_env.sh: environment is not set up as configured." >&2
    echo "  Check greatlakes_config.sh, or run: bash greatlakes_setup.sh" >&2
    return 1 2>/dev/null || exit 1
fi

# Source the hook explicitly: the inherited CONDA_EXE points at a dead mount,
# so `conda info --base` resolves to nothing usable.
source "$CONDA_SH"
conda activate "$ENV_PREFIX"

# Activating is what puts xtb on PATH; without it generation dies mid-run.
if ! command -v xtb >/dev/null; then
    echo "greatlakes_env.sh: xtb not on PATH after activating $ENV_PREFIX" >&2
    echo "  The env is missing its conda-forge xtb package." >&2
    return 1 2>/dev/null || exit 1
fi

# xtb spawns threads per worker; let Slurm's CPU allocation govern.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$OUTPUT_ROOT" "$HF_HOME" "$SCRATCH/logs"
