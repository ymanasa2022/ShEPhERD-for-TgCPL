#!/usr/bin/env bash
# Environment for local (Mac, CPU) runs. Laptop counterpart of greatlakes_env.sh.

CONDA_ENV="shepherd"

# This file lives in setup/, so the repo root is one level up.
TGCPL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "$TGCPL_DIR/.." && pwd)/shepherd2"
OUTPUT_ROOT="$TGCPL_DIR/tests/local_outputs"

SCRIPTS_DIR="$TGCPL_DIR/scripts"
GENERATE_DIR="$SCRIPTS_DIR/generate"
TGCPL_PKG_DIR="$SCRIPTS_DIR/tgcpl"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Activating is what puts xtb on PATH; calling the env's python directly gives
# FileNotFoundError: 'xtb'.
if ! command -v xtb >/dev/null; then
    echo "FAIL: xtb not on PATH — is the '$CONDA_ENV' env built?" >&2
    return 1 2>/dev/null || exit 1
fi

# CPU inference: share cores between torch and xtb without oversubscribing.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "$OUTPUT_ROOT"
