#!/usr/bin/env bash
# =============================================================================
# local_env.sh — shared environment setup for local (Mac, CPU) runs.
#   source local_env.sh
# The laptop counterpart of greatlakes_env.sh. Edit paths here once.
# =============================================================================

CONDA_ENV="shepherd"
TGCPL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$TGCPL_DIR/.." && pwd)/shepherd2"
OUTPUT_ROOT="$TGCPL_DIR/local_outputs"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Activating the env is what puts `xtb` on PATH; calling the env's python
# directly gives FileNotFoundError: 'xtb'.
if ! command -v xtb >/dev/null; then
    echo "FAIL: xtb not on PATH — is the '$CONDA_ENV' env built?" >&2
    return 1 2>/dev/null || exit 1
fi

# CPU inference: let torch and xtb share the cores without oversubscribing.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "$OUTPUT_ROOT"
