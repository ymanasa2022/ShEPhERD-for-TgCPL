#!/usr/bin/env bash
# =============================================================================
# local_smoke_test.sh — prove the install works on this Mac, on CPU.
#
#   ./local_smoke_test.sh            # stages 1-2 (~5 min)
#   ./local_smoke_test.sh tgcpl      # stage 3 only: one real lead, 1 sample (~4 min)
#   ./local_smoke_test.sh all        # everything (~9 min)
#
# The laptop counterpart of greatlakes_smoke_test.sh. Run this before pushing
# anything to Great Lakes: it catches env breakage without burning GPU hours.
# =============================================================================

set -euo pipefail

STAGE="${1:-default}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/local_env.sh"

echo "=== Local smoke test ==="
echo "host    : $(hostname)"
echo "started : $(date)"
echo "env     : $CONDA_ENV   ($(python -c 'import sys; print(sys.version.split()[0])'))"
echo "stage   : $STAGE"
echo

run_stage() { [[ "$STAGE" == "all" || "$STAGE" == "$1" || ( "$STAGE" == "default" && "$2" == "yes" ) ]]; }

# --- 1. imports and tools ----------------------------------------------------
if run_stage env yes; then
echo "--- 1. environment ---"
python - <<'PY'
import shutil, sys
import torch, rdkit
import shepherd
print("torch          :", torch.__version__)
print("rdkit          :", rdkit.__version__)
print("shepherd       :", getattr(shepherd, "__version__", "editable"))
print("mps available  :", torch.backends.mps.is_available())
print("xtb on PATH    :", shutil.which("xtb") or "*** MISSING ***")
if not shutil.which("xtb"):
    sys.exit("FAIL: xtb missing from PATH")
PY
echo "PASS: environment"
echo
fi

# --- 2. shepherd2 end-to-end, benzene ----------------------------------------
if run_stage benzene yes; then
echo "--- 2. shepherd2 end-to-end (benzene, 3 samples, CPU ~3-4 min) ---"
python "$REPO_DIR/examples/basic_inference/basic_inference_test.py" \
    --smiles "c1ccccc1" \
    --output-dir "$OUTPUT_ROOT/smoke_benzene" \
    --batch-size 3
echo "PASS: benzene (expect 3/3 valid)"
echo
fi

# --- 3. a real TgCPL lead ----------------------------------------------------
# Bigger than benzene but still a few minutes on CPU. This is the correctness
# check for tgcpl_generate.py itself: conformer embedding, xTB profile, generation.
if run_stage tgcpl no; then
echo "--- 3. tgcpl_generate.py (Cpd18, 1 sample, CPU ~3-4 min) ---"
cd "$TGCPL_DIR"
python tgcpl_generate.py \
    --ref-smiles Cpd18 \
    --n-samples 1 \
    --batch-size 1 \
    --device cpu \
    --output-dir "$OUTPUT_ROOT/smoke_tgcpl" \
    --seed 42
echo "PASS: tgcpl_generate.py"
echo
fi

echo "=== finished: $(date) ==="
echo "Outputs under: $OUTPUT_ROOT"
