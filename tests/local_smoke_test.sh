#!/usr/bin/env bash
# Prove the install works on this Mac, on CPU. Run before pushing to Great
# Lakes: catches env breakage without burning GPU hours.
# Stage arg: default (1-2), tgcpl (stage 3 only), all.

set -euo pipefail

STAGE="${1:-default}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/setup/local_env.sh"

echo "=== Local smoke test ==="
echo "host    : $(hostname)"
echo "started : $(date)"
echo "env     : $CONDA_ENV   ($(python -c 'import sys; print(sys.version.split()[0])'))"
echo "stage   : $STAGE"
echo

run_stage() { [[ "$STAGE" == "all" || "$STAGE" == "$1" || ( "$STAGE" == "default" && "$2" == "yes" ) ]]; }

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

if run_stage benzene yes; then
echo "--- 2. shepherd2 end-to-end (benzene, 3 samples, CPU ~3-4 min) ---"
python "$REPO_DIR/examples/basic_inference/basic_inference_test.py" \
    --smiles "c1ccccc1" \
    --output-dir "$OUTPUT_ROOT/smoke_benzene" \
    --batch-size 3
echo "PASS: benzene (expect 3/3 valid)"
echo
fi

# The correctness check for tgcpl_generate.py itself: conformer embedding,
# xTB profile, generation.
if run_stage tgcpl no; then
echo "--- 3. tgcpl_generate.py (Cpd18, 1 sample, CPU ~3-4 min) ---"
cd "$TGCPL_DIR"
python "$TGCPL_PKG_DIR/tgcpl_generate.py" \
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
