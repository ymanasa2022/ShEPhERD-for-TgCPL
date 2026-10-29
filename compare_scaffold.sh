#!/usr/bin/env bash
# =============================================================================
# compare_scaffold.sh — same lead, same seed, two scaffold cores.
#
#   ./compare_scaffold.sh [N_SAMPLES] [REF]      # defaults: 6 Cpd18
#
# The only difference between the two runs is which atoms are held fixed:
#
#   warhead_morpholine   nitrile + triazine + P1 morpholine   (14/37 atoms on Cpd18)
#   warhead              nitrile + triazine only              ( 8/37)
#
# Freeing the morpholine should let S1 vary. Whether it actually does — and at
# what cost to the 3D scores — is what this is for. Same seed both sides, so
# differences are the constraint, not sampling noise.
# =============================================================================

set -euo pipefail

N="${1:-6}"
REF="${2:-Cpd18}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/local_env.sh"
cd "$SCRIPT_DIR"

for CORE in warhead_morpholine warhead; do
    OUT="$OUTPUT_ROOT/scaffold_${CORE}_${REF}"
    echo
    echo "############################################################"
    echo "#  core=$CORE  ref=$REF  n=$N"
    echo "############################################################"
    python tgcpl_generate.py \
        --ref-smiles "$REF" \
        --n-samples "$N" \
        --batch-size 3 \
        --device cpu \
        --scaffold-constraint \
        --scaffold-core "$CORE" \
        --output-dir "$OUT" \
        --seed 42
done

echo
echo "############################################################"
echo "#  SUMMARIES"
echo "############################################################"
for CORE in warhead_morpholine warhead; do
    echo
    echo ">>> core=$CORE"
    python summarize_run.py "$OUTPUT_ROOT/scaffold_${CORE}_${REF}"
done
