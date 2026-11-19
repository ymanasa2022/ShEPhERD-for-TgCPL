#!/usr/bin/env bash
# Same lead, same seed, two scaffold cores: warhead_morpholine holds the
# nitrile + triazine + P1 morpholine fixed, warhead only the first two. Same
# seed both sides, so differences are the constraint, not sampling noise.

set -euo pipefail

N="${1:-6}"
REF="${2:-Cpd18}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/setup/local_env.sh"
cd "$REPO_ROOT"

for CORE in warhead_morpholine warhead; do
    OUT="$OUTPUT_ROOT/scaffold_${CORE}_${REF}"
    echo
    echo "############################################################"
    echo "#  core=$CORE  ref=$REF  n=$N"
    echo "############################################################"
    python "$TGCPL_PKG_DIR/tgcpl_generate.py" \
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
    python "$GENERATE_DIR/summarize_run.py" "$OUTPUT_ROOT/scaffold_${CORE}_${REF}"
done
