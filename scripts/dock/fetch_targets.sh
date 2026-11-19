#!/usr/bin/env bash
# Pull the two mature protease sequences (TgCPL 3F75, HsCPL 5MAJ) from RCSB.
# The results are committed under data/targets/; this only refreshes them.
#
# 3F75 chain B is the propeptide, carrying an MRGSHHHHHH purification tag that
# would occlude the site if folded. Hence matching on the entity description
# rather than taking the first record.

set -euo pipefail

OUT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/data/targets}"
mkdir -p "$OUT_DIR"

# pdb_id | entity description to keep | output basename | expected length
TARGETS=(
    "3F75|Cathepsin L Protease|tgcpl_3f75|224"
    "5MAJ|Cathepsin L1|hscpl_5maj|220"
)

fetch_one() {
    local pdb="$1" want="$2" base="$3" expect="$4"
    local raw="$OUT_DIR/.${base}.rcsb.fasta"

    echo "[$pdb] fetching..."
    if ! curl -fsS --max-time 60 \
        "https://www.rcsb.org/fasta/entry/${pdb}/display" -o "$raw"; then
        echo "  ERROR: could not reach RCSB for $pdb" >&2
        return 1
    fi

    # awk, not grep -A: records wrap over several lines.
    awk -v want="$want" '
        /^>/ { keep = (index($0, want) > 0) }
        keep { print }
    ' "$raw" > "$OUT_DIR/${base}.fasta"
    rm -f "$raw"

    local n
    n=$(grep -vc '^>' "$OUT_DIR/${base}.fasta" || true)
    if [ "$n" -eq 0 ]; then
        echo "  ERROR: no entity matching '$want' in $pdb" >&2
        return 1
    fi

    local len
    # RCSB serves CRLF, so \r has to go too or every sequence reads one long.
    len=$(grep -v '^>' "$OUT_DIR/${base}.fasta" | tr -d '\n\r ' | wc -c | tr -d ' ')
    echo "  -> $OUT_DIR/${base}.fasta  (${len} aa, expected ${expect})"
    if [ "$len" != "$expect" ]; then
        echo "  WARNING: length changed from the committed sequence." >&2
        echo "  Check the entry before docking against it." >&2
    fi

    # Catalytic cysteine motif (Cys31 TgCPL / Cys25 HsCPL); missing = wrong construct.
    if ! grep -v '^>' "$OUT_DIR/${base}.fasta" | tr -d "\n\r" | grep -q 'C[GS]SCWAF'; then
        echo "  WARNING: catalytic C.SCWAF motif not found." >&2
    fi
}

for spec in "${TARGETS[@]}"; do
    IFS='|' read -r pdb want base expect <<< "$spec"
    fetch_one "$pdb" "$want" "$base" "$expect"
done

echo
echo "Done. Both sequences are the mature enzyme only -- no propeptide, no tag."
