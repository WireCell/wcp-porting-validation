#!/bin/bash
# doc 85 round 2 -- build the six Bee zips from the prod0830 pick lists.
# LOCAL ONLY: uploading is a separate, owner-authorised step (CLAUDE.md
# escalation rule 6), so this script deliberately does NOT call
# upload-to-bee.sh.  Run it, check the zips, then upload by hand.
#
# Usage: scripts/bee/build_d85r2_bee.sh [pick-dir] [out-dir]
set -eu
SX=$(cd "$(dirname "$0")/../.." && pwd -P)
cd "$SX"
PICKS=${1:-docs/85r2_dists}
OUT=${2:-bee/d85r2}
mkdir -p "$OUT"

# the three numu-sample sets: events span mcp1k and mcp2k, so both roots are
# offered and make_pr_bee.py takes the first that has each event.
for c in cosmiclike nuelike neither; do
    [ -s "$PICKS/d85r2-$c.txt" ] || { echo "skip $c (empty pick list)"; continue; }
    python3 scripts/bee/make_pr_bee.py \
        -q work-mcp1k-grp0825   -q work-mcp2k-grp0825 \
        -p work-mcp1k-prod0830  -p work-mcp2k-prod0830 \
        -o "$OUT/d85r2-$c.zip" $(cat "$PICKS/d85r2-$c.txt")
done

# nueCC events FAILING the nue cut: the set deliberately includes rows with no
# PR evaluation at all -- they fail the selection too, and that is the point.
if [ -s "$PICKS/d85r2-nuecc-failnue.txt" ]; then
    python3 scripts/bee/make_pr_bee.py \
        -q work-nuecc48-grp0825 -p work-nuecc48-prod0830 --allow-unevaluated \
        -o "$OUT/d85r2-nuecc-failnue.zip" $(cat "$PICKS/d85r2-nuecc-failnue.txt")
fi

for c in ncpi0-numupass ncpi0-leak ncpi0-nearmiss; do
    [ -s "$PICKS/d85r2-$c.txt" ] || continue
    python3 scripts/bee/make_pr_bee.py \
        -q work-ncpi0-grp0825 -p work-ncpi0-prod0830 \
        -o "$OUT/d85r2-$c.zip" $(cat "$PICKS/d85r2-$c.txt")
done

ls -la "$OUT"/*.zip
