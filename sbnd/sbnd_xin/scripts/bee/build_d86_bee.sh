#!/bin/bash
# doc 86 -- build the nine video Bee zips from the prod0830 pick lists.
#
# One zip per category, 2-3 events each, so that Bee's event/<i>/ index (which
# follows UPLOAD ORDER, not event id) is trivially readable from the pick list.
# The -q/-p roots are ordered so the INTENDED sample is tried first for each
# set; scripts/bee/verify_d86_bee.py then re-checks run/subrun per member, so
# an event-id collision across samples cannot pass silently.
#
# LOCAL ONLY: uploading is a separate owner-authorised step (CLAUDE.md
# escalation rule 6).  Run this, verify, then upload.
#
# Usage: scripts/bee/build_d86_bee.sh [pick-dir] [out-dir]
set -eu
SX=$(cd "$(dirname "$0")/../.." && pwd -P)
cd "$SX"
PICKS=${1:-docs/86_video}
OUT=${2:-bee/d86}
mkdir -p "$OUT"

Q=(-q work-nuecc48-grp0825 -q work-ncpi0-grp0825 -q work-mcp1k-grp0825 -q work-mcp2k-grp0825)
P=(-p work-nuecc48-prod0830 -p work-ncpi0-prod0830 -p work-mcp1k-prod0830 -p work-mcp2k-prod0830)

for c in nuecc numucc-cathode numucc ccpi0 ncpi0 cosmiclike multinu fail-busy fail-em; do
    L="$PICKS/d86-set-$c.txt"
    [ -s "$L" ] || { echo "skip $c (no pick list)"; continue; }
    echo "=== $c: $(tr '\n' ' ' < "$L")"
    # fail-* sets deliberately include events with no PR evaluation at all --
    # failing the selection IS the point of showing them.
    EXTRA=""
    case "$c" in fail-*) EXTRA="--allow-unevaluated" ;; esac
    python3 scripts/bee/make_pr_bee.py "${Q[@]}" "${P[@]}" $EXTRA \
        -o "$OUT/d86-$c.zip" $(cat "$L")
done
ls -la "$OUT"/*.zip
