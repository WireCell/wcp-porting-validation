#!/usr/bin/env bash
# Doc 63: knob-off byte-identical gate between two campaign arms.
# Compares, for every nusel_evt<ID> present in BOTH roots:
#   - nusel-evt<ID>.tsv        (plain diff)
#   - mabc-pr.zip              (hash_archive.py member content hashes, M2)
#   - pctree-pr-evt<ID>.tar.gz (hash_archive.py member content hashes)
# Usage: compare_off.sh <ref-root> <test-root>
set -u
cd "$(dirname "$0")/.."
REF=${1:?usage: compare_off.sh <ref-root> <test-root>}
TEST=${2:?}
HASH=$(cd ../../abtest && pwd)/hash_archive.py
fail=0
n=0
for d in "$TEST"/nusel_evt*; do
    evt=${d#*nusel_evt}
    rd=$REF/nusel_evt$evt
    [ -d "$rd" ] || continue
    n=$((n+1))
    if ! diff -q "$rd/nusel-evt$evt.tsv" "$d/nusel-evt$evt.tsv" > /dev/null; then
        echo "TSV DIFF evt$evt"; fail=1
    fi
    for f in mabc-pr.zip "pctree-pr-evt$evt.tar.gz"; do
        # field 1 only: the tool's output lines end with the archive PATH,
        # which legitimately differs between arms
        h1=$(python3 "$HASH" "$rd/$f" 2>/dev/null | awk '{print $1}')
        h2=$(python3 "$HASH" "$d/$f" 2>/dev/null | awk '{print $1}')
        if [ "$h1" != "$h2" ]; then echo "ARCHIVE DIFF evt$evt $f"; fail=1; fi
    done
done
if [ "$fail" = 0 ]; then echo "GATE PASS: $n event(s) byte-identical ($REF vs $TEST)"
else echo "GATE FAIL"; exit 1; fi
