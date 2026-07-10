#!/bin/bash
# Byte-identity gate between two sweep labels.
#
# Usage: ./ab_check.sh <label> <base>
#   1. Bee zips: content hashes (hashes.txt, mtime-insensitive) must match.
#   2. track_com ROOT files are never byte-identical (timestamps/UUIDs), so
#      each side is validated through wire-cell-uboone-tagger-compare against
#      the prototype reference; the two verbose logs must then be identical.
# Exit 0 only if both gates pass for every event.
set -u
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
QLPORT=$(dirname "$SCRIPTS")

LABEL=${1:?usage: ab_check.sh <label> <base>}
BASE=${2:?usage: ab_check.sh <label> <base>}
LDIR="$SCRIPTS/sweep/$LABEL" BDIR="$SCRIPTS/sweep/$BASE"

COMPARE=$(command -v wire-cell-uboone-tagger-compare || true)
[ -n "$COMPARE" ] || COMPARE="$QLPORT/../../toolkit/build/root/wire-cell-uboone-tagger-compare"
[ -x "$COMPARE" ] || { echo "cannot find wire-cell-uboone-tagger-compare" >&2; exit 2; }

fail=0

# -- gate 1: Bee zip content hashes --
if diff <(awk '{print $1"  "$3}' "$BDIR/hashes.txt") \
        <(awk '{print $1"  "$3}' "$LDIR/hashes.txt") > /dev/null; then
    echo "ZIPS: $(wc -l < "$LDIR/hashes.txt")/$(wc -l < "$BDIR/hashes.txt") content-identical"
else
    echo "ZIPS: MISMATCH"
    diff <(awk '{print $1"  "$3}' "$BDIR/hashes.txt") \
         <(awk '{print $1"  "$3}' "$LDIR/hashes.txt") | grep '^[<>]' | head -20
    fail=1
fi

# -- gate 2: tagger-compare logs --
tagger_log() {  # <side-dir-for-event> <sr> <ev>
    local d=$1 sr=$2 ev=$3
    local log="$d/tagger_${ev}.log"
    if [ ! -s "$log" ]; then
        local proto="$QLPORT/prototype/nue_5384_${sr}_${ev}.root"
        [ -f "$proto" ] || { echo "no-proto"; return; }
        "$COMPARE" -p "$proto" -t "$d/track_com_5384_${ev}.root" -v > "$log" 2>&1
    fi
    echo "$log"
}

nsame=0 ndiff=0 nskip=0
running=0
: > "$LDIR/ab_check_${BASE}.log"
for d in "$LDIR"/*_*/; do
    b=$(basename "$d")
    ev=${b#*_} idx=${b%%_*}
    sr=$(sed -nE 's/.* subrun=([0-9]+).*/\1/p' "$d/meta.txt")
    (
        l1=$(tagger_log "$d" "$sr" "$ev")
        l2=$(tagger_log "$BDIR/$b" "$sr" "$ev")
        if [ "$l1" = "no-proto" ] || [ "$l2" = "no-proto" ]; then
            echo "TAGGER idx=$idx ev=$ev: SKIP (no prototype file)"
        elif cmp -s "$l1" "$l2"; then
            echo "TAGGER idx=$idx ev=$ev: identical"
        else
            echo "TAGGER idx=$idx ev=$ev: DIFF"
        fi
    ) >> "$LDIR/ab_check_${BASE}.log" &
    running=$((running+1))
    if [ "$running" -ge 6 ]; then wait -n; running=$((running-1)); fi
done
wait
nsame=$(grep -c ': identical' "$LDIR/ab_check_${BASE}.log" || true)
ndiff=$(grep -c ': DIFF' "$LDIR/ab_check_${BASE}.log" || true)
nskip=$(grep -c ': SKIP' "$LDIR/ab_check_${BASE}.log" || true)
echo "TAGGER: identical=$nsame diff=$ndiff skip=$nskip (log: $LDIR/ab_check_${BASE}.log)"
[ "$ndiff" -eq 0 ] || fail=1

exit $fail
