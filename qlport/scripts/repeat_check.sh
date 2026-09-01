#!/bin/bash
# Determinism gate: run one filelist event N times (ASLR ON, i.e. production
# conditions) and count distinct content hashes.  1 distinct hash =
# deterministic.
#
# TWO gates, because one is not enough (doc 90 sec 4 and 7):
#   1. Bee zip      -- the CLUSTERING output.
#   2. track_com    -- the TAGGER output (T_kine, T_tagger, T_rec_charge).
# uBooNE event 5384-136-6805 is bistable in gate 2 while gate 1 stays
# byte-identical every run: four kine_pio_* variables swing between two states
# (kine_pio_angle 14.81 vs 109.51).  For as long as this script checked only
# the Bee zip it reported "DETERMINISTIC" on that event.
#
# Gate 2 hashes the MULTISET of rows, not the sequence: T_rec_charge emits its
# per-point rows in a run-dependent order (same rows, permuted), so a
# sequence-sensitive hash flags every event and detects nothing.  Row order is
# reported separately rather than hidden -- see hash_root_trees.py.
#
# Usage: ./repeat_check.sh <idx> [n=4] [label=rep] [conc=4]
# Dirs: sweep/<label>_<k>/<idx>_<EV>/
# Exit: 0 if BOTH gates report 1 distinct hash, 1 otherwise.
set -u
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
QLPORT=$(dirname "$SCRIPTS")
ABTEST=$(cd "$QLPORT/../abtest" && pwd)

IDX=${1:?usage: repeat_check.sh <idx> [n] [label] [conc]}
N=${2:-4}
LABEL=${3:-rep}
CONC=${4:-4}

running=0
for ((k=1; k<=N; k++)); do
    ASLR=1 "$SCRIPTS/run_one.sh" "$IDX" "${LABEL}_${k}" > /dev/null 2>&1 &
    running=$((running+1))
    if [ "$running" -ge "$CONC" ]; then wait -n; running=$((running-1)); fi
done
wait

ZIPS=()
for ((k=1; k<=N; k++)); do
    z=$(ls "$SCRIPTS/sweep/${LABEL}_${k}/${IDX}_"*/mabc_${IDX}.zip 2>/dev/null | head -1)
    [ -n "$z" ] && ZIPS+=("$z")
done
# NOT `| tee /dev/stderr`: under `repeat_check.sh ... > log 2>&1`, /dev/stderr
# is /proc/self/fd/2, which tee REOPENS at offset 0 -- the second such pipeline
# then overwrites the first one's output in the log.  That silently ate gate 1
# entirely the first time both gates ran.  Capture, print, count instead.
zout=$(python3 "$ABTEST/hash_archive.py" "${ZIPS[@]}")
printf '%s\n' "$zout"
nzip=$(printf '%s\n' "$zout" | awk '{print $1}' | sort -u | wc -l)
awk -v d="$nzip" -v n="${#ZIPS[@]}" 'BEGIN{print "distinct hashes: "d" of "n" runs" (d==1 ? "  DETERMINISTIC" : "  NONDETERMINISTIC")}'

# -- gate 2: tagger output (doc 90 sec 7) --
ROOTS=()
for ((k=1; k<=N; k++)); do
    r=$(ls "$SCRIPTS/sweep/${LABEL}_${k}/${IDX}_"*/track_com_*.root 2>/dev/null | head -1)
    [ -n "$r" ] && ROOTS+=("$r")
done
nroot=0
if [ "${#ROOTS[@]}" -gt 0 ]; then
    rout=$(python3 "$SCRIPTS/hash_root_trees.py" "${ROOTS[@]}")
    printf '%s\n' "$rout"
    nroot=$(printf '%s\n' "$rout" | awk '{print $1}' | sort -u | wc -l)
    awk -v d="$nroot" -v n="${#ROOTS[@]}" 'BEGIN{print "distinct tagger hashes: "d" of "n" runs" (d==1 ? "  DETERMINISTIC" : "  NONDETERMINISTIC")}'
else
    echo "distinct tagger hashes: (no track_com_*.root found)"
fi

# A PASS requires BOTH.  Quoting only the Bee-zip result is what hid 6805.
[ "$nzip" = "1" ] && { [ "${#ROOTS[@]}" -eq 0 ] || [ "$nroot" = "1" ]; }
