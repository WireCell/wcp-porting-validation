#!/bin/bash
# Determinism gate: run one filelist event N times (ASLR ON, i.e. production
# conditions) and count distinct Bee-zip content hashes.  1 distinct hash =
# deterministic.
#
# Usage: ./repeat_check.sh <idx> [n=4] [label=rep] [conc=4]
# Dirs: sweep/<label>_<k>/<idx>_<EV>/
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
python3 "$ABTEST/hash_archive.py" "${ZIPS[@]}" | tee /dev/stderr \
    | awk '{print $1}' | sort -u | wc -l \
    | awk -v n="${#ZIPS[@]}" '{print "distinct hashes: "$1" of "n" runs" ($1==1 ? "  DETERMINISTIC" : "  NONDETERMINISTIC")}'
