#!/bin/bash
# doc pr/81 Phase B -- guard-in-loop checkpoint selection.
# For each fold k, replay-guard the arm's CP<E> on that fold's VAL events
# only (runs/hr-folds/fold<k>.txt), E in {0,2,5,8,11,14,17}.
# Usage: hr_guardsel.sh <arm>   (arm = hr1 | hr2 | hr3)
set -u
ARM=$1
cd "$(dirname "$0")"
OUT=runs/${ARM}-guardsel
mkdir -p "$OUT"
for k in 0 1 2 3 4 5; do
  (
    for E in 0 2 5 8 11 14 17; do
      W=runs/$ARM/fold$k/CP$E.pth
      [ -f "$W" ] || { echo "MISSING $W" > "$OUT/f$k-E$E.log"; continue; }
      OMP_NUM_THREADS=1 python3 calib_guard.py \
        --name ${ARM}-f$k-E$E --weights "$W" \
        --events-file runs/hr-folds/fold$k.txt --jobs 4 \
        > "$OUT/f$k-E$E.log" 2>&1
      echo "rc=$? f$k E$E"
    done
  ) &
done
wait
echo "== summary: guard-predicted delta per fold/epoch =="
for k in 0 1 2 3 4 5; do
  for E in 0 2 5 8 11 14 17; do
    L=$OUT/f$k-E$E.log
    [ -f "$L" ] || continue
    d=$(grep -oP 'delta \+?-?\d+' "$L" | head -1)
    r=$(grep 'raw net top1 ratio' "$L" | head -1 | grep -oP 'median \d+\.\d+' | head -1)
    ra=$(grep -oP 'reject->ACCEPT \d+' "$L" | head -1)
    echo "$ARM f$k E$E  $d  conf-top1-$r  $ra"
  done
done
