#!/bin/bash
# doc pr/89 Arm B -- guard-in-loop checkpoint selection, round-4 edition of
# hr_guardsel.sh (the pr/81 script is kept unchanged as that round's record;
# this one passes the round-4 --tags/--arm-roots of doc pr/89 sec 11.0 and
# the hr4-folds val lists).
# For each fold k, replay-guard the arm's CP<E> on that fold's VAL events
# only (runs/hr4-folds/fold<k>.txt), E in {0,2,5,8,11,14,17}.
# Usage: hr4_guardsel.sh <arm>   (arm = hr4b | hr4b-hum | hr4b-maxa)
set -u
ARM=$1
cd "$(dirname "$0")"
T="--tags vtxscan-mcp2k vtxscan-mcp2k-auto vtxscan-harv3-nuecc48 vtxscan-harv3-ncpi0 vtxscan-harv3-mcp1k vtxscan-harv3-delta"
R="--arm-roots vtxscan-mcp2k=work-mcp2k-harv3 vtxscan-mcp2k-auto=work-mcp2k-harv3 vtxscan-harv3-nuecc48=work-nuecc48-harv3 vtxscan-harv3-ncpi0=work-ncpi0-harv3 vtxscan-harv3-mcp1k=work-mcp1k-harv3 vtxscan-harv3-delta=@arm"
OUT=runs/${ARM}-guardsel
mkdir -p "$OUT"
for k in 0 1 2 3 4 5; do
  (
    for E in 0 2 5 8 11 14 17; do
      W=runs/$ARM/fold$k/CP$E.pth
      [ -f "$W" ] || { echo "MISSING $W" > "$OUT/f$k-E$E.log"; continue; }
      OMP_NUM_THREADS=1 python3 calib_guard.py \
        --name ${ARM}-f$k-E$E --weights "$W" $T $R \
        --events-file runs/hr4-folds/fold$k.txt --jobs 4 \
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
