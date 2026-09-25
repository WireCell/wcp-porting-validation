#!/bin/bash
# doc pdvd/118: convert the run-39305 kaon-candidate art files (jjo,
# pdvd/input_data_kaon -> /nfs/data/1/jjo/data/pdvd_kaon_candidates_full)
# into the standalone-chain inputs:
#   input_kaon/run039305/evt_<artEvt>/protodune-orig-frames-anode{0..7}.tar.bz2
#   input_kaon/light/np02vd_raw_run039305_evt<artEvt>_rawwf.root
# Both lar jobs run in the SL7 apptainer (container_exec.sh).
# Usage: ./convert_kaon.sh [evt ...]      (default: every file in MANIFEST order)
# Parallel cap: KAON_MAX_JOBS (default 5).
KDIR=$(cd "$(dirname "$0")" && pwd)
PDVD=$(dirname "$KDIR")
SRC=$PDVD/input_data_kaon
DST=$PDVD/input_kaon
LAR=$DST/_lar
MAXJ=${KAON_MAX_JOBS:-5}
if [ $# -gt 0 ]; then EVTS=("$@"); else
    mapfile -t EVTS < <(ls $SRC/pdvd_kaon_run39305_evt*.root | sed -E 's/.*_evt([0-9]+)\.root/\1/')
fi
one() {
    local e=$1 f=$SRC/pdvd_kaon_run39305_evt$1.root rc
    $KDIR/container_exec.sh $KDIR/_orig_inner.sh $f $LAR/orig_$e > $LAR/orig_$e.out 2>&1; rc=$?
    [ $rc -eq 0 ] || { echo "evt $e orig rc=$rc"; return $rc; }
    mkdir -p $DST/run039305/evt_$e
    mv $LAR/orig_$e/protodune-orig-frames-anode*.tar.bz2 $DST/run039305/evt_$e/
    $KDIR/container_exec.sh $KDIR/_light_inner.sh $f $LAR/light_$e np02vd_raw_run039305_evt$e > $LAR/light_$e.out 2>&1; rc=$?
    [ $rc -eq 0 ] || { echo "evt $e light rc=$rc"; return $rc; }
    mkdir -p $DST/light
    cp $LAR/light_$e/np02vd_raw_run039305_evt${e}_rawwf.root $DST/light/
    echo "evt $e ok"
}
mkdir -p $LAR
nfail=0
for e in "${EVTS[@]}"; do
    while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do wait -n || nfail=$((nfail+1)); done
    one $e &
done
while [ $(jobs -rp | wc -l) -gt 0 ]; do wait -n || nfail=$((nfail+1)); done
echo "convert_kaon: ${#EVTS[@]} events, $nfail failed"
exit $nfail
