#!/bin/bash
# gperftools CPU-profile clustering of one event.
# Usage: ./profile_clus.sh <pdhd|pdvd> <run> <evt> [out.prof]
# Pre-compiles the cfg with wcsonnet (SIGPROF kills gojsonnet GC otherwise).
set -e
AB_DIR=$(cd "$(dirname "$0")" && pwd)
BASE_DIR=$(dirname "$AB_DIR")
DET=${1:?det} RUN=${2:?run} EVT=${3:?evt}
OUT=${4:-/home/xqian/tmp/clus_${DET}_${RUN}_${EVT}.prof}

RUN_PADDED=$(printf '%06d' "$((10#$RUN))")
RUN_STRIPPED=$((10#$RUN))
WORKDIR="$BASE_DIR/$DET/work/${RUN_PADDED}_${EVT}"
cd "$BASE_DIR/$DET"
export WIRECELL_PATH=/nfs/data/1/xqian/toolkit-dev/toolkit/cfg:/nfs/data/1/xqian/toolkit-dev/wire-cell-data:${WIRECELL_PATH}

if [ "$DET" = "pdhd" ]; then
    ANODE_CODE="[0,1,2,3]"
    FIRST_CLUS=$(ls "$WORKDIR"/clusters-apa-apa*-ms-active.tar.gz | head -1)
else
    ANODE_CODE="[0,1,2,3,4,5,6,7]"
    FIRST_CLUS=$(ls "$WORKDIR"/clusters-apa-anode*-ms-active.tar.gz | head -1)
fi
EVENT_NO=$(tar tzf "$FIRST_CLUS" | head -1 | sed -E 's/.*cluster_([0-9]+)_.*/\1/')

CFG=/home/xqian/tmp/prof_clus_${DET}_${RUN_PADDED}_${EVT}.json
wcsonnet -A "input=${WORKDIR}" -S "anode_indices=${ANODE_CODE}" \
         -A "output_dir=${WORKDIR}" -S "run=${RUN_STRIPPED}" -S "subrun=0" \
         -S "event=${EVENT_NO}" -o "$CFG" wct-clustering.jsonnet

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libprofiler.so.0 \
CPUPROFILE="$OUT" CPUPROFILE_FREQUENCY=250 \
wire-cell -l stderr -L info -c "$CFG"

echo "profile -> $OUT"
echo "view: google-pprof --text $(which wire-cell) $OUT | head -40"
