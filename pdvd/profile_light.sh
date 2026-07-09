#!/bin/bash
# gperftools CPU-profile (or heap-profile) one event of the PDVD light chain.
# Usage: ./profile_light.sh <run> <event> [out.prof]
# Pre-compiles the cfg with wcsonnet (SIGPROF kills gojsonnet GC otherwise)
# and points output_dir at scratch so profiling never clobbers the real
# work/ archives.
# Env overrides:
#   RAW_FILE - input file (default: first input_data_light match, as run_light_evt.sh)
#   PROFLIB  - LD_PRELOAD lib (default libtcmalloc_and_profiler)
#   OUTDIR   - where wire-cell writes outputs (default scratch)
#   HEAPOUT  - if set, do tcmalloc HEAPPROFILE instead of CPU profile
set -e
PDVD_DIR=$(cd "$(dirname "$0")" && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}

RUN=${1:?run} EVENT=${2:?event}
RUN_PADDED=$(printf "%06d" "$RUN")
OUT=${3:-/home/xqian/tmp/light_pdvd_${RUN_PADDED}_${EVENT}.prof}

if [ -z "$RAW_FILE" ]; then
    RAW_FILE=$(ls "$PDVD_DIR"/input_data_light/np02vd_raw_run${RUN_PADDED}_*_rawwf.root 2>/dev/null | head -1)
fi
[ -f "$RAW_FILE" ] || { echo "no raw file (set RAW_FILE)" >&2; exit 1; }

OUTDIR=${OUTDIR:-/home/xqian/tmp/prof_light_pdvd_${RUN_PADDED}_${EVENT}}
mkdir -p "$OUTDIR"

CFG=$OUTDIR/.wct-light.json
wcsonnet \
    -A input_file="$RAW_FILE" \
    -A output_dir="$OUTDIR" \
    -S run="$RUN" \
    -S event="$EVENT" \
    -o "$CFG" \
    "$PDVD_DIR/wct-light-reco.jsonnet"

PROFLIB=${PROFLIB:-/usr/lib/x86_64-linux-gnu/libtcmalloc_and_profiler.so.4}
if [ -n "$HEAPOUT" ]; then
    LD_PRELOAD="$PROFLIB" HEAPPROFILE="$HEAPOUT" \
    wire-cell -l stderr -L info -c "$CFG"
    OUT="$HEAPOUT"
else
    LD_PRELOAD="$PROFLIB" \
    CPUPROFILE="$OUT" CPUPROFILE_FREQUENCY=1000 \
    wire-cell -l stderr -L info -c "$CFG"
fi

echo "profile -> $OUT"
echo "view: google-pprof --text $(which wire-cell) $OUT | head -40"
