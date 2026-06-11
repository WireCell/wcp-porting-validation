#!/bin/bash
# gperftools CPU-profile one anode of one event's imaging.
# Usage: ./profile_img.sh <pdhd|pdvd> <run> <evt> <anode> [out.prof]
# Pre-compiles the cfg with wcsonnet (SIGPROF kills gojsonnet GC otherwise).
# Env overrides:
#   PROFLIB  - LD_PRELOAD lib (default libtcmalloc_and_profiler: production
#              runs preload tcmalloc, so glibc-malloc profiles overstate
#              allocator costs -- round-4 lesson)
#   OUTDIR   - where wire-cell writes outputs (default the event WORKDIR;
#              point at scratch to avoid racing real outputs)
#   HEAPOUT  - if set, do tcmalloc HEAPPROFILE instead of CPU profile
set -e
AB_DIR=$(cd "$(dirname "$0")" && pwd)
BASE_DIR=$(dirname "$AB_DIR")
DET=${1:?det} RUN=${2:?run} EVT=${3:?evt} AI=${4:?anode}
OUT=${5:-/home/xqian/tmp/img_${DET}_${RUN}_${EVT}_a${AI}.prof}

RUN_PADDED=$(printf '%06d' "$((10#$RUN))")
WORKDIR="$BASE_DIR/$DET/work/${RUN_PADDED}_${EVT}"
cd "$BASE_DIR/$DET"
export WIRECELL_PATH=/nfs/data/1/xqian/toolkit-dev/toolkit/cfg:/nfs/data/1/xqian/toolkit-dev/wire-cell-data:${WIRECELL_PATH}

if [ "$DET" = "pdhd" ]; then
    PREFIX="$WORKDIR/protodunehd-sp-dnnroi-frames"
    NPY=$(tar tjf "${PREFIX}-anode${AI}.tar.bz2" | grep -m1 "^frame_gauss${AI}_")
    TMP=$(mktemp -d /home/xqian/tmp/prof.XXXXXX)
    tar xjf "${PREFIX}-anode${AI}.tar.bz2" -C "$TMP" "$NPY"
    NTICKS=$(python3 -c "import numpy as np; print(np.load('$TMP/$NPY', mmap_mode='r').shape[1])")
    rm -rf "$TMP"
    EXTRA=(-S "nticks=${NTICKS}")
else
    PREFIX="$WORKDIR/protodune-sp-frames"
    EXTRA=()
fi

CFG=/home/xqian/tmp/prof_cfg_${DET}_${RUN_PADDED}_${EVT}_a${AI}.json
wcsonnet -A "input_prefix=${PREFIX}" -S "anode_indices=[$AI]" \
         -A "output_dir=${OUTDIR:-$WORKDIR}" "${EXTRA[@]}" -o "$CFG" wct-img-all.jsonnet

PROFLIB=${PROFLIB:-/usr/lib/x86_64-linux-gnu/libtcmalloc_and_profiler.so.4}
if [ -n "$HEAPOUT" ]; then
    LD_PRELOAD="$PROFLIB" HEAPPROFILE="$HEAPOUT" \
    wire-cell -l stderr -L info -c "$CFG"
    OUT="$HEAPOUT"
else
    LD_PRELOAD="$PROFLIB" \
    CPUPROFILE="$OUT" CPUPROFILE_FREQUENCY=250 \
    wire-cell -l stderr -L info -c "$CFG"
fi

echo "profile -> $OUT"
echo "view: google-pprof --text $(which wire-cell) $OUT | head -40"
