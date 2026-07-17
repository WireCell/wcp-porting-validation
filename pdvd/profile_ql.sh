#!/bin/bash
# gperftools CPU-profile (or heap-profile) one event's clustering+QLMatching.
#   Usage: ./profile_ql.sh <run> <idx> [out.prof]
# Two steps:
#   1. stage a throwaway tag and run run_clus_evt.sh once with
#      PDVD_KEEP_CFG=1 -- this computes all runner -S args (offsets, readout
#      ticks, QL knobs) and leaves the compiled JSON in the tag dir
#      (wcsonnet precompile: SIGPROF corrupts the gojsonnet GC, CLAUDE.md M17).
#      This unprofiled run also leaves the headline wall/RSS numbers
#      (clus_resource_*.txt) untainted by the profiler preload.
#   2. rerun wire-cell on that JSON under libtcmalloc_and_profiler.
# The tag dir is a throwaway (profql_*), not a record; outputs of step 2
# overwrite step 1's inside it.
# Env overrides:
#   TAG      - work tag (default profql_<run6>_<idx>; refuses to reuse)
#   PROFLIB  - LD_PRELOAD lib (default libtcmalloc_and_profiler)
#   HEAPOUT  - if set, tcmalloc HEAPPROFILE prefix instead of CPU profile
#   SKIP_RUN - if set, reuse an existing tag dir's compiled JSON (no step 1)
set -e
PDVD_DIR=$(cd "$(dirname "$0")" && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}

RUN=${1:?run} IDX=${2:?idx}
RUN_PADDED=$(printf '%06d' "$((10#$RUN))")
TAG=${TAG:-profql_${RUN_PADDED}_${IDX}}
OUT=${3:-/home/xqian/tmp/ql_pdvd_${RUN_PADDED}_${IDX}.prof}
WORKDIR="$PDVD_DIR/work/${RUN_PADDED}_${IDX}_${TAG}"
CFG_JSON="$WORKDIR/.wct-clus.json"

if [ -z "$SKIP_RUN" ]; then
    "$PDVD_DIR/scripts/stage_ql_tag.sh" "$RUN" "$IDX" "$TAG"
    ( cd "$PDVD_DIR" && \
      env PDVD_KEEP_CFG=1 PDVD_LIGHT_SUFFIX="${PDVD_LIGHT_SUFFIX:-_keep}" \
          ./run_clus_evt.sh "$RUN" "$IDX" -s "$TAG" )
fi
[ -s "$CFG_JSON" ] || { echo "no compiled cfg at $CFG_JSON" >&2; exit 1; }

PROFLIB=${PROFLIB:-/usr/lib/x86_64-linux-gnu/libtcmalloc_and_profiler.so.4}
cd "$PDVD_DIR"
if [ -n "$HEAPOUT" ]; then
    env LD_PRELOAD="$PROFLIB" HEAPPROFILE="$HEAPOUT" GOGC=off \
        wire-cell -l stderr -L info -c "$CFG_JSON"
    OUT="$HEAPOUT"
else
    env LD_PRELOAD="$PROFLIB" CPUPROFILE="$OUT" CPUPROFILE_FREQUENCY=1000 GOGC=off \
        wire-cell -l stderr -L info -c "$CFG_JSON"
fi

echo "profile -> $OUT"
echo "view: google-pprof --text $(which wire-cell) $OUT | head -40"
