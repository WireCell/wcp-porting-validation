#!/bin/bash
# gperftools CPU-profile (or heap-profile) one event's PR job (doc pdvd/28).
#   Usage: ./profile_pr.sh <run> <idx> [out.prof]
# Steps: stage a throwaway PR tag from SRC_TAG (default d27fresh), compile the
# job config through the runner (PDVD_PR_COMPILE_ONLY=1 keeps every -S TLA
# right and leaves .wct-pr_<tag>.json; wcsonnet precompile because SIGPROF
# corrupts the gojsonnet GC, CLAUDE.md M17), then run wire-cell on that JSON
# under libtcmalloc_and_profiler (production preloads tcmalloc, so a glibc
# profile would overstate allocator cost).  Never under setarch -R (SIGPROF
# dies).  The tag dir (profpr_*) is a throwaway, not a record.
# Env: TAG, SRC_TAG, PROFLIB, HEAPOUT (tcmalloc HEAPPROFILE prefix instead of
#      CPU), FREQ (CPUPROFILE_FREQUENCY, default 250), PR_ARGS (runner args,
#      default "-stm-fit"), LD_LIBRARY_PATH honoured (pin the binary).
set -e
PDVD_DIR=$(cd "$(dirname "$0")" && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}
RUN=${1:?run} IDX=${2:?idx}
RUN_PADDED=$(printf '%06d' "$((10#$RUN))")
TAG=${TAG:-profpr_${RUN_PADDED}_${IDX}}
OUT=${3:-/home/xqian/tmp/doc28/pr_${RUN_PADDED}_${IDX}.prof}
WORKDIR="$PDVD_DIR/work/${RUN_PADDED}_${IDX}_${TAG}"
CFG_JSON="$WORKDIR/.wct-pr_${TAG}.json"
[ -d "$WORKDIR" ] || "$PDVD_DIR/scripts/stage_pr_tag.sh" "$RUN" "$IDX" "$TAG" "${SRC_TAG:-d27fresh}"
( cd "$PDVD_DIR" && env PDVD_PR_COMPILE_ONLY=1 ./run_pr_evt.sh ${PR_ARGS:--stm-fit} -s "$TAG" "$RUN" "$IDX" )
[ -s "$CFG_JSON" ] || { echo "no compiled cfg at $CFG_JSON" >&2; exit 1; }
PROFLIB=${PROFLIB:-/usr/lib/x86_64-linux-gnu/libtcmalloc_and_profiler.so.4}
LOG="$WORKDIR/wct_pr_${RUN_PADDED}_${IDX}.log"
cd "$WORKDIR"
if [ -n "$HEAPOUT" ]; then
    env LD_PRELOAD="$PROFLIB" HEAPPROFILE="$HEAPOUT" GOGC=off \
        wire-cell -l stderr -l "$LOG:debug" -L debug -c "$CFG_JSON"
    OUT="$HEAPOUT"
else
    env LD_PRELOAD="$PROFLIB" CPUPROFILE="$OUT" CPUPROFILE_FREQUENCY=${FREQ:-250} GOGC=off \
        wire-cell -l stderr -l "$LOG:debug" -L debug -c "$CFG_JSON"
fi
echo "profile -> $OUT"
echo "view: google-pprof --text $(which wire-cell) $OUT | head -40"
