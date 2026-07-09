#!/bin/bash
# gperftools CPU-profile (or heap-profile) one event of the PDHD all-PD light
# chain.  Requires the decoana inputs to already exist (run
# run_light_allpd_evt.sh once for the event first).
# Usage: ./profile_light.sh <run> <event> [out.prof]
# Pre-compiles the cfg with wcsonnet (SIGPROF kills gojsonnet GC otherwise)
# and points output_dir at scratch so profiling never clobbers the real
# work/ archives.
# Env overrides:
#   PROFLIB  - LD_PRELOAD lib (default libtcmalloc_and_profiler)
#   OUTDIR   - where wire-cell writes outputs (default scratch)
#   HEAPOUT  - if set, do tcmalloc HEAPPROFILE instead of CPU profile
set -e
PDHD_DIR=$(cd "$(dirname "$0")" && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}

RUN=${1:?run} EVENT=${2:?event}
RUN_PADDED=$(printf "%06d" "$RUN")
OUT=${3:-/home/xqian/tmp/light_pdhd_${RUN_PADDED}_${EVENT}.prof}

# reuse the decoana inputs and offset already derived by run_light_allpd_evt.sh
ALLPD="$PDHD_DIR/work/${RUN_PADDED}_allpd${EVENT}"
SNIPCONV="$PDHD_DIR/work/${RUN_PADDED}_snip${EVENT}/snippet_decoana.root"
FSCONV="$PDHD_DIR/work/${RUN_PADDED}_fs${EVENT}/fullstream_decoana.root"
[ -s "$SNIPCONV" ] || SNIPCONV="$ALLPD/snippet_decoana.root"
[ -s "$FSCONV" ] || FSCONV="$ALLPD/fullstream_decoana.root"
if [ ! -s "$SNIPCONV" ] || [ ! -s "$FSCONV" ]; then
    echo "decoana inputs missing; run run_light_allpd_evt.sh $RUN $EVENT first" >&2
    exit 1
fi
OFFSET_US=$(python3 -c "
import json
cfg = json.load(open('$ALLPD/.wct-light-allpd.json'))
for c in cfg:
    if c.get('type') == 'OpFlashFinder':
        print(c['data'].get('offset_us', 0)); break
else: print(0)" 2>/dev/null || echo 0)
echo "offset_us: $OFFSET_US"

OUTDIR=${OUTDIR:-/home/xqian/tmp/prof_light_pdhd_${RUN_PADDED}_${EVENT}}
mkdir -p "$OUTDIR"

CFG=$OUTDIR/.wct-light-allpd.json
wcsonnet \
    -A "snip_file=${SNIPCONV}" \
    -A "fs_file=${FSCONV}" \
    -A "output_dir=${OUTDIR}" \
    -S "run=${RUN}" \
    -S "event=${EVENT}" \
    -S "offset_us=${OFFSET_US}" \
    -o "$CFG" \
    "$PDHD_DIR/wct-light-allpd-reco.jsonnet"

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
