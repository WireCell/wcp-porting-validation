#!/bin/bash
# gperftools CPU- or heap-profile the uboone-mabc job for one filelist entry.
#
# Usage: ./profile_mabc.sh <idx> [cpu|heap] [out]
# Pre-compiles the cfg with wcsonnet (SIGPROF corrupts gojsonnet otherwise),
# then runs wire-cell under libtcmalloc_and_profiler.  Work dir (and default
# profile output) is /home/xqian/tmp/prof_mabc_<EV>/ -- never /tmp.
# Env overrides: PROFLIB, FREQ (CPU sample Hz, default 250).
set -eu
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
QLPORT=$(dirname "$SCRIPTS")

IDX=${1:?usage: profile_mabc.sh <idx> [cpu|heap] [out]}
MODE=${2:-cpu}

FILE=$(sed -n "$((IDX+1))p" "$QLPORT/filelist")
[[ "$FILE" =~ nuselEval_([0-9]+)_([0-9]+)_([0-9]+)\.root ]] \
    || { echo "unparseable filelist line: $FILE" >&2; exit 1; }
RUN=${BASH_REMATCH[1]} SR=${BASH_REMATCH[2]} EV=${BASH_REMATCH[3]}

WORK=/home/xqian/tmp/prof_mabc_${EV}
mkdir -p "$WORK"
ln -sfn "$QLPORT/rootfiles" "$WORK/rootfiles"
ln -sfn "$QLPORT/uboone_track_fitting.json" "$WORK/uboone_track_fitting.json"
OUT=${3:-$WORK/mabc_${EV}.${MODE}.prof}

CFG="$WORK/mabc.json"
wcsonnet -A kind=both -A "beezip=mabc_${IDX}.zip" -A "initial_index=$IDX" \
    -A "initial_runNo=$RUN" -A "initial_subRunNo=$SR" -A "initial_eventNo=$EV" \
    -A "infiles=$FILE" -o "$CFG" "$QLPORT/uboone-mabc.jsonnet"

cd "$WORK"
PROFLIB=${PROFLIB:-/usr/lib/x86_64-linux-gnu/libtcmalloc_and_profiler.so.4}
if [ "$MODE" = "heap" ]; then
    LD_PRELOAD="$PROFLIB" HEAPPROFILE="$OUT" \
        wire-cell -l stderr -L info -c "$CFG"
    echo "heap profiles -> $OUT.*.heap"
    echo "view: google-pprof --text --inuse_space \$(which wire-cell) ${OUT}.<N>.heap | head -40"
    echo "      google-pprof --text --alloc_space \$(which wire-cell) ${OUT}.<N>.heap | head -40"
else
    LD_PRELOAD="$PROFLIB" CPUPROFILE="$OUT" CPUPROFILE_FREQUENCY="${FREQ:-250}" \
        wire-cell -l stderr -L info -c "$CFG"
    echo "cpu profile -> $OUT"
    echo "view: google-pprof --text --cum \$(which wire-cell) $OUT | head -60"
fi
