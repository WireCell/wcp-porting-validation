#!/bin/bash
# doc sbnd_xin/119 round 4 (memory): re-run the SBND peak-RSS TAIL events at CURRENT production.
#
# Fork by duplication (M10) of scripts/d119/stageB_lever.sh, which stays untouched -- it is the
# record of rounds 1-3 and its event set is hard-wired to the 62-event d118 gate manifest.  That
# manifest is exactly what this round cannot use:
#
#   THE REASON THIS SCRIPT EXISTS.  Doc 116 sec 14 measured the post-flip nuecc peak-RSS tail at
#   2.21 GiB, with 5 events of 2001 above 2 GiB -- a 0.25 % tail.  The d118/d119 gate manifest is
#   24 nuecc events and its maximum is 1.43 GiB.  It does not contain a single tail event and at
#   24 draws it never could.  Rounds 1-3 therefore say NOTHING about the tail, and reading their
#   "0 events above 2 GiB" as "the tail is gone" is a manifest-size artefact, not a measurement.
#   This script names the 5 tail events from the doc-116 arm's own .time.meta records and re-runs
#   THOSE, plus a p99 and a p50 control so the tail can be read against normal composition.
#
# The arm it produces is a MEASUREMENT arm, not a gate arm: it is the config to hand the heap
# profiler (scripts/perf/profile_pr119r4.sh), because .wct-cfg-evt<ID>.json written here IS
# current production by construction.  The doc-116 arm's own config is NOT -- it pins
# SBND_TRACKFIT_JSON to docs/pr/149_figs/149_tf_sbnd_kf.json, i.e. the measured cell, which
# predates both the proj_pad flip (d7d4da83) and round 3 (c590ae36).
#
# M1: rounds 1-3 ran on two binaries and this round must be unambiguous about which one it used.
# The round-3 pin is asserted, not assumed.  Note that `md5sum local/lib/libWireCellClus.so` does
# NOT match the pin: HEAD 253f1845 relinked the library after the pin was taken, and its only
# non-test change (clus/inc/WireCellClus/Facade_Util.h) is a COMMENT BLOCK.  Same code, different
# build id.  The pin is used because it is the binary every round-3 number was measured on.
#
# ALLOC picks the allocator, and getting this wrong voids the round:
#   ALLOC=prod  (default) SBND_PR_TCMALLOC unset, so run_pr_chain_batch.sh's own default applies
#               -- which is ON.  Round 1 flipped libtcmalloc_minimal into the SBND PR chain
#               (doc 119 sec 5), so TODAY'S PRODUCTION IS TCMALLOC.  Arm: work-r3nue-d119tail.
#   ALLOC=glibc SBND_PR_TCMALLOC=0, the pre-round-1 process environment.  This is NOT production
#               any more; it is the CONTROL, and it is the arm that is directly comparable with
#               doc 116 sec 14's tail numbers, which were all measured before round 1 existed.
#               Arm: work-r3nue-d119tailg.
#
# Keeping both is the point.  The allocator is the one term that differs between doc 116's tail and
# today's, so a single arm cannot say whether a change in the tail is the code or the allocator.
#
# Usage: [ALLOC=prod|glibc] [JOBS=n] stageB_tail.sh            # all 7 events
#        [ALLOC=prod|glibc] [JOBS=n] stageB_tail.sh f060/11239 # a subset, as <sub>/<evt>
set -u
cd -P "$(dirname "$0")/../.." || exit 1
SX=$PWD
J=${JOBS:-7}

R3_CLUS_MD5=4ff75274e43e766eaec4eef8b6107ae7   # round 3, toolkit c590ae36 == 253f1845 (comment-only)
export LIBSNAP=${LIBSNAP:-$HOME/tmp/d119r3-libpin}
export LD_LIBRARY_PATH=$LIBSNAP:${LD_LIBRARY_PATH:-}
[ -e "$LIBSNAP/libWireCellClus.so" ] || { echo "ERROR: no pin at $LIBSNAP" >&2; exit 1; }
CLUS_MD5=$(md5sum "$LIBSNAP/libWireCellClus.so" | cut -d' ' -f1)
[ "$CLUS_MD5" = "$R3_CLUS_MD5" ] \
    || { echo "REFUSING: pin is $CLUS_MD5, round 3 was measured on $R3_CLUS_MD5 (M1)." >&2; exit 2; }

# Production exactly as it runs today: no fit-JSON override (the in-tree sbnd_track_fitting.json
# now carries the flipped trajectory keys AND proj_pad), no cell TLAs.
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES:-pr_display}
unset PR_EXTRA_TLA SBND_TRACKFIT_JSON PR_GROUP_SIZE SBND_NO_DL PR_CFG_TREE
ALLOC=${ALLOC:-prod}
case "$ALLOC" in
    # UNSET, not =1.  run_pr_chain_batch.sh turns its own default OFF when it detects harness use
    # (PR_PIPELINE set, or SBND_PROTECT_BUNDLE=0); letting that logic run is what makes this arm
    # production rather than an arm that merely looks like it.
    prod)  # doc sbnd_xin/119 sec 11.8: the runner's allocator default MOVED to jemalloc.  This arm was
    # measured on tcmalloc, so it PINS tcmalloc rather than inheriting a default that no longer
    # means what it meant when the arm was produced.  Without this line a re-run silently yields
    # a JEMALLOC arm under a tcmalloc name -- the arm-naming trap of sec 11.7, one level down.
    unset SBND_PR_TCMALLOC
           export SBND_PR_ALLOC_LIB=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4; ARMSUF="" ;;
    glibc) export SBND_PR_TCMALLOC=0; ARMSUF="g" ;;
    *) echo "unknown ALLOC=$ALLOC (prod|glibc)" >&2; exit 2 ;;
esac

# The tail, named from work-r3nue-d116tfull/*/.time.meta (getrusage CHILDREN high-water).
# rank  peak RSS   sub / event
#   1    2.208 GiB  f060/11239      <- the doc-116 maximum
#   2    2.203      f049/12202
#   3    2.050      f188/11811
#   4    2.021      f100/5265
#   5    2.006      f196/6248
#   p99  1.602      f075/9393       <- the shoulder
#   p50  1.165      f054/7582       <- the CONTROL: what normal composition looks like
ALL=(f060/11239 f049/12202 f188/11811 f100/5265 f196/6248 f075/9393 f054/7582)
if [ $# -gt 0 ]; then SEL=("$@"); else SEL=("${ALL[@]}"); fi

A=$SX/work-r3nue-d115
B=$SX/work-r3nue-d119tail$ARMSUF
[ -d "$A" ] || { echo "ERROR: no stage-A arm $A" >&2; exit 1; }

# M13: never write into an arm this script did not create.
if [ -e "$B" ]; then
    [ "$(cat "$B/.d119_arm" 2>/dev/null)" = "tail-$ALLOC" ] \
        || { echo "REFUSING: $B exists and is not a d119 tail-$ALLOC arm (M13)" >&2; exit 2; }
    echo "=== resuming existing $B"
else
    mkdir -p "$B" && echo "tail-$ALLOC" > "$B/.d119_arm"
fi
LOGD=$HOME/tmp/d119-stageB-tail$ARMSUF; mkdir -p "$LOGD"

echo "=== d119 round 4: SBND peak-RSS tail at current production  $(date +%F_%H:%M:%S)"
echo "=== pin: $LIBSNAP  clus md5 $CLUS_MD5"
echo "=== toolkit HEAD: $(git -C /nfs/data/1/xqian/toolkit-dev/toolkit rev-parse HEAD)"
echo "=== fit JSON md5: $(md5sum /nfs/data/1/xqian/toolkit-dev/toolkit/cfg/pgrapher/experiment/sbnd/sbnd_track_fitting.json | cut -c1-12)"
echo "=== events: ${#SEL[@]}  JOBS=$J  ALLOC=$ALLOC  SBND_PR_TCMALLOC=${SBND_PR_TCMALLOC:-<unset: runner default = ON>}"
echo "=== arm: $B"
t0=$(date +%s)

# Each tail event lives in its own sub-root, so run_pr_chain_batch.sh gets one event per call and
# its own PR_JOBS cannot parallelise anything.  Fan out across the SPECS instead, capped at $J.
# Peak RSS is what this round measures and it is not contention-sensitive; wall_s from this arm is
# contaminated by the fan-out and must not be compared against the rounds 1-3 ladders.
for spec in "${SEL[@]}"; do
    sub=${spec%%/*}; evt=${spec##*/}
    qa=$A/$sub; qb=$B/$sub
    [ -d "$qa/evt$evt" ] || { echo "[$spec] ERROR: no stage-A input $qa/evt$evt" >&2; continue; }
    mkdir -p "$qb"
    while [ "$(jobs -rp | wc -l)" -ge "$J" ]; do wait -n; done
    ( PR_JOBS=1 "$SX/run_pr_chain_batch.sh" "$qa" "$qb" sim "$evt" \
          > "$LOGD/$sub-$evt.log" 2>&1
      echo "$?" > "$LOGD/$sub-$evt.rc" ) &
done
wait

for spec in "${SEL[@]}"; do
    sub=${spec%%/*}; evt=${spec##*/}
    meta="$B/$sub/pr_evt$evt/.time.meta"
    rss=$(tr ' ' '\n' < "$meta" 2>/dev/null | sed -n 's/^maxrss_kb=//p' | head -1)
    echo "[$spec] rc=$(cat "$LOGD/$sub-$evt.rc" 2>/dev/null) peak_rss=$(awk -v k="${rss:-0}" 'BEGIN{printf "%.3f", k/1048576}') GiB"
done

t1=$(date +%s)
echo "=== pin end: $(md5sum "$LIBSNAP/libWireCellClus.so" | cut -d' ' -f1)"
echo "=== finished in $((t1-t0)) s"
echo "=== pr_evt dirs : $(ls -d "$B"/*/pr_evt* 2>/dev/null | wc -l)"
echo "=== rc!=0 events: $(grep -L 'rc=0' "$B"/*/pr_evt*/rc.txt 2>/dev/null | wc -l)"
exit 0
