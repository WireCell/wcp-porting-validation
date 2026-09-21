#!/bin/bash
# doc sbnd_xin/119 round 4 -- jemalloc HEAP profile of one PDVD PR job, at the PEAK and at exit.
#
# Fork by duplication (M10) of pdvd/profile_pr.sh, which stays untouched: it is the record of doc
# pdvd/28 and doc 30.  One thing differs, and it is the reason this file exists:
#
#   profile_pr.sh's heap path is gperftools HEAPPROFILE.  Doc pdvd/28 sec 0 measured that path at
#   ~25x slower, which is why every heap study in this tree since has used jemalloc sampling plus
#   pdhd/stm/perf/je2pprof.py.  The HEAPOUT branch in profile_pr.sh has therefore been dead since
#   doc 28 -- it is kept there as a record, and replaced here.
#
# It also takes a dump SERIES via lg_prof_interval rather than one dump at exit.  Round 4 needs
# the PEAK, not the exit state:
#
#   The doc-119 census shows CreateSteinerGraph adding a MEAN of +0.35 GiB but a MAX of +1.97 GiB
#   on PDVD (+1.45 on PDHD) -- a 5x tail on the same code.  A tail that is transient inside the
#   stage is invisible to an exit-time dump and would read as "Steiner holds nothing".  Peak dump
#   vs final dump is what tells those apart.
#
#   NOT via prof_gdump, which is the obvious trigger and is unusable at this scale: on a 10-second
#   SBND event it produced 29 470 dumps and 5.1 GB, since in a monotonically growing heap nearly
#   every extending allocation is a new VM high-water mark.
#
# CAVEAT TO CARRY: jemalloc REPLACES the allocator.  PDVD production preloads libtcmalloc_minimal,
# SBND's PR chain preloads nothing (glibc).  So RSS from this run is not any detector's production
# RSS and must never be compared across detectors -- doc 119 sec 9.3 records the round-3 RSS claim
# that this exact confound already broke once.  Live-heap COMPOSITION is what transfers.
#
# Usage:  [TAG=...] [PR_ARGS=...] ./profile_pr_heap.sh <run> <idx> [outdir]
# Env: SRC_TAG (default p100flip -- what work/039252_8_d119vr3 itself symlinks its pctree from,
#      i.e. the census arm's own input; d30_run_pdvd_arm.sh's SRC=d48nu7 default does not exist
#      for this event), LG_SAMPLE (lg_prof_sample, default 19 = 512 KiB), PR_ARGS (runner args, default
#      "-nu -stm-fit", matching how pdhd/stm/perf/d30_run_pdvd_arm.sh drove the d119vr3
#      census arm -- MODE=-nu STMFIT=1 SRC=d48nu7, so those are the defaults here too),
#      LG_INTERVAL (lg_prof_interval, default 28 = a dump per 256 MiB allocated),
#      LD_LIBRARY_PATH honoured (pin the binary).
set -e
PDVD_DIR=$(cd "$(dirname "$0")" && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}
RUN=${1:?run} IDX=${2:?idx}
RUN_PADDED=$(printf '%06d' "$((10#$RUN))")
TAG=${TAG:-heappr_${RUN_PADDED}_${IDX}}
OD=${3:-/home/xqian/tmp/d119r4/pdvd_${RUN_PADDED}_${IDX}}
WORKDIR="$PDVD_DIR/work/${RUN_PADDED}_${IDX}_${TAG}"
CFG_JSON="$WORKDIR/.wct-pr_${TAG}.json"
# The tag dir is a throwaway (profile_pr.sh's own convention); it is NOT one of the doc-30 or
# doc-119 census tags, so M13 is not in play -- but the name is distinct from profpr_* so a heap
# run can never land on a CPU run's tag either.
[ -d "$WORKDIR" ] || "$PDVD_DIR/scripts/stage_pr_tag.sh" "$RUN" "$IDX" "$TAG" "${SRC_TAG:-p100flip}"
( cd "$PDVD_DIR" && env PDVD_PR_COMPILE_ONLY=1 ./run_pr_evt.sh ${PR_ARGS:--nu -stm-fit} -s "$TAG" "$RUN" "$IDX" )
[ -s "$CFG_JSON" ] || { echo "no compiled cfg at $CFG_JSON" >&2; exit 1; }

JEMALLOC=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
[ -e "$JEMALLOC" ] || { echo "ERROR: no jemalloc at $JEMALLOC" >&2; exit 1; }
# doc pdvd/28 sec 27: the DL vertex is PDVD production; its SCN import needs libpython preloaded
# (as run_pr_evt.sh does) or the profile runs the geometric fallback silently.
PRELOAD="$JEMALLOC"
if [ "${WCT_PYLIB:-on}" != "off" ]; then
    PYLIB=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.11.so.1.0
    PRELOAD="$PYLIB:$JEMALLOC"
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
mkdir -p "$OD"; OD=$(cd "$OD" && pwd -P)
LOG="$WORKDIR/wct_pr_${RUN_PADDED}_${IDX}.log"
PFX="$OD/je_evt${RUN_PADDED}_${IDX}"

echo "=== d119 round 4 PDVD heap profile: run $RUN_PADDED idx $IDX  args ${PR_ARGS:--nu -stm-fit}"
echo "=== prefix: $PFX"
cd "$WORKDIR"
# No wrapper process carries the profiling env: MALLOC_CONF is set on wire-cell alone.  A wrapper
# that inherits it dumps over the same prefix and the series becomes unreadable -- the same trap
# the SBND CPU profiler hit with timecmd.py.
rc=0
env LD_PRELOAD="$PRELOAD" \
    MALLOC_CONF="prof:true,prof_active:true,lg_prof_sample:${LG_SAMPLE:-19},prof_prefix:$PFX,prof_final:true,lg_prof_interval:${LG_INTERVAL:-28}" \
    wire-cell -l stderr -l "$LOG:debug" -L debug -c "$CFG_JSON" > "$OD/heap.stdout" 2>&1 &
WCPID=$!
MAPPED=""
for _ in $(seq 1 60); do
    MAPPED=$(sed -n 's#.* \(/.*libWireCellClus\.so\)$#\1#p' "/proc/$WCPID/maps" 2>/dev/null | head -1) || true
    [ -n "$MAPPED" ] && break
    sleep 1
done
# Record what the PROCESS mapped, not what we asked it to map -- a freshness proof, not a claim.
if [ -n "$MAPPED" ]; then
    echo "=== process mapped: $MAPPED  md5 $(md5sum "$MAPPED" | cut -d' ' -f1)" | tee "$OD/libproof.txt"
else
    echo "=== WARNING: could not read libWireCellClus.so from /proc/$WCPID/maps" | tee "$OD/libproof.txt"
fi
wait $WCPID || rc=$?

echo "rc=$rc"
echo "=== dumps: $(find "$(dirname "$PFX")" -maxdepth 1 -name "$(basename "$PFX").*.heap" | wc -l)  (.i* = per-interval, .f = final/exit)"
echo "=== ladder peak from the job log:"
grep -o 'res=[0-9.e+]*K' "$LOG" 2>/dev/null | sed 's/res=//;s/K//' \
    | sort -g | tail -1 | awk '{printf "    MEM ladder peak res = %.3f GiB\n", $1/1048576}'
echo "view: sbnd/sbnd_xin/scripts/d119/heap_rank.py $OD ${RUN_PADDED}_${IDX}"
exit $rc
