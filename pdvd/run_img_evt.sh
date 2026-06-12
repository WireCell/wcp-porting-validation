#!/bin/bash
# Run imaging for one event.
# Usage: ./run_img_evt.sh [-I] [-1] [-a anode] [-s sel_tag] <run> <evt|all>
#        ./run_img_evt.sh                # list available runs
#
#   Per-anode sequential mode is the DEFAULT: one wire-cell process per
#   anode instead of all anodes in one process.  Outputs are identical (the
#   per-anode pipelines are independent); peak RSS drops from the sum of
#   all anode pipelines to the busiest single anode.  Configs are
#   pre-compiled per anode with wcsonnet to avoid N jsonnet compiles.
#   -1:  revert to the old single-process all-anode mode.
#   -P:  accepted for back-compat (per-anode, now the default).
#
# EVT may be 'all' to run every discovered event in parallel (capped at nproc,
# override with PDVD_MAX_JOBS=N).  Events with missing inputs are skipped.
#
# Input:  work/<RUN_PADDED>_<EVT>/protodune-sp-frames-anode{0..7}.tar.bz2  (preferred)
#         input_data/<run_dir>/<evt_dir>/protodune-sp-frames-anode{0..7}.tar.bz2  (fallback)
#   -I:  force loading SP frames from input_data even if work dir has them
#   -s:  work/<RUN_PADDED>_<EVT>_sel<TAG>/input/ (from run_select_evt.sh)
# Output: work/<run>_<evt>[_sel<TAG>]/clusters-apa-anode{N}-ms-{active,masked}.tar.gz

set -e

PDVD_DIR=$(cd "$(dirname "$0")" && pwd)

WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}

# Preload tcmalloc for the imaging wire-cell processes: imaging is allocator
# bound (graph lifecycle churn) and tcmalloc cuts busy-event wall ~25-50%
# with slightly lower RSS; outputs verified byte-identical on the A/B event
# set.  Disable with WCT_TCMALLOC=off.  (Clustering also preloads tcmalloc
# since the BlobLess pointer-order fix; see run_clus_evt.sh and
# clus/docs/imgclus-optimization-log.md.)
TCMALLOC_SO=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
WC_PRELOAD=""
if [ "${WCT_TCMALLOC:-on}" = "on" ] && [ -f "$TCMALLOC_SO" ]; then
    WC_PRELOAD="LD_PRELOAD=$TCMALLOC_SO"
fi

. "$PDVD_DIR/_runlib.sh"

ANODE=""
SEL_TAG=""
FORCE_INPUT_DATA=""
PER_ANODE=true
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -I) FORCE_INPUT_DATA=1; shift ;;
        -P) PER_ANODE=true; shift ;;
        -1) PER_ANODE=false; shift ;;
        -a) ANODE="$2"; shift 2 ;;
        -a*) ANODE="${1#-a}"; shift ;;
        -s) SEL_TAG="$2"; shift 2 ;;
        -s*) SEL_TAG="${1#-s}"; shift ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"

if [ $# -eq 0 ]; then
    list_runs; exit 0
fi

if [ $# -lt 2 ]; then
    echo "Usage: $0 [-I] [-1] [-a anode] [-s sel_tag] <run> <evt|all>" >&2
    exit 1
fi
RUN=$1
EVT=$2

RUN_STRIPPED=$(echo "$RUN" | sed 's/^0*//')
[ -z "$RUN_STRIPPED" ] && RUN_STRIPPED=0
RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")

find_evtdir() {
    local base="$PDVD_DIR/input_data"
    for rname in "run${RUN}" "run${RUN_PADDED}" "run${RUN_STRIPPED}"; do
        local rdir="$base/$rname"
        [ -d "$rdir" ] || continue
        for ename in "evt${EVT}" "evt_${EVT}"; do
            local cand="$rdir/$ename"
            if [ -d "$cand" ] && [ -n "$(ls -A "$cand" 2>/dev/null)" ]; then
                echo "$cand"; return 0
            fi
        done
        if ls "$rdir/protodune-sp-frames-anode"*.tar.bz2 >/dev/null 2>&1; then
            echo "$rdir"; return 0
        fi
    done
    return 1
}

process_event() {
    local RUN=$1 EVT=$2
    local RUN_STRIPPED RUN_PADDED WORKDIR EVTDIR SP_PREFIX ANODE_CODE TAG_SUFFIX LOG
    RUN_STRIPPED=$(echo "$RUN" | sed 's/^0*//')
    [ -z "$RUN_STRIPPED" ] && RUN_STRIPPED=0
    RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")

    if [ -n "$SEL_TAG" ]; then
        WORKDIR="$PDVD_DIR/work/${RUN_PADDED}_${EVT}_${SEL_TAG}"
        EVTDIR="$WORKDIR/input"
        if [ ! -d "$EVTDIR" ]; then
            echo "[skip] run=$RUN evt=$EVT: selection dir not found: $EVTDIR" >&2
            return 2
        fi
    else
        EVTDIR=$(find_evtdir) || EVTDIR=""
        if [ -z "$EVTDIR" ]; then
            echo "[skip] run=$RUN evt=$EVT: no event dir found under input_data/" >&2
            return 2
        fi
        WORKDIR="$PDVD_DIR/work/${RUN_PADDED}_${EVT}"
    fi
    echo "Event dir: $EVTDIR"

    if [ -z "$SEL_TAG" ] && [ -z "$FORCE_INPUT_DATA" ] && \
       ls "$WORKDIR/protodune-sp-frames-anode"*.tar.bz2 >/dev/null 2>&1; then
        SP_PREFIX="$WORKDIR/protodune-sp-frames"
    else
        SP_PREFIX="$EVTDIR/protodune-sp-frames"
    fi
    echo "SP prefix: $SP_PREFIX"

    local -a ANODE_INDICES
    if [ -n "$ANODE" ]; then
        ANODE_CODE="[$ANODE]"
        ANODE_INDICES=("$ANODE")
        TAG_SUFFIX="_a${ANODE}"
    else
        ANODE_CODE="[0,1,2,3,4,5,6,7]"
        ANODE_INDICES=(0 1 2 3 4 5 6 7)
        TAG_SUFFIX=""
    fi

    mkdir -p "$WORKDIR"
    LOG="$WORKDIR/wct_img_${RUN_PADDED}_${EVT}${TAG_SUFFIX}.log"
    echo "Work dir:  $WORKDIR"
    echo "Log:       $LOG"

    cd "$PDVD_DIR"
    rm -f "$LOG"
    if $PER_ANODE; then
        # One wire-cell process per anode, sequential.  Per-anode pipelines
        # are independent and write only their own clusters-apa-* files, so
        # outputs are identical to the all-anode run.
        local CFG_JSON ALOG ai ai_t0
        for ai in "${ANODE_INDICES[@]}"; do
            wcsonnet \
                -A "input_prefix=${SP_PREFIX}" \
                -S "anode_indices=[$ai]" \
                -A "output_dir=${WORKDIR}" \
                -o "$WORKDIR/.wct-img-a${ai}.json" wct-img-all.jsonnet &
        done
        wait
        for ai in "${ANODE_INDICES[@]}"; do
            CFG_JSON="$WORKDIR/.wct-img-a${ai}.json"
            if [ ! -s "$CFG_JSON" ]; then
                echo "ERROR: wcsonnet failed for anode${ai}" >&2; return 1
            fi
        done
        for ai in "${ANODE_INDICES[@]}"; do
            CFG_JSON="$WORKDIR/.wct-img-a${ai}.json"
            ALOG="$WORKDIR/wct_img_${RUN_PADDED}_${EVT}_a${ai}.log"
            rm -f "$ALOG"
            ai_t0=$SECONDS
            env $WC_PRELOAD GOGC=off wire-cell -l stderr -l "${ALOG}:debug" -L debug -c "$CFG_JSON"
            echo "anode${ai} imaging: $((SECONDS - ai_t0)) s"
            rm -f "$CFG_JSON"
        done
    else
        env $WC_PRELOAD wire-cell \
            -l stderr \
            -l "${LOG}:debug" \
            -L debug \
            --tla-str "input_prefix=${SP_PREFIX}" \
            --tla-code "anode_indices=${ANODE_CODE}" \
            --tla-str "output_dir=${WORKDIR}" \
            -c wct-img-all.jsonnet
    fi

    echo "Imaging done -> $WORKDIR"
}

mkdir -p "$PDVD_DIR/work"
if [ "$EVT" = "all" ]; then
    batch_init
    mapfile -t _events < <(discover_events "$RUN" "$RUN_PADDED")
    if [ ${#_events[@]} -eq 0 ]; then
        echo "no events found for run=$RUN under input_data/ or work/" >&2; exit 1
    fi
    echo "Found ${#_events[@]} event(s) for run=$RUN: ${_events[*]}"
    echo "Parallel jobs: $BATCH_MAX"
    for _e in "${_events[@]}"; do
        _blogfile="$PDVD_DIR/work/.batch_img_${RUN_PADDED}_${_e}.log"
        batch_wait_slot
        ( process_event "$RUN" "$_e" ) > "$_blogfile" 2>&1 &
        BATCH_PIDS[$!]=$_e
        echo "  [start] evt=$_e  log: $_blogfile"
    done
    batch_drain
    batch_summary
    exit $?
else
    ( process_event "$RUN" "$EVT" )
    exit $?
fi
