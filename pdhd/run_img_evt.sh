#!/bin/bash
# Run imaging for one event.
# Usage: ./run_img_evt.sh [-I] [-P] [-a anode] [-S] [-s sel_tag] [-d on|off] <run> <evt|all>
#        ./run_img_evt.sh                # list available runs
#
#   -P:  per-anode sequential mode.  Runs one wire-cell process per anode
#        instead of all anodes in one process.  Outputs are identical (the
#        per-anode pipelines are independent); peak RSS drops from the sum
#        of all anode pipelines to the busiest single anode.  Configs are
#        pre-compiled per anode with wcsonnet to avoid N jsonnet compiles.
#
# EVT may be 'all' to run every discovered event in parallel (capped at nproc,
# override with PDHD_MAX_JOBS=N).  Events with missing inputs are skipped.
#
# Input:  work/<RUN_PADDED>_<EVT>/protodunehd-sp-frames-anode{0..3}.tar.bz2  (preferred)
#         input_data/<run_dir>/<evt_dir>/protodunehd-sp-frames-anode{0..3}.tar.bz2  (fallback)
#   -I:  force loading SP frames from input_data even if work dir has them
#   By default the dense archive is used.  If the dense archive for an anode is
#   missing and a sparse variant (*-sparseon.tar.bz2) exists, the sparse variant
#   is used automatically as a fallback.
#   -S:  force-prefer the sparse variant for every anode that has one.
#   -s:  work/<RUN_PADDED>_<EVT>_sel<TAG>/input/ (from run_select_evt.sh)
#   -d:  on|off (default off).  When 'on', consume DNN-ROI output
#        (protodunehd-sp-dnnroi-frames-anode{N}.tar.bz2 from work/) instead
#        of the standard SP frames.  Produced by run_nf_sp_dnnroi_evt.sh.
# Output: work/<run>_<evt>[_sel<TAG>]/clusters-apa-apa{N}-ms-{active,masked}.tar.gz

set -e

PDHD_DIR=$(cd "$(dirname "$0")" && pwd)

WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}

# Preload tcmalloc for the imaging wire-cell processes: imaging is allocator
# bound (graph lifecycle churn) and tcmalloc cuts busy-event wall ~25-50%
# with slightly lower RSS; outputs verified byte-identical on the A/B event
# set.  Disable with WCT_TCMALLOC=off.  Do NOT apply to clustering: a
# pointer-order-sensitive path there changes results under tcmalloc (see
# clus/docs/imgclus-optimization-log.md).
TCMALLOC_SO=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
WC_PRELOAD=""
if [ "${WCT_TCMALLOC:-on}" = "on" ] && [ -f "$TCMALLOC_SO" ]; then
    WC_PRELOAD="LD_PRELOAD=$TCMALLOC_SO"
fi

. "$PDHD_DIR/_runlib.sh"

ANODE=""
SEL_TAG=""
FORCE_SPARSE=false
FORCE_INPUT_DATA=""
USE_DNNROI="off"
PER_ANODE=false
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -I) FORCE_INPUT_DATA=1; shift ;;
        -P) PER_ANODE=true; shift ;;
        -a) ANODE="$2"; shift 2 ;;
        -a*) ANODE="${1#-a}"; shift ;;
        -s) SEL_TAG="$2"; shift 2 ;;
        -s*) SEL_TAG="${1#-s}"; shift ;;
        -S) FORCE_SPARSE=true; shift ;;
        -d) USE_DNNROI="$2"; shift 2 ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"

case "$USE_DNNROI" in
    on|off) ;;
    *) echo "[err] -d must be 'on' or 'off' (got '$USE_DNNROI')" >&2; exit 1 ;;
esac
if [ "$USE_DNNROI" = "on" ]; then
    INPUT_BASENAME="protodunehd-sp-dnnroi-frames"
else
    INPUT_BASENAME="protodunehd-sp-frames"
fi

if [ $# -eq 0 ]; then
    list_runs; exit 0
fi

if [ $# -lt 2 ]; then
    echo "Usage: $0 [-I] [-P] [-a anode] [-S] [-s sel_tag] [-d on|off] <run> <evt|all>" >&2
    exit 1
fi
RUN=$1
EVT=$2

RUN_STRIPPED=$(echo "$RUN" | sed 's/^0*//')
[ -z "$RUN_STRIPPED" ] && RUN_STRIPPED=0
RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")

# Scans every input_data* root (the reorg split data into
# input_data_<gain>_<old|new>_coh_grouping); first match wins.
find_evtdir() {
    local base
    for base in "$PDHD_DIR"/input_data "$PDHD_DIR"/input_data_*; do
        [ -d "$base" ] || continue
        for rname in "run${RUN}" "run${RUN_PADDED}" "run${RUN_STRIPPED}"; do
            local rdir="$base/$rname"
            [ -d "$rdir" ] || continue
            for ename in "evt${EVT}" "evt_${EVT}"; do
                local cand="$rdir/$ename"
                if [ -d "$cand" ] && [ -n "$(ls -A "$cand" 2>/dev/null)" ]; then
                    echo "$cand"; return 0
                fi
            done
            if ls "$rdir/${INPUT_BASENAME}-anode"*.tar.bz2 >/dev/null 2>&1; then
                echo "$rdir"; return 0
            fi
        done
    done
    return 1
}

process_event() {
    local RUN=$1 EVT=$2
    local RUN_STRIPPED RUN_PADDED EVTDIR WORKDIR
    local ANODE_CODE TAG_SUFFIX LOG INPUT_PREFIX NEED_STAGE STAGE_DIR ai dense sparse
    local -a ANODE_INDICES
    RUN_STRIPPED=$(echo "$RUN" | sed 's/^0*//')
    [ -z "$RUN_STRIPPED" ] && RUN_STRIPPED=0
    RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")

    if [ -n "$SEL_TAG" ]; then
        WORKDIR="$PDHD_DIR/work/${RUN_PADDED}_${EVT}_${SEL_TAG}"
        EVTDIR="$WORKDIR/input"
        if [ ! -d "$EVTDIR" ]; then
            echo "[skip] run=$RUN evt=$EVT: selection dir not found: $EVTDIR" >&2
            return 2
        fi
    else
        EVTDIR=$(find_evtdir) || EVTDIR=""
        if [ -z "$EVTDIR" ]; then
            echo "[skip] run=$RUN evt=$EVT: no event dir under input_data/" >&2
            return 2
        fi
        WORKDIR="$PDHD_DIR/work/${RUN_PADDED}_${EVT}"
    fi
    echo "Event dir: $EVTDIR"

    if [ -n "$ANODE" ]; then
        ANODE_CODE="[$ANODE]"
        ANODE_INDICES=("$ANODE")
        TAG_SUFFIX="_a${ANODE}"
    else
        ANODE_CODE="[0,1,2,3]"
        ANODE_INDICES=(0 1 2 3)
        TAG_SUFFIX=""
    fi

    mkdir -p "$WORKDIR"

    # Prefer SP frames produced locally in work dir; -I forces input_data.
    # When -d on, only the work-dir DNN-ROI archive is searched (no
    # sparse fallback and no input_data fallback — those don't apply).
    if [ -z "$SEL_TAG" ] && [ -z "$FORCE_INPUT_DATA" ] && \
       ls "$WORKDIR/${INPUT_BASENAME}-anode"*.tar.bz2 >/dev/null 2>&1; then
        INPUT_PREFIX="${WORKDIR}/${INPUT_BASENAME}"
        echo "SP prefix: $INPUT_PREFIX"
    elif [ "$USE_DNNROI" = "on" ]; then
        echo "[skip] run=$RUN evt=$EVT: no ${INPUT_BASENAME}-anode*.tar.bz2 in $WORKDIR (run run_nf_sp_dnnroi_evt.sh first)" >&2
        return 2
    else
        # Determine per-anode archive: dense by default; sparse if forced (-S) or
        # dense is missing.  Stage symlinks only when at least one anode uses sparse
        # (sparse archive name differs from FrameFileSource's expected pattern).
        NEED_STAGE=false
        for ai in "${ANODE_INDICES[@]}"; do
            dense="${EVTDIR}/${INPUT_BASENAME}-anode${ai}.tar.bz2"
            sparse="${EVTDIR}/${INPUT_BASENAME}-anode${ai}-sparseon.tar.bz2"
            if $FORCE_SPARSE && [ -f "$sparse" ]; then
                NEED_STAGE=true; break
            elif [ ! -f "$dense" ] && [ -f "$sparse" ]; then
                NEED_STAGE=true; break
            fi
        done

        if $NEED_STAGE; then
            STAGE_DIR="${WORKDIR}/sp_stage"
            mkdir -p "$STAGE_DIR"
            for ai in "${ANODE_INDICES[@]}"; do
                dense="${EVTDIR}/${INPUT_BASENAME}-anode${ai}.tar.bz2"
                sparse="${EVTDIR}/${INPUT_BASENAME}-anode${ai}-sparseon.tar.bz2"
                if $FORCE_SPARSE && [ -f "$sparse" ]; then
                    ln -sf "$sparse" "${STAGE_DIR}/${INPUT_BASENAME}-anode${ai}.tar.bz2"
                    echo "  anode${ai}: sparse (forced)"
                elif [ -f "$dense" ]; then
                    ln -sf "$dense" "${STAGE_DIR}/${INPUT_BASENAME}-anode${ai}.tar.bz2"
                    echo "  anode${ai}: dense"
                elif [ -f "$sparse" ]; then
                    ln -sf "$sparse" "${STAGE_DIR}/${INPUT_BASENAME}-anode${ai}.tar.bz2"
                    echo "  anode${ai}: sparse (dense not found)"
                else
                    echo "[skip] run=$RUN evt=$EVT: no archive for anode${ai} in $EVTDIR" >&2
                    return 2
                fi
            done
            INPUT_PREFIX="${STAGE_DIR}/${INPUT_BASENAME}"
        else
            INPUT_PREFIX="${EVTDIR}/${INPUT_BASENAME}"
        fi
    fi

    # Probe the actual frame tick count from the first selected anode's
    # archive; readout length varies run to run and the Reframer must match
    # it exactly (too short truncates real activity, see
    # pdvd/docs/sp-img-readout-window-truncation.md).
    local _PROBE_ANODE=${ANODE_INDICES[0]}
    local _PROBE_TAR="${INPUT_PREFIX}-anode${_PROBE_ANODE}.tar.bz2"
    local _FRAME_NPY _SHAPE_TMP NTICKS
    _FRAME_NPY=$(tar tjf "$_PROBE_TAR" | grep -m1 "^frame_gauss${_PROBE_ANODE}_") || {
        echo "ERROR: no frame_gauss${_PROBE_ANODE}_* in $_PROBE_TAR" >&2; return 2; }
    _SHAPE_TMP=$(mktemp -d /home/xqian/tmp/imgnticks.XXXXXX)
    tar xjf "$_PROBE_TAR" -C "$_SHAPE_TMP" "$_FRAME_NPY"
    NTICKS=$(python3 -c "
import numpy as np
a = np.load('${_SHAPE_TMP}/${_FRAME_NPY}', mmap_mode='r')
print(a.shape[1])
")
    rm -rf "$_SHAPE_TMP"
    if ! echo "$NTICKS" | grep -qE '^[0-9]+$'; then
        echo "ERROR: could not determine nticks from $_FRAME_NPY (got: '$NTICKS')" >&2
        return 2
    fi
    echo "Frame tick count: $NTICKS (from ${_FRAME_NPY})"

    LOG="$WORKDIR/wct_img_${RUN_PADDED}_${EVT}${TAG_SUFFIX}.log"
    echo "Work dir:  $WORKDIR"
    echo "Log:       $LOG"

    cd "$PDHD_DIR"
    rm -f "$LOG"
    if $PER_ANODE; then
        # One wire-cell process per anode, sequential.  Per-anode pipelines
        # are independent and write only their own clusters-apa-* files, so
        # outputs are identical to the all-anode run.
        local CFG_JSON ALOG ai_t0
        for ai in "${ANODE_INDICES[@]}"; do
            wcsonnet \
                -A "input_prefix=${INPUT_PREFIX}" \
                -S "anode_indices=[$ai]" \
                -A "output_dir=${WORKDIR}" \
                -S "nticks=${NTICKS}" \
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
            --tla-str "input_prefix=${INPUT_PREFIX}" \
            --tla-code "anode_indices=${ANODE_CODE}" \
            --tla-str "output_dir=${WORKDIR}" \
            --tla-code "nticks=${NTICKS}" \
            -c wct-img-all.jsonnet
    fi

    echo "Imaging done -> $WORKDIR"
}

mkdir -p "$PDHD_DIR/work"
if [ "$EVT" = "all" ]; then
    batch_init
    mapfile -t _events < <(discover_events "$RUN" "$RUN_PADDED")
    if [ ${#_events[@]} -eq 0 ]; then
        echo "no events found for run=$RUN under input_data/ or work/" >&2; exit 1
    fi
    echo "Found ${#_events[@]} event(s) for run=$RUN: ${_events[*]}"
    echo "Parallel jobs: $BATCH_MAX"
    for _e in "${_events[@]}"; do
        _blogfile="$PDHD_DIR/work/.batch_img_${RUN_PADDED}_${_e}.log"
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
