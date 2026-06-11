#!/bin/bash
# Convert SP frame archives for one event to per-anode Magnify ROOT files.
# Usage: ./run_sp_to_magnify_evt.sh [-I] [-d] [-s sel_tag] [-O suffix] <run> <evt|all> [subrun]
#        ./run_sp_to_magnify_evt.sh      # list available runs
#
# EVT may be 'all' to run every discovered event in parallel (capped at nproc,
# override with PDVD_MAX_JOBS=N).  Events with missing inputs are skipped.
#
# Input:  work/<RUN_PADDED>_<EVT>[<O_suffix>]/protodune-sp-frames-anode{0..7}.tar.bz2 (preferred)
#         input_data/<run_dir>/<evt_dir>/protodune-sp-frames-anode{0..7}.tar.bz2 (fallback)
#   -I:  force loading SP/raw frames from input_data even if work dir has them
#   -d:  DNN-ROI mode: read protodune-sp-dnnroi-frames-anode{N}.tar.bz2 (from
#        run_nf_sp_dnnroi_evt.sh).  That archive is a standard SP archive whose
#        gauss<N> traces are the DNN-ROI output, so this writes the usual
#        hu/hv/hw_{gauss,wiener,threshold}<N> histograms (gauss = DNN-ROI
#        output); the output ROOT name gets a '-dnnroi' suffix.
#   -s:  work/<RUN_PADDED>_<EVT>_sel<TAG>/input/ (from run_select_evt.sh)
#   -O:  Append <suffix> to the workdir name AND to the output ROOT filename,
#        matching run_nf_sp_dnnroi_evt.sh -O (e.g. -O _hybrid reads from
#        work/<RUN>_<EVT>_hybrid/ and writes magnify-...-anode<N>-dnnroi-hybrid.root).
# Orig frames (protodune-orig-frames-anode{N}.tar.bz2) are always sourced from
# input_data when present, producing hu/hv/hw_orig<N> histograms in Magnify.
# Output: work/<run>_<evt>[<O_suffix>][_sel<TAG>]/magnify-run<RUN>-evt<EVT>-anode<N>[-dnnroi][<O_suffix>].root

set -e

PDVD_DIR=$(cd "$(dirname "$0")" && pwd)

WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}

. "$PDVD_DIR/_runlib.sh"

SEL_TAG=""
FORCE_INPUT_DATA=""
INCLUDE_RAWDECON=0
INCLUDE_DECON=0
DNN_MODE=0
WORK_SUFFIX=""
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -I) FORCE_INPUT_DATA=1; shift ;;
        -d) DNN_MODE=1; shift ;;
        -s) SEL_TAG="$2"; shift 2 ;;
        -s*) SEL_TAG="${1#-s}"; shift ;;
        -O) WORK_SUFFIX="$2"; shift 2 ;;
        -O*) WORK_SUFFIX="${1#-O}"; shift ;;
        -R) INCLUDE_RAWDECON=1; INCLUDE_DECON=1; shift ;;   # rawdecon+decon TH2 in magnify (special mode)
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"

if [ $# -eq 0 ]; then
    list_runs; exit 0
fi

if [ $# -lt 2 ]; then
    echo "Usage: $0 [-I] [-d] [-s sel_tag] <run> <evt|all> [subrun]" >&2
    exit 1
fi
RUN=$1
EVT=$2
SUBRUN=${3:-0}

# DNN-ROI mode reads the post-DNN archives and suffixes the output ROOT name.
if [ "$DNN_MODE" -eq 1 ]; then
    INPUT_BASENAME="protodune-sp-dnnroi-frames"
    OUT_SUFFIX="-dnnroi"
else
    INPUT_BASENAME="protodune-sp-frames"
    OUT_SUFFIX=""
fi
# -O appends to both the workdir and the output rootfile basename so A/B
# comparison runs don't clobber each other.  Note: WORK_SUFFIX already
# includes the leading underscore (matches run_nf_sp_dnnroi_evt.sh -O).
if [ -n "$WORK_SUFFIX" ]; then
    OUT_SUFFIX="${OUT_SUFFIX}${WORK_SUFFIX}"
fi

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
        if ls "$rdir/${INPUT_BASENAME}-anode"*.tar.bz2 >/dev/null 2>&1; then
            echo "$rdir"; return 0
        fi
    done
    return 1
}

process_event() {
    local RUN=$1 EVT=$2
    local RUN_STRIPPED RUN_PADDED WORKDIR EVTDIR SP_DIR LOG
    local ANODE0_ARCHIVE EVENT_NO RAW_ARCHIVE RAW_DIR RAW_ARGS ORIG_ARCHIVE ORIG_ARGS
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
        WORKDIR="$PDVD_DIR/work/${RUN_PADDED}_${EVT}${WORK_SUFFIX}"
    fi
    echo "Event dir: $EVTDIR"

    if [ -z "$SEL_TAG" ] && [ -z "$FORCE_INPUT_DATA" ] && \
       ls "$WORKDIR/${INPUT_BASENAME}-anode"*.tar.bz2 >/dev/null 2>&1; then
        SP_DIR="$WORKDIR"
    else
        SP_DIR="$EVTDIR"
    fi
    echo "SP frames from: $SP_DIR"

    ANODE0_ARCHIVE="$SP_DIR/${INPUT_BASENAME}-anode0.tar.bz2"
    if [ ! -s "$ANODE0_ARCHIVE" ]; then
        echo "[skip] run=$RUN evt=$EVT: missing or empty $ANODE0_ARCHIVE" >&2
        return 2
    fi
    EVENT_NO=$(tar tjf "$ANODE0_ARCHIVE" | head -1 | sed -E 's/.*_([0-9]+)\.npy.*/\1/')
    if ! echo "$EVENT_NO" | grep -qE '^[0-9]+$'; then
        echo "ERROR: could not parse event number from $ANODE0_ARCHIVE (got: '$EVENT_NO')" >&2
        return 1
    fi
    echo "Art event number: $EVENT_NO"

    # Probe the actual frame tick count; readout length varies run to run
    # (6400/8000/10000 ticks seen) and the default nticks=6000 would stamp a
    # wrong Trun total_time_bin (see
    # docs/sp-img-readout-window-truncation.md).
    local _FRAME_NPY _SHAPE_TMP NTICKS
    _FRAME_NPY=$(tar tjf "$ANODE0_ARCHIVE" | grep -m1 "^frame_gauss") || {
        echo "ERROR: no frame_gauss* in $ANODE0_ARCHIVE" >&2; return 1; }
    _SHAPE_TMP=$(mktemp -d /home/xqian/tmp/magnticks.XXXXXX)
    tar xjf "$ANODE0_ARCHIVE" -C "$_SHAPE_TMP" "$_FRAME_NPY"
    NTICKS=$(python3 -c "
import numpy as np
a = np.load('${_SHAPE_TMP}/${_FRAME_NPY}', mmap_mode='r')
print(a.shape[1])
")
    rm -rf "$_SHAPE_TMP"
    if ! echo "$NTICKS" | grep -qE '^[0-9]+$'; then
        echo "ERROR: could not determine nticks from $_FRAME_NPY (got: '$NTICKS')" >&2
        return 1
    fi
    echo "Frame tick count: $NTICKS (from ${_FRAME_NPY})"

    mkdir -p "$WORKDIR"
    echo "Work dir: $WORKDIR"

    cd "$PDVD_DIR"

    local PROCESSED=0 N OUTPUT
    for N in 0 1 2 3 4 5 6 7; do
        local f="$SP_DIR/${INPUT_BASENAME}-anode${N}.tar.bz2"
        if [ ! -s "$f" ]; then
            echo "Skipping anode ${N} (missing or empty $f)"
            continue
        fi

        OUTPUT="$WORKDIR/magnify-run${RUN_PADDED}-evt${EVT}-anode${N}${OUT_SUFFIX}.root"
        LOG="$WORKDIR/wct_magnify_${RUN_PADDED}_${EVT}_anode${N}.log"
        echo "--- Anode ${N}: $OUTPUT"
        rm -f "$LOG"

        if [ -z "$SEL_TAG" ] && [ -z "$FORCE_INPUT_DATA" ] && \
           [ -s "$WORKDIR/protodune-sp-frames-raw-anode${N}.tar.bz2" ]; then
            RAW_ARCHIVE="$WORKDIR/protodune-sp-frames-raw-anode${N}.tar.bz2"
        else
            RAW_ARCHIVE="$EVTDIR/protodune-sp-frames-raw-anode${N}.tar.bz2"
        fi
        if [ -s "$RAW_ARCHIVE" ]; then
            RAW_DIR=$(dirname "$RAW_ARCHIVE")
            echo "    + raw: $RAW_ARCHIVE"
            RAW_ARGS="--tla-code include_raw=true --tla-str raw_input_prefix=${RAW_DIR}/protodune-sp-frames-raw"
        else
            RAW_ARGS="--tla-code include_raw=false"
        fi

        ORIG_ARCHIVE="$EVTDIR/protodune-orig-frames-anode${N}.tar.bz2"
        if [ -s "$ORIG_ARCHIVE" ]; then
            echo "    + orig: $ORIG_ARCHIVE"
            ORIG_ARGS="--tla-code include_orig=true --tla-str orig_input_prefix=${EVTDIR}/protodune-orig-frames"
        else
            ORIG_ARGS="--tla-code include_orig=false"
        fi

        local RAWDECON_ARGS=""
        if [ "$INCLUDE_RAWDECON" -eq 1 ]; then
            RAWDECON_ARGS="--tla-code include_rawdecon=true"
        fi

        wire-cell \
            -l stderr \
            -l "${LOG}:debug" \
            -L debug \
            --tla-str  "input_prefix=${SP_DIR}/${INPUT_BASENAME}" \
            --tla-code "anode_indices=[${N}]" \
            --tla-str  "output_file=${OUTPUT}" \
            --tla-code "run=${RUN_STRIPPED}" \
            --tla-code "subrun=${SUBRUN}" \
            --tla-code "event=${EVENT_NO}" \
            --tla-code "nticks=${NTICKS}" \
            ${RAW_ARGS} \
            ${ORIG_ARGS} \
            ${RAWDECON_ARGS} \
            -c wct-sp-to-magnify.jsonnet

        echo "    done -> $OUTPUT"
        PROCESSED=$((PROCESSED + 1))
    done

    if [ "$PROCESSED" -eq 0 ]; then
        echo "[skip] run=$RUN evt=$EVT: no anode archives found in $EVTDIR" >&2
        return 2
    fi

    echo "Magnify done: $PROCESSED anode(s) written to $WORKDIR/"
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
        _blogfile="$PDVD_DIR/work/.batch_magnify_${RUN_PADDED}_${_e}.log"
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
