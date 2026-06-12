#!/bin/bash
# Convert SBND SP frames for one event to per-anode Magnify ROOT files.  -h for help.
# Usage: ./run_sp_to_magnify_evt.sh [mc|data] [-N n] [-s sel_tag] <idx|all> [run] [subrun]
#        ./run_sp_to_magnify_evt.sh [mc|data] [-N n]   # list available events
#   mode:    mc (default) | data — selects input_files/input-<N>evt-<mode>/frames-dnn.tar.bz2
#   -N:      event-sample size (default 10); e.g. -N 100 uses input-100evt-<mode>
#   idx:     1-based event index into the chosen sample/mode; all = every event (parallel)
#   all:     process all events in parallel (up to nproc jobs; override with SBND_MAX_JOBS=N)
#   run:     run number stored in ROOT Trun tree (default 0 for MC)
#   subrun:  subrun number (default 0)
#   -s:      use work/evt<ID>_<SEL_TAG>/input/sp-frames.tar.bz2 (from run_select_evt.sh)
# Input:   input_files/input-<N>evt-<mode>/frames-dnn.tar.bz2  (per-event subset extracted on use)
# Output:  work/evt<ID>[_<SEL_TAG>]/magnify-evt<ID>-anode{0,1}.root
#           work/evt<ID>[_<SEL_TAG>]/sbnd-sp-frames-anode{0,1}.tar.bz2  (for run_select_evt.sh)

set -e

SBND_DIR=$(cd "$(dirname "$0")" && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}

. "$SBND_DIR/_runlib.sh"

usage() {
    cat <<EOF
Convert one (or all) SBND event's SP frames to per-anode Magnify ROOT files.

Usage: $(basename "$0") [mc|data] [-N n] [-s sel_tag] <idx|all> [run] [subrun]
       $(basename "$0") [mc|data] [-N n]            # list available events

  mc|data   input set (default mc)
  idx       1-based event index into the chosen sample/mode (see no-arg listing);
            'all' processes every event in parallel (cap nproc, SBND_MAX_JOBS=N)
  run/subrun  numbers stored in the ROOT Trun tree (default 0 0)
  -s        use work/evt<ID>_<sel_tag>/input/sp-frames.tar.bz2 (run_select_evt.sh)

Output: work/evt<EVT_ID>[_<sel_tag>]/magnify-evt<ID>-anode{0,1}.root (+ sp-frames archives)
EOF
    sbnd_common_help
}

MODE=mc
SEL_TAG=""
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -N) SBND_SAMPLE="$2"; shift 2 ;;
        -N*) SBND_SAMPLE="${1#-N}"; shift ;;
        mc|data) MODE="$1"; shift ;;
        -s) SEL_TAG="$2"; shift 2 ;;
        -s*) SEL_TAG="${1#-s}"; shift ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"

sbnd_check_sample "$MODE" || exit 1
load_events "$MODE" || exit 1

if [ $# -eq 0 ]; then
    list_events; exit 0
fi

if [ $# -lt 1 ]; then
    usage >&2
    exit 1
fi

IDX=$1
RUN=${2:-0}
SUBRUN=${3:-0}

process_event() {
    local IDX=$1
    local EVT_ID RUN_L SUBRUN_L WORKDIR SP_ARCHIVE
    local OUTPUT_PREFIX LOG SP_FRAME_PREFIX
    EVT_ID=$(lookup_evt_id "$IDX") || return 1
    RUN_L=${RUN:-0}
    SUBRUN_L=${SUBRUN:-0}

    if [ -n "$SEL_TAG" ]; then
        WORKDIR="$SBND_DIR/work/evt${EVT_ID}_${SEL_TAG}"
        SP_ARCHIVE="$WORKDIR/input/sp-frames.tar.bz2"
        if [ ! -s "$SP_ARCHIVE" ]; then
            echo "[skip] idx=$IDX evt=$EVT_ID: selection archive not found: $SP_ARCHIVE" >&2
            echo "  Run: ./run_select_evt.sh $IDX $SEL_TAG" >&2
            return 2
        fi
    else
        WORKDIR="$SBND_DIR/work/evt${EVT_ID}"
        SP_ARCHIVE="$WORKDIR/sp-frames.tar.bz2"
        # Extract this event's per-event subset from the mode frames archive.
        mkdir -p "$WORKDIR"
        echo "Extracting event $EVT_ID from $(sp_frames_archive "$MODE") ..."
        ensure_sp_frames "$MODE" "$EVT_ID" "$SP_ARCHIVE" || return 2
        echo "  → $SP_ARCHIVE"
    fi

    mkdir -p "$WORKDIR"

    OUTPUT_PREFIX="$WORKDIR/magnify-evt${EVT_ID}"
    LOG="$WORKDIR/wct_magnify_evt${EVT_ID}.log"
    SP_FRAME_PREFIX="$WORKDIR/sbnd-sp-frames"

    echo "Event index:  $IDX → EVT_ID=$EVT_ID"
    echo "Work dir:     $WORKDIR"
    echo "Input:        $SP_ARCHIVE"
    echo "Output:       ${OUTPUT_PREFIX}-anode{0,1}.root"
    echo "              ${SP_FRAME_PREFIX}-anode{0,1}.tar.bz2"
    echo "Log:          $LOG"

    cd "$SBND_DIR"
    rm -f "$LOG"
    wire-cell \
        -l stderr \
        -l "${LOG}:debug" \
        -L debug \
        --tla-str  "input=${SP_ARCHIVE}" \
        --tla-code "anode_indices=[0,1]" \
        --tla-str  "output_file_prefix=${OUTPUT_PREFIX}" \
        --tla-str  "sp_frame_prefix=${SP_FRAME_PREFIX}" \
        --tla-code "run=${RUN_L}" \
        --tla-code "subrun=${SUBRUN_L}" \
        --tla-code "event=${EVT_ID}" \
        -c wct-sp-to-magnify.jsonnet

    echo "Magnify done:"
    echo "  ${OUTPUT_PREFIX}-anode0.root"
    echo "  ${OUTPUT_PREFIX}-anode1.root"
    echo "SP frame archives (for woodpecker):"
    echo "  ${SP_FRAME_PREFIX}-anode0.tar.bz2"
    echo "  ${SP_FRAME_PREFIX}-anode1.tar.bz2"
}

mkdir -p "$SBND_DIR/work"
if [ "$IDX" = "all" ]; then
    batch_init
    echo "Found ${#SBND_EVENTS[@]} event(s). Parallel jobs: $BATCH_MAX"
    for _i in $(discover_event_indices); do
        _evtid="${SBND_EVENTS[$((_i-1))]}"
        _blogfile="$SBND_DIR/work/.batch_magnify_evt${_evtid}.log"
        batch_wait_slot
        ( process_event "$_i" ) > "$_blogfile" 2>&1 &
        BATCH_PIDS[$!]=$_i
        echo "  [start] idx=$_i evt=$_evtid  log: $_blogfile"
    done
    batch_drain
    batch_summary
    exit $?
else
    process_event "$IDX"
    exit $?
fi
