#!/bin/bash
# Run 3D imaging for one SBND event — standalone (no LArSoft).  Run with -h for help.
# Usage: ./run_img_evt.sh [mc|data] [-N n] [-a anode] [-s sel_tag] <idx|all>
#        ./run_img_evt.sh [mc|data] [-N n]   # list available events
#   mode:  mc (default) | data — selects input_files/input-<N>evt-<mode>/frames-dnn.tar.bz2
#   -N:    event-sample size (default 10); e.g. -N 100 uses input-100evt-<mode>
#   idx:   1-based event index into the chosen sample/mode; all = every event (parallel)
#   all:   process all events in parallel (up to nproc jobs; override with SBND_MAX_JOBS=N)
#   -a:    restrict to one anode (0 or 1); default processes both
#   -s:    use work/evt<ID>_<SEL_TAG>/input/sp-frames.tar.bz2 (from run_select_evt.sh)
# Input:  input_files/input-<N>evt-<mode>/frames-dnn.tar.bz2 (per-event subset extracted on use)
# Output: work/evt<ID>[_<SEL_TAG>]/icluster-apa{0,1}-{active,masked}.npz

set -e

SBND_DIR=$(cd "$(dirname "$0")" && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}

. "$SBND_DIR/_runlib.sh"

usage() {
    cat <<EOF
Run 3D imaging for one (or all) SBND event(s) — standalone, no LArSoft.

Usage: $(basename "$0") [mc|data] [-N n] [-a anode] [-s sel_tag] <idx|all>
       $(basename "$0") [mc|data] [-N n]            # list available events

  mc|data   input set (default mc)
  idx       1-based event index into the chosen sample/mode (see no-arg listing);
            'all' images every event in parallel (cap nproc, override SBND_MAX_JOBS=N)
  -a        restrict to one anode (0 or 1); default both
  -s        use a selection from run_select_evt.sh (work/evt<ID>_<tag>/input/)

Output: work/evt<EVT_ID>[_<sel_tag>]/icluster-apa{0,1}-{active,masked}.npz
EOF
    sbnd_common_help
}

MODE=mc
ANODE=""
SEL_TAG=""
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -N) SBND_SAMPLE="$2"; shift 2 ;;
        -N*) SBND_SAMPLE="${1#-N}"; shift ;;
        mc|data) MODE="$1"; shift ;;
        -a) ANODE="$2"; shift 2 ;;
        -a*) ANODE="${1#-a}"; shift ;;
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

IDX=$1

process_event() {
    local IDX=$1
    local EVT_ID WORKDIR SP_ARCHIVE ANODE_CODE TAG_SUFFIX LOG
    EVT_ID=$(lookup_evt_id "$IDX") || return 1

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
        # Self-extract this event's SP frames from the mode archive (fresh each run).
        mkdir -p "$WORKDIR"
        ensure_sp_frames "$MODE" "$EVT_ID" "$SP_ARCHIVE" || return 2
    fi

    if [ -n "$ANODE" ]; then
        ANODE_CODE="[$ANODE]"
        TAG_SUFFIX="_a${ANODE}"
    else
        ANODE_CODE="[0,1]"
        TAG_SUFFIX=""
    fi

    mkdir -p "$WORKDIR"
    LOG="$WORKDIR/wct_img_evt${EVT_ID}${TAG_SUFFIX}.log"

    echo "Event index:  $IDX → EVT_ID=$EVT_ID"
    echo "Work dir:     $WORKDIR"
    echo "Input:        $SP_ARCHIVE"
    echo "Anodes:       $ANODE_CODE"
    echo "Log:          $LOG"

    cd "$SBND_DIR"
    rm -f "$LOG"
    wire-cell \
        -l stderr \
        -l "${LOG}:debug" \
        -L debug \
        --tla-str  "input=${SP_ARCHIVE}" \
        --tla-code "anode_indices=${ANODE_CODE}" \
        --tla-str  "output_dir=${WORKDIR}" \
        -c wct-img-all.jsonnet

    echo "Imaging done -> $WORKDIR"
}

mkdir -p "$SBND_DIR/work"
if [ "$IDX" = "all" ]; then
    batch_init
    echo "Found ${#SBND_EVENTS[@]} event(s). Parallel jobs: $BATCH_MAX"
    for _i in $(discover_event_indices); do
        _evtid="${SBND_EVENTS[$((_i-1))]}"
        _blogfile="$SBND_DIR/work/.batch_img_evt${_evtid}.log"
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
