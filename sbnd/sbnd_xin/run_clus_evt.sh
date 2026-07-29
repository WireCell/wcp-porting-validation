#!/bin/bash
# Run SBND per-APA and all-APA blob clustering — standalone (no LArSoft).  -h for help.
# Usage: ./run_clus_evt.sh [mc|data] [-N n] [-a anode] [-s sel_tag] <idx|all> [run] [subrun]
#        ./run_clus_evt.sh [mc|data] [-N n]   # list available events
#   mode:  mc (default) | data — selects the event list (clustering graph is mode-agnostic)
#   -N:    event-sample size (default 10); e.g. -N 100 uses input-100evt-<mode>
#   idx:   1-based event index into the chosen sample/mode; all = every event (parallel)
#   all:   process all events in parallel (up to nproc jobs; override with SBND_MAX_JOBS=N)
#   run:   run number stored in bee RSE metadata (default 0)
#   subrun: subrun number (default 0)
#   -a:    restrict to one anode (0 or 1)
#   -s:    use work/evt<ID>_<SEL_TAG>/ as working directory
# Input:  work/evt<ID>[_<SEL_TAG>]/icluster-apa{0,1}-{active,masked}.npz (from run_img_evt.sh)
# Output: work/evt<ID>[_<SEL_TAG>]/mabc-<anode>-face0.zip, mabc-all-apa.zip

set -e

SBND_DIR=$(cd "$(dirname "$0")" && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}

# Preload tcmalloc for the clustering wire-cell process.  Safe since the
# BlobLess pointer-order fix: verified run-to-run deterministic AND
# glibc==tcmalloc byte-identical on 5 events x 3 archives (incl. the
# historically nondeterministic evt138670); see
# clus/docs/imgclus-optimization-log.md entry 19.  Disable with
# WCT_TCMALLOC=off.
TCMALLOC_SO=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
WC_PRELOAD=""
if [ "${WCT_TCMALLOC:-on}" = "on" ] && [ -f "$TCMALLOC_SO" ]; then
    WC_PRELOAD="LD_PRELOAD=$TCMALLOC_SO"
fi

. "$SBND_DIR/_runlib.sh"

usage() {
    cat <<EOF
Run SBND per-APA + all-APA blob clustering on imaged clusters — no LArSoft.

Usage: $(basename "$0") [mc|data] [-N n] [-a anode] [-s sel_tag] <idx|all> [run] [subrun]
       $(basename "$0") [mc|data] [-N n]            # list available events

  mc|data   input set (default mc); clustering graph itself is mode-agnostic
  idx       1-based event index into the chosen sample/mode (see no-arg listing);
            'all' clusters every event in parallel (cap nproc, SBND_MAX_JOBS=N)
  run/subrun  RSE metadata stored in the Bee output (default 0 0)
  -a        restrict to one anode (0 or 1)
  -s        use work/evt<ID>_<sel_tag>/ (from run_select_evt.sh)

Requires: run_img_evt.sh first (per-event icluster npz).
Output:   work/evt<EVT_ID>[_<sel_tag>]/mabc-<anode>-face0.zip, mabc-all-apa.zip
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
RUN=${2:-0}
SUBRUN=${3:-0}
case "$MODE" in mc) REALITY=sim ;; data) REALITY=data ;; esac

# True if the .npz exists, is nonempty on disk, AND contains at least one array.
# A "no clusters" run still produces a 22-byte zip header (no .npy inside),
# which makes ClusterFileSource hit EOS at call=0 and stalls the all-apa
# PointTreeMerging fan-in (multiplicity expects every branch to deliver).
npz_has_content() {
    [ -s "$1" ] || return 1
    python3 -c "import numpy as np,sys; sys.exit(0 if len(np.load(sys.argv[1]).files)>0 else 1)" "$1" 2>/dev/null
}

process_event() {
    local IDX=$1
    local EVT_ID RUN_L SUBRUN_L WORKDIR
    local candidates KEEP ANODE_CODE TAG_SUFFIX LOG
    EVT_ID=$(lookup_evt_id "$IDX") || return 1
    RUN_L=${RUN:-0}
    SUBRUN_L=${SUBRUN:-0}

    if [ -n "$SEL_TAG" ]; then
        WORKDIR="$SBND_DIR/work/evt${EVT_ID}_${SEL_TAG}"
    else
        WORKDIR="$SBND_DIR/work/evt${EVT_ID}"
    fi

    if [ -n "$ANODE" ]; then
        candidates=("$ANODE")
    else
        candidates=(0 1)
    fi

    KEEP=()
    for a in "${candidates[@]}"; do
        local npz="$WORKDIR/icluster-apa${a}-active.npz"
        if npz_has_content "$npz"; then
            KEEP+=("$a")
        else
            echo "WARNING: skipping anode $a — $npz is missing or has no active clusters" >&2
        fi
    done

    if [ ${#KEEP[@]} -eq 0 ]; then
        echo "[skip] idx=$IDX evt=$EVT_ID: no non-empty icluster-apa*-active.npz found in $WORKDIR" >&2
        echo "  Run: ./run_img_evt.sh $IDX" >&2
        return 2
    fi

    ANODE_CODE="[$(IFS=,; echo "${KEEP[*]}")]"
    if [ ${#KEEP[@]} -eq 1 ]; then
        TAG_SUFFIX="_a${KEEP[0]}"
    else
        TAG_SUFFIX=""
    fi

    mkdir -p "$WORKDIR"
    LOG="$WORKDIR/wct_clus_evt${EVT_ID}${TAG_SUFFIX}.log"

    echo "Event index:  $IDX → EVT_ID=$EVT_ID"
    echo "Work dir:     $WORKDIR"
    echo "Anodes:       $ANODE_CODE"
    echo "RSE:          run=$RUN_L subrun=$SUBRUN_L event=$EVT_ID"
    echo "Log:          $LOG"

    cd "$SBND_DIR"
    rm -f "$LOG"
    # Pre-compile the config with wcsonnet and feed wire-cell pure JSON,
    # with GOGC=off: with a pure-JSON config wire-cell never evaluates
    # jsonnet (no Go heap) and GOGC=off disables the embedded gojsonnet
    # runtime's collector entirely, including the 2-minute periodic forced
    # GC that was identified as an intermittent-SIGABRT vector on long
    # jobs (same pattern as pdhd/pdvd run_clus_evt.sh).
    local CFG_JSON="$WORKDIR/.wct-clus${TAG_SUFFIX}.json"
    wcsonnet \
        -A "input=${WORKDIR}" \
        -S "anode_indices=${ANODE_CODE}" \
        -A "output_dir=${WORKDIR}" \
        -S "run=${RUN_L}" \
        -S "subrun=${SUBRUN_L}" \
        -S "event=${EVT_ID}" \
        -A "reality=${REALITY}" \
        -S "DL=4.0" \
        -S "DT=8.8" \
        -S "lifetime=35" \
        -S "driftSpeed=1.563" \
        -o "$CFG_JSON" wct-clustering.jsonnet
    if [ ! -s "$CFG_JSON" ]; then
        echo "ERROR: wcsonnet failed to compile wct-clustering.jsonnet" >&2
        return 1
    fi
    env $WC_PRELOAD GOGC=off wire-cell \
        -l stderr \
        -l "${LOG}:debug" \
        -L debug \
        -c "$CFG_JSON"
    rm -f "$CFG_JSON"

    echo "Clustering done -> $WORKDIR"
}

mkdir -p "$SBND_DIR/work"
if [ "$IDX" = "all" ]; then
    batch_init
    echo "Found ${#SBND_EVENTS[@]} event(s). Parallel jobs: $BATCH_MAX"
    for _i in $(discover_event_indices); do
        _evtid="${SBND_EVENTS[$((_i-1))]}"
        _blogfile="$SBND_DIR/work/.batch_clus_evt${_evtid}.log"
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
