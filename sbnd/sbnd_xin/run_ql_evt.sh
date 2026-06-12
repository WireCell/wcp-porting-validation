#!/bin/bash
# Per-event SBND charge-light (Q/L) matching — standalone, self-contained.  -h for help.
# Usage: ./run_ql_evt.sh [mc|data] [-N n] <idx|all> [-a anode]
#        ./run_ql_evt.sh [mc|data] [-N n]            # list available events
#   mode:  mc (default) | data
#   -N:    event-sample size (default 10); e.g. -N 100 uses input-100evt-<mode>
#   idx:   1-based event index into the chosen sample/mode; all = every event (parallel)
#   -a:    restrict to one anode (0 or 1)
#
# Self-contained: reads the toolkit's OWN per-event imaging output
#   work/evt<ID>/icluster-apa{0,1}-{active,masked}.npz   (from ./run_img_evt.sh)
# plus that event's opflash (split from input_files/input-<N>evt-<mode>/), runs
# per-APA clustering + Q/L matching, and writes work/ql_evt<ID>/mabc-all-apa.zip
# (img + clustering + 2-view dead-area + op/Q-L layers).
#
# Prerequisite:  ./run_img_evt.sh <mode> [-N n] <idx>  (produces the per-event active+masked npz)
# Workflow:      run_img_evt.sh <mode>  ->  run_ql_evt.sh <mode>
#
# Both mc and data are wired: per-event imaging comes from
# input_files/input-<N>evt-<mode>/frames-dnn.tar.bz2 via run_img_evt.sh, and the
# opflash is split from input_files/input-<N>evt-<mode>/opflash_apa{0,1}.tar.gz.
# NB: opflash members are keyed by event id, so an event lacking opflash in the
# sample (e.g. parts of the 100evt set) is skipped with a clear message.

set -e

SBND_DIR=$(cd "$(dirname "$0")" && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WCT_BASE}/wire-cell-data/sbnd/photodet:${WIRECELL_PATH}

. "$SBND_DIR/_runlib.sh"

SEMIMODEL=semi-analytical-sbnd.json
JSONNET="$SBND_DIR/wct-clus-matching-perevt.jsonnet"
# Q/L drift / diffusion (documented values; same as run_clust_QL_evt.sh).
DL=6.2; DT=9.8; LIFETIME=6; DRIFTSPEED=1.563

usage() {
    cat <<EOF
Per-event SBND charge-light (Q/L) matching — standalone, self-contained.

Usage: $(basename "$0") [mc|data] [-N n] [-a anode] <idx|all>
       $(basename "$0") [mc|data] [-N n]            # list available events

  mc|data   input set (default mc)
  idx       1-based event index into the chosen sample/mode (see no-arg listing);
            'all' matches every event in parallel (cap nproc, SBND_MAX_JOBS=N)
  -a        restrict to one anode (0 or 1)
  -calib    also dump work/ql_evt<ID>/calib-evt<ID>.json for the ql_scan
            hand-scan viewer (matched zip output is unchanged)
  -cathode-diag  log the cathode-crossing TPC0/TPC1 offset three-vector
            diagnostic (grep QLCATHODE in the run log; output unchanged)
  -auto-mask  enable the per-event dynamic dead-PMT auto-mask (masks a PMT
            that is dead in THIS event while its live neighbours fire; off
            by default => byte-identical; grep QLAUTOMASK in the run log)

Requires: run_img_evt.sh first (work/evt<ID>/icluster-apa{0,1}-{active,masked}.npz).
Opflash comes from input_files/input-<N>evt-<mode>/opflash_apa{0,1}.tar.gz (keyed
by event id; events with no opflash in the sample are skipped).
Output: work/ql_evt<EVT_ID>/mabc-all-apa.zip
EOF
    sbnd_common_help
}

# --- Args ---
MODE=mc
ANODE=""
# Joint matching is the default: both TPCs go into ONE QLMatching node (which
# matches each APA and merges, replacing the separate PointTreeMerging). Pass
# --per-apa (or SBND_JOINT=0) to run the legacy two-node per-APA path instead.
# Both are byte-identical today; the joint node is where the joint algorithm lands.
JOINT=true
[ "${SBND_JOINT:-}" = "0" ] && JOINT=false
# -calib: also dump the per-event Q/L hand-scan calibration JSON
# (work/ql_evt<ID>/calib-evt<ID>.json) for the ql_scan viewer. Off by default;
# the matched mabc-all-apa.zip output is byte-identical with or without it.
CALIB=""
# CALIB_SUFFIX: optional suffix inserted before .json in the calib-dump filename
# (e.g. CALIB_SUFFIX=.nl -> calib-evt<ID>.nl.json), so an NL rerun does not clobber
# the linear dump the hand-scan was based on. Default empty = calib-evt<ID>.json.
CALIB_SUFFIX="${CALIB_SUFFIX:-}"
# PMT_NL: enable the per-PMT predicted-light non-linearity correction (threaded into
# the matching jsonnet as --tla-code pmt_nl). true (default, SBND going forward) maps
# predicted PE into the saturated space. PMT_NL=false = identity (OFF baseline).
PMT_NL="${PMT_NL:-true}"
# -cathode-diag: log the cathode-crossing TPC0/TPC1 offset three-vector diagnostic
# (grep "QLCATHODE" in the run log). Off by default; matched output is unchanged.
CATHODE=""
# -auto-mask: enable the per-event dynamic dead-PMT auto-mask (QLMatching auto_mask).
# Off by default (production byte-identical); masks a PMT that is dead in THIS event
# while its live neighbours fire (a run-dead channel absent from the static ch_mask).
AUTOMASK="false"
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -N) SBND_SAMPLE="$2"; shift 2 ;;
        -N*) SBND_SAMPLE="${1#-N}"; shift ;;
        mc|data) MODE="$1"; shift ;;
        -auto-mask|--auto-mask) AUTOMASK="true"; shift ;;   # before -a* (it starts with -a)
        -a) ANODE="$2"; shift 2 ;;
        -a*) ANODE="${1#-a}"; shift ;;
        -s|--per-apa|--separate) JOINT=false; shift ;;
        -calib|--calib) CALIB=1; shift ;;
        -cathode-diag|--cathode-diag) CATHODE=1; shift ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"

case "$MODE" in
    mc)   REALITY=sim ;;
    data) REALITY=data ;;
esac

sbnd_check_sample "$MODE" || exit 1
INPUT_DIR=$(sbnd_input_dir "$MODE")
[ -f "$JSONNET" ]   || { echo "ERROR: missing jsonnet: $JSONNET" >&2; exit 1; }

# Event-id list/order from the mode's frames-dnn archive (mode- and
# sample-agnostic; same order downstream pipelines stream events, so idx is
# stable).  The 100evt-mc set carries duplicate frames; load_events dedups.
load_events "$MODE" || exit 1
EVENT_IDS=("${SBND_EVENTS[@]}")

if [ $# -eq 0 ]; then
    echo "Sample: input-${SBND_SAMPLE}evt-$MODE   (${#EVENT_IDS[@]} events)"
    echo "Events for mode '$MODE' (idx -> EVT_ID):"
    for i in "${!EVENT_IDS[@]}"; do printf "  %2d -> %s\n" $((i + 1)) "${EVENT_IDS[$i]}"; done
    exit 0
fi

ANODE_CODE="[0,1]"
[ -n "$ANODE" ] && ANODE_CODE="[$ANODE]"

process_event() {
    local IDX=$1
    local EVT_ID="${EVENT_IDS[$((IDX - 1))]}"
    [ -n "$EVT_ID" ] || { echo "ERROR: invalid idx $IDX (1..${#EVENT_IDS[@]})" >&2; return 1; }

    local IMGDIR="$SBND_DIR/work/evt${EVT_ID}"     # per-event imaging output (run_img_evt.sh)
    local QLDIR="$SBND_DIR/work/ql_evt${EVT_ID}"    # isolated Q/L workspace + output
    local LOG="$QLDIR/wct_ql_evt${EVT_ID}.log"

    # Require the toolkit's per-event imaging output (active + masked, both anodes).
    local n kind f
    for n in 0 1; do
        for kind in active masked; do
            f="$IMGDIR/icluster-apa${n}-${kind}.npz"
            if [ ! -s "$f" ]; then
                echo "ERROR: missing $f — run ./run_img_evt.sh $IDX first" >&2
                return 2
            fi
        done
    done

    # Fresh isolated workspace: symlink the imaging npz, stage the per-event opflash.
    rm -rf "$QLDIR"; mkdir -p "$QLDIR"
    for n in 0 1; do
        ln -s "$IMGDIR/icluster-apa${n}-active.npz" "$QLDIR/icluster-apa${n}-active.npz"
        ln -s "$IMGDIR/icluster-apa${n}-masked.npz" "$QLDIR/icluster-apa${n}-masked.npz"
    done

    # Split this event's opflash from the bundled archive (members are uniquely
    # suffixed by event ident: opflash_tensorset_<ID>_* and opflash_tensor_<ID>_*).
    for n in 0 1; do
        local src="$INPUT_DIR/opflash_apa${n}.tar.gz"
        [ -s "$src" ] || { echo "ERROR: missing opflash: $src" >&2; return 1; }
        # Skip events that have no opflash in this sample (e.g. parts of the
        # 100evt set): without flashes Q/L matching cannot run.
        if ! tar tzf "$src" | grep -q "^opflash_tensorset_${EVT_ID}_"; then
            echo "[skip] evt $EVT_ID: no opflash for apa${n} in $(basename "$src")" >&2
            return 2
        fi
        local stage="$QLDIR/.opflash_stage_apa${n}"
        mkdir -p "$stage"
        tar xzf "$src" -C "$stage" --wildcards "opflash_tensorset_${EVT_ID}_*" "opflash_tensor_${EVT_ID}_*"
        ( cd "$stage" && tar czf "$QLDIR/opflash_apa${n}.tar.gz" opflash_tensorset_${EVT_ID}_* opflash_tensor_${EVT_ID}_* )
        rm -rf "$stage"
    done

    # Optional hand-scan calibration dump (one per-event JSON, both TPCs).
    local CALIB_TLA=()
    [ -n "$CALIB" ] && CALIB_TLA=(--tla-str "calib_dump=$QLDIR/calib-evt${EVT_ID}${CALIB_SUFFIX}.json")
    # Optional cathode-crossing offset diagnostic (logs QLCATHODE lines to $LOG).
    local CATHODE_TLA=()
    [ -n "$CATHODE" ] && CATHODE_TLA=(--tla-str "cathode_diag=on")

    echo "[evt $EVT_ID] Q/L matching (anodes $ANODE_CODE, joint=$JOINT${CALIB:+, calib}${CATHODE:+, cathode-diag}) -> $QLDIR/mabc-all-apa.zip"
    rm -f "$LOG"
    wire-cell \
        -l stderr -l "${LOG}:debug" -L debug \
        --tla-str  "input=$QLDIR" \
        --tla-code "anode_indices=${ANODE_CODE}" \
        --tla-str  "output_dir=$QLDIR" \
        --tla-code "run=0" --tla-code "subrun=0" --tla-code "event=${EVT_ID}" \
        --tla-str  "reality=$REALITY" \
        --tla-str  "semimodel_file=$SEMIMODEL" \
        --tla-code "DL=$DL" --tla-code "DT=$DT" \
        --tla-code "lifetime=$LIFETIME" --tla-code "driftSpeed=$DRIFTSPEED" \
        --tla-code "joint=$JOINT" \
        --tla-code "pmt_nl=$PMT_NL" \
        --tla-code "auto_mask=$AUTOMASK" \
        "${CALIB_TLA[@]}" \
        "${CATHODE_TLA[@]}" \
        -c "$JSONNET"
    echo "[evt $EVT_ID] done -> $QLDIR/mabc-all-apa.zip${CALIB:+ (+ calib-evt${EVT_ID}.json)}"
}

mkdir -p "$SBND_DIR/work"
IDX="$1"
if [ "$IDX" = "all" ]; then
    batch_init
    echo "Mode $MODE: ${#EVENT_IDS[@]} events. Parallel jobs: $BATCH_MAX"
    for i in $(seq 1 "${#EVENT_IDS[@]}"); do
        _evtid="${EVENT_IDS[$((i - 1))]}"
        _blog="$SBND_DIR/work/.batch_ql_evt${_evtid}.log"
        batch_wait_slot
        ( process_event "$i" ) > "$_blog" 2>&1 &
        BATCH_PIDS[$!]=$_evtid
        echo "  [start] idx=$i evt=$_evtid  log: $_blog"
    done
    batch_drain
    batch_summary
    exit $?
else
    process_event "$IDX"
    exit $?
fi
