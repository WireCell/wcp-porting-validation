#!/bin/bash
# Per-event SBND charge-light (Q/L) matching — standalone, self-contained.
# Usage: ./run_ql_evt.sh [mc|data] <idx|all> [-a anode]
#        ./run_ql_evt.sh [mc|data]            # list available events
#   mode:  mc (default) | data
#   idx:   1-based event index into the mode's event list; all = every event (parallel)
#   -a:    restrict to one anode (0 or 1)
#
# Self-contained: reads the toolkit's OWN per-event imaging output
#   work/evt<ID>/icluster-apa{0,1}-{active,masked}.npz   (from ./run_img_evt.sh)
# plus that event's opflash (split from input_files/input-10evt-<mode>/), runs
# per-APA clustering + Q/L matching, and writes work/ql_evt<ID>/mabc-all-apa.zip
# (img + clustering + 2-view dead-area + op/Q-L layers).
#
# Prerequisite:  ./run_img_evt.sh <mode> <idx>   (produces the per-event active+masked npz)
# Workflow:      run_img_evt.sh <mode>  ->  run_ql_evt.sh <mode>
#
# Both mc and data are wired: per-event imaging comes from
# input_files/input-10evt-<mode>/frames-dnn.tar.bz2 via run_img_evt.sh, and the
# opflash is split from input_files/input-10evt-<mode>/opflash_apa{0,1}.tar.gz.

set -e

SBND_DIR=$(cd "$(dirname "$0")" && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WCT_BASE}/wire-cell-data/sbnd/photodet:${WIRECELL_PATH}

. "$SBND_DIR/_runlib.sh"

SEMIMODEL=semi-analytical-sbnd.json
JSONNET="$SBND_DIR/wct-clus-matching-perevt.jsonnet"
# Q/L drift / diffusion (documented values; same as run_clust_QL_evt.sh).
DL=6.2; DT=9.8; LIFETIME=6; DRIFTSPEED=1.563

# --- Args ---
MODE=mc
ANODE=""
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        mc|data) MODE="$1"; shift ;;
        -a) ANODE="$2"; shift 2 ;;
        -a*) ANODE="${1#-a}"; shift ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"

case "$MODE" in
    mc)   REALITY=sim ;;
    data) REALITY=data ;;
esac

INPUT_DIR="$SBND_DIR/input_files/input-10evt-$MODE"
[ -d "$INPUT_DIR" ] || { echo "ERROR: missing input dir: $INPUT_DIR" >&2; exit 1; }
[ -f "$JSONNET" ]   || { echo "ERROR: missing jsonnet: $JSONNET" >&2; exit 1; }

# Event-id list/order, derived from the mode's active npz (mode-agnostic; same
# order ClusterFileSource streams, so idx is stable).
mapfile -t EVENT_IDS < <(python3 -c "
import numpy as np, re
z = np.load('$INPUT_DIR/icluster-apa0-active.npz')
seen = []
for k in z.files:
    m = re.match(r'cluster_(\d+)_', k)
    if m and m.group(1) not in seen:
        seen.append(m.group(1))
print('\n'.join(seen))
")
[ "${#EVENT_IDS[@]}" -gt 0 ] || { echo "ERROR: no events in $INPUT_DIR/icluster-apa0-active.npz" >&2; exit 1; }

if [ $# -eq 0 ]; then
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
        local stage="$QLDIR/.opflash_stage_apa${n}"
        mkdir -p "$stage"
        tar xzf "$src" -C "$stage" --wildcards "opflash_tensorset_${EVT_ID}_*" "opflash_tensor_${EVT_ID}_*"
        ( cd "$stage" && tar czf "$QLDIR/opflash_apa${n}.tar.gz" opflash_tensorset_${EVT_ID}_* opflash_tensor_${EVT_ID}_* )
        rm -rf "$stage"
    done

    echo "[evt $EVT_ID] Q/L matching (anodes $ANODE_CODE) -> $QLDIR/mabc-all-apa.zip"
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
        -c "$JSONNET"
    echo "[evt $EVT_ID] done -> $QLDIR/mabc-all-apa.zip"
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
