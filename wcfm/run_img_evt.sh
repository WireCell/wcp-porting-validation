#!/bin/bash
# Run imaging (+ BlobDepoFill truth tiers) for one workspace event, one wire-cell process per anode.
# Usage: ./run_img_evt.sh [-a anode] [-T] [-t time_offset_us] [-C [-L wires]] [-O suffix] <run> <evt|all>
#   -a N   only this anode ident (default: every anode with a sim-frames file)
#   -T     no truth tiers (plain PDHD-style imaging)
#   -t US  BlobDepoFill time_offset in microseconds (default: wcfm_params depofill_time_offset)
#   -C     sub-blob generator: BlobCutting ahead of BlobClustering (doc 03; default off)
#   -L N   BlobCutting length_threshold in wires (default 20; only with -C)
#   -O S   write into work/<run6>_<evt><S>/ (reads the frames from work/<run6>_<evt>/); for scans
# Input:  work/<run6>_<evt>/sim-frames-anode<N>.tar.bz2, sim-depos.tar.bz2 (run_sim_evt.sh)
# Output: work/<run6>_<evt>/clusters-apa-anode<N>-ms-{active,masked}.tar.gz,
#         clusters-{tru0,tru}-anode<N>-ms-active.tar.gz, dead-channels-anode<N>.json,
#         wct_img_<run6>_<evt>_a<N>.log, time_img_a<N>.txt, img-provenance.txt
# Env: WCFM_MAX_JOBS (batch cap, default 6), WCFM_SETARCH=1 (setarch x86_64 -R), WCT_TCMALLOC=off
set -e
WCFM_DIR=$(cd "$(dirname "$0")" && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCFM_DIR}:${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data
. "$WCFM_DIR/_runlib.sh"

TCMALLOC_SO=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
WC_PRELOAD=""
[ "${WCT_TCMALLOC:-on}" = "on" ] && [ -f "$TCMALLOC_SO" ] && WC_PRELOAD="LD_PRELOAD=$TCMALLOC_SO"
SETARCH=""
[ "${WCFM_SETARCH:-0}" = "1" ] && SETARCH="setarch x86_64 -R"

ANODE=""; TRUTH=1; TOFF_US=""; SUFFIX=""; CUT=0; CUT_LEN=""
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -a) ANODE="$2"; shift 2 ;;
        -T) TRUTH=0; shift ;;
        -t) TOFF_US="$2"; shift 2 ;;
        -O) SUFFIX="$2"; shift 2 ;;
        -C) CUT=1; shift ;;
        -L) CUT_LEN="$2"; shift 2 ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"
[ $# -lt 2 ] && { echo "Usage: $0 [-a anode] [-T] [-t time_offset_us] [-C [-L wires]] [-O suffix] <run> <evt|all>" >&2; exit 1; }
RUN=$1; EVT=$2
RUN_STRIPPED=$(echo "$RUN" | sed 's/^0*//'); [ -z "$RUN_STRIPPED" ] && RUN_STRIPPED=0
RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")

process_event() {
    local RUN=$1 EVT=$2
    local INDIR="$WCFM_DIR/work/${RUN_PADDED}_${EVT}"
    local WORKDIR="$WCFM_DIR/work/${RUN_PADDED}_${EVT}${SUFFIX}"
    local DEPOS="$INDIR/sim-depos.tar.bz2"
    mkdir -p "$WORKDIR"
    local -a ANODES
    if [ -n "$ANODE" ]; then ANODES=("$ANODE")
    else mapfile -t ANODES < <(ls "$INDIR"/sim-frames-anode*.tar.bz2 2>/dev/null | sed -E 's|.*anode([0-9]+)\.tar\.bz2|\1|' | sort -n); fi
    [ ${#ANODES[@]} -eq 0 ] && { echo "no sim-frames-anode*.tar.bz2 in $INDIR" >&2; return 1; }
    local DEPO_ARG=()
    if [ "$TRUTH" = 1 ]; then
        [ -f "$DEPOS" ] || { echo "no $DEPOS (use -T for no truth)" >&2; return 1; }
        DEPO_ARG=(-A "depos=$DEPOS")
    fi
    local TOFF_ARG=()
    [ -n "$TOFF_US" ] && TOFF_ARG=(-S "time_offset=${TOFF_US}*1000")
    local CUT_ARG=()
    [ "$CUT" = 1 ] && CUT_ARG=(-S "blob_cutting=true")
    [ "$CUT" = 1 ] && [ -n "$CUT_LEN" ] && CUT_ARG+=(-S "cut_length=$CUT_LEN")
    echo "event $RUN/$EVT: anodes ${ANODES[*]} truth=$TRUTH time_offset_us=${TOFF_US:-default} blob_cutting=$CUT${CUT_LEN:+/$CUT_LEN} -> $WORKDIR"
    cd "$WCFM_DIR"
    local ai CFG rc=0
    for ai in "${ANODES[@]}"; do
        CFG="$WORKDIR/.wct-img-a${ai}.json"
        wcsonnet -A "input_prefix=$INDIR/sim-frames" -S "anode_indices=[$ai]" -A "output_dir=$WORKDIR" \
            "${DEPO_ARG[@]}" "${TOFF_ARG[@]}" "${CUT_ARG[@]}" -o "$CFG" wct-img-all.jsonnet
        [ -s "$CFG" ] || { echo "wcsonnet failed for anode $ai" >&2; return 1; }
    done
    {
        echo "toolkit=$(git -C $WCT_BASE/toolkit rev-parse --short HEAD)"
        echo "wcp=$(git -C $WCT_BASE/wcp-porting-img rev-parse --short HEAD)"
        echo "wires=$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(sorted({n["data"]["filename"] for n in c if n.get("type")=="WireSchemaFile"})[0])' "$WORKDIR/.wct-img-a${ANODES[0]}.json")"
        echo "truth=$TRUTH"
        echo "blob_cutting=$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(sorted({n["data"]["length_threshold"] for n in c if n.get("type")=="BlobCutting"} or {"off"})[0])' "$WORKDIR/.wct-img-a${ANODES[0]}.json")"
        echo "time_offset=$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(sorted({n["data"]["time_offset"] for n in c if n.get("type")=="BlobDepoFill"} or {"none"})[0])' "$WORKDIR/.wct-img-a${ANODES[0]}.json")"
        echo "imaged=$(date -Is)"
    } > "$WORKDIR/img-provenance.txt"
    for ai in "${ANODES[@]}"; do
        CFG="$WORKDIR/.wct-img-a${ai}.json"
        local LOG="$WORKDIR/wct_img_${RUN_PADDED}_${EVT}_a${ai}.log"
        rm -f "$LOG" "$WORKDIR"/clusters-*-anode${ai}-ms-*.tar.gz
        python3 "$WCT_BASE/wcp-porting-img/abtest/timecmd.py" "$WORKDIR/time_img_a${ai}.txt" \
            env $WC_PRELOAD GOGC=off $SETARCH wire-cell -l stderr -l "${LOG}:debug" -L debug -c "$CFG" || rc=$?
        rm -f "$CFG"
        echo "anode $ai imaging rc=$rc $(cat "$WORKDIR/time_img_a${ai}.txt" | tr '\n' ' ')"
        [ $rc -ne 0 ] && return $rc
        # dead-channel sidecar (the GNN doc sec 7.3 input): the frame's "bad" channel mask
        python3 - "$INDIR/sim-frames-anode${ai}.tar.bz2" "$WORKDIR/dead-channels-anode${ai}.json" <<'EOF'
import json, sys
from wirecell.util import ario
f = ario.load(sys.argv[1]); out = {}
for k in f.keys():
    if k.startswith('chanmask_'):
        a = f[k]; out[k] = [[int(x) for x in row] for row in a.reshape(-1, 3)]
json.dump(out, open(sys.argv[2], 'w'))
EOF
    done
    echo "Imaging done -> $WORKDIR"
    return $rc
}

if [ "$EVT" = "all" ]; then
    batch_init
    mapfile -t _events < <(ls -d "$WCFM_DIR/work/${RUN_PADDED}_"*/sim-depos.tar.bz2 2>/dev/null | sed -E "s|.*/${RUN_PADDED}_([0-9]+)/.*|\1|" | sort -n -u)
    [ ${#_events[@]} -eq 0 ] && { echo "no simulated events for run $RUN" >&2; exit 1; }
    echo "Found ${#_events[@]} event(s): ${_events[*]}  (parallel jobs: $BATCH_MAX)"
    for _e in "${_events[@]}"; do
        batch_wait_slot
        ( process_event "$RUN" "$_e" ) > "$WCFM_DIR/work/.batch_img_${RUN_PADDED}_${_e}.log" 2>&1 &
        BATCH_PIDS[$!]=$_e
        echo "  [start] evt=$_e"
    done
    batch_drain
    batch_summary; exit $?
else
    ( process_event "$RUN" "$EVT" ); exit $?
fi
