#!/bin/bash
# Run clustering for one workspace event (both face volumes, all imaged anodes, pctree saved).
# Usage: ./run_clus_evt.sh [-O suffix] <run> <evt|all>
# Input:  work/<run6>_<evt><suffix>/clusters-apa-anode<N>-ms-{active,masked}.tar.gz (run_img_evt.sh)
# Output: work/<run6>_<evt><suffix>/mabc-anode<N>-face<F>.zip, mabc-groupf{0,1}.zip, mabc-all-apa.zip,
#         pctree-evt<evt>.tar.gz, wct_clus_<run6>_<evt>.log, time_clus.txt, clus-provenance.txt
# Env: WCFM_MAX_JOBS (batch cap, default 6), WCFM_SETARCH=1 (setarch x86_64 -R), WCT_TCMALLOC=off,
#      WCFM_CLUS_TLA="-S key=val ..." extra wcsonnet args
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

SUFFIX=""
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -O) SUFFIX="$2"; shift 2 ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"
[ $# -lt 2 ] && { echo "Usage: $0 [-O suffix] <run> <evt|all>" >&2; exit 1; }
RUN=$1; EVT=$2
RUN_STRIPPED=$(echo "$RUN" | sed 's/^0*//'); [ -z "$RUN_STRIPPED" ] && RUN_STRIPPED=0
RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")

process_event() {
    local RUN=$1 EVT=$2
    local WORKDIR="$WCFM_DIR/work/${RUN_PADDED}_${EVT}${SUFFIX}"
    local -a ANODES
    mapfile -t ANODES < <(ls "$WORKDIR"/clusters-apa-anode*-ms-active.tar.gz 2>/dev/null | sed -E 's|.*anode([0-9]+)-ms-active.*|\1|' | sort -n)
    [ ${#ANODES[@]} -eq 0 ] && { echo "no clusters-apa-anode*-ms-active.tar.gz in $WORKDIR" >&2; return 1; }
    local ANODE_CODE="[$(IFS=,; echo "${ANODES[*]}")]"
    local LOG="$WORKDIR/wct_clus_${RUN_PADDED}_${EVT}.log"
    local CFG="$WORKDIR/.wct-clus.json"
    local PCTREE="$WORKDIR/pctree-evt${EVT}.tar.gz"
    echo "event $RUN/$EVT: anodes $ANODE_CODE -> $WORKDIR"
    cd "$WCFM_DIR"
    wcsonnet -A "input=$WORKDIR" -S "anode_indices=$ANODE_CODE" -A "output_dir=$WORKDIR" \
        -S "run=$RUN_STRIPPED" -S "subrun=1" -S "event=$EVT" -A "save_tensors=$PCTREE" \
        ${WCFM_CLUS_TLA:-} -o "$CFG" wct-clustering.jsonnet
    [ -s "$CFG" ] || { echo "wcsonnet failed" >&2; return 1; }
    local CLUS_WIRES IMG_WIRES
    CLUS_WIRES=$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(sorted({n["data"]["filename"] for n in c if n.get("type")=="WireSchemaFile"})[0])' "$CFG")
    IMG_WIRES=$(awk -F= '$1=="wires"{print $2}' "$WORKDIR/img-provenance.txt" 2>/dev/null)
    [ "$IMG_WIRES" = "$CLUS_WIRES" ] || { echo "ERROR: imaging wires=$IMG_WIRES but clustering compiles wires=$CLUS_WIRES" >&2; rm -f "$CFG"; return 3; }
    {
        echo "toolkit=$(git -C $WCT_BASE/toolkit rev-parse --short HEAD)"
        echo "wcp=$(git -C $WCT_BASE/wcp-porting-img rev-parse --short HEAD)"
        echo "wires=$CLUS_WIRES"
        echo "anodes=$ANODE_CODE"
        echo "clustered=$(date -Is)"
    } > "$WORKDIR/clus-provenance.txt"
    rm -f "$LOG" "$WORKDIR"/mabc-*.zip "$PCTREE" "$WORKDIR"/trash-*.tar.gz
    local rc=0
    python3 "$WCT_BASE/wcp-porting-img/abtest/timecmd.py" "$WORKDIR/time_clus.txt" \
        env $WC_PRELOAD GOGC=off $SETARCH wire-cell -l stderr -l "${LOG}:debug" -L debug -c "$CFG" || rc=$?
    rm -f "$CFG"
    echo "clustering rc=$rc $(cat "$WORKDIR/time_clus.txt" | tr '\n' ' ') -> $WORKDIR"
    return $rc
}

if [ "$EVT" = "all" ]; then
    batch_init
    mapfile -t _events < <(ls "$WCFM_DIR/work/${RUN_PADDED}_"*/clusters-apa-anode*-ms-active.tar.gz 2>/dev/null | sed -E "s|.*/${RUN_PADDED}_([0-9]+)/.*|\1|" | sort -n -u)
    [ ${#_events[@]} -eq 0 ] && { echo "no imaged events for run $RUN" >&2; exit 1; }
    echo "Found ${#_events[@]} event(s): ${_events[*]}  (parallel jobs: $BATCH_MAX)"
    for _e in "${_events[@]}"; do
        batch_wait_slot
        ( process_event "$RUN" "$_e" ) > "$WCFM_DIR/work/.batch_clus_${RUN_PADDED}_${_e}.log" 2>&1 &
        BATCH_PIDS[$!]=$_e
        echo "  [start] evt=$_e"
    done
    batch_drain
    batch_summary; exit $?
else
    ( process_event "$RUN" "$EVT" ); exit $?
fi
