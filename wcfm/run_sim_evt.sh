#!/bin/bash
# Run the FD-HD 1x2x6 workspace simulation (iso-track gun -> NF -> SP) for one event.
# Usage: ./run_sim_evt.sh [-n] [-r] <run> <evt|all>
#   -n   no noise (sim.signal_pipelines)
#   -r   also save the NF output (raw<N>) frames
# Input:  events/<run6>_<evt>.json (gen_iso_tracks.py): tracks (cm), anodes, seed
# Output: work/<run6>_<evt>/sim-frames-anode<N>.tar.bz2 (gauss/wiener + masks),
#         work/<run6>_<evt>/sim-depos.tar.bz2 (drifted depos + priors),
#         wct_sim_<run6>_<evt>.log, time_sim.txt (abtest/timecmd.py: rc/wall_s/maxrss_kb), sim-provenance.txt
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

NOISE=true; SAVE_RAW=false
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -n) NOISE=false; shift ;;
        -r) SAVE_RAW=true; shift ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"
[ $# -lt 2 ] && { echo "Usage: $0 [-n] [-r] <run> <evt|all>" >&2; exit 1; }
RUN=$1; EVT=$2
RUN_STRIPPED=$(echo "$RUN" | sed 's/^0*//'); [ -z "$RUN_STRIPPED" ] && RUN_STRIPPED=0
RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")

process_event() {
    local RUN=$1 EVT=$2
    local EVJSON="$WCFM_DIR/events/${RUN_PADDED}_${EVT}.json"
    [ -f "$EVJSON" ] || { echo "no event file $EVJSON" >&2; return 1; }
    local WORKDIR="$WCFM_DIR/work/${RUN_PADDED}_${EVT}"
    mkdir -p "$WORKDIR"
    local LOG="$WORKDIR/wct_sim_${RUN_PADDED}_${EVT}.log"
    local TRACKS ANODES SEED STEP
    TRACKS=$(python3 -c 'import json,sys; e=json.load(open(sys.argv[1])); print(json.dumps([{k:t[k] for k in ("tail","head","charge")} for t in e["tracks"]]))' "$EVJSON")
    ANODES=$(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["anodes"]))' "$EVJSON")
    SEED=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["seed"])' "$EVJSON")
    STEP=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["step_mm"])' "$EVJSON")
    echo "event $RUN/$EVT: anodes $ANODES seed $SEED step $STEP mm noise=$NOISE"
    cd "$WCFM_DIR"
    rm -f "$LOG" "$WORKDIR"/sim-frames-*anode*.tar.bz2 "$WORKDIR/sim-depos.tar.bz2"
    local CFG="$WORKDIR/.wct-sim.json"
    wcsonnet --tla-code "tracks=$TRACKS" --tla-code "anode_indices=$ANODES" \
        --tla-str "output_prefix=$WORKDIR/sim-frames" --tla-str "depo_outname=$WORKDIR/sim-depos.tar.bz2" \
        --tla-code "seed=$SEED" --tla-code "noise=$NOISE" --tla-code "step_mm=$STEP" --tla-code "save_raw=$SAVE_RAW" \
        -o "$CFG" wct-sim-iso-track-nf-sp.jsonnet
    [ -s "$CFG" ] || { echo "wcsonnet failed" >&2; return 1; }
    {
        echo "toolkit=$(git -C $WCT_BASE/toolkit rev-parse --short HEAD)"
        echo "wcp=$(git -C $WCT_BASE/wcp-porting-img rev-parse --short HEAD)"
        echo "event_json=$EVJSON"
        echo "wires=$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(sorted({n["data"]["filename"] for n in c if n.get("type")=="WireSchemaFile"})[0])' "$CFG")"
        echo "simulated=$(date -Is)"
    } > "$WORKDIR/sim-provenance.txt"
    local rc=0
    python3 "$WCT_BASE/wcp-porting-img/abtest/timecmd.py" "$WORKDIR/time_sim.txt" env $WC_PRELOAD $SETARCH wire-cell -l stderr -l "${LOG}:debug" -L debug -c "$CFG" || rc=$?
    rm -f "$CFG"
    echo "sim rc=$rc -> $WORKDIR"
    return $rc
}

mkdir -p "$WCFM_DIR/work"
if [ "$EVT" = "all" ]; then
    batch_init
    mapfile -t _events < <(ls "$WCFM_DIR/events/${RUN_PADDED}_"*.json 2>/dev/null | sed -E 's|.*_([0-9]+)\.json|\1|' | sort -n)
    [ ${#_events[@]} -eq 0 ] && { echo "no events/${RUN_PADDED}_*.json" >&2; exit 1; }
    echo "Found ${#_events[@]} event(s): ${_events[*]}  (parallel jobs: $BATCH_MAX)"
    for _e in "${_events[@]}"; do
        batch_wait_slot
        ( process_event "$RUN" "$_e" ) > "$WCFM_DIR/work/.batch_sim_${RUN_PADDED}_${_e}.log" 2>&1 &
        BATCH_PIDS[$!]=$_e
        echo "  [start] evt=$_e"
    done
    batch_drain
    batch_summary; exit $?
else
    ( process_event "$RUN" "$EVT" ); exit $?
fi
