#!/bin/bash
# Run the FD-HD 1x2x6 workspace simulation from an EXTERNAL depo file (wcfm/docs/07 E1,
# events 101+: WC_FM_Sim pilot GENIE+G4 depos) for ONE event.  Fork by duplication of
# run_sim_evt.sh; no `all` mode on purpose (the gun runner's `all` re-simulates events 1-100).
# Usage: ./run_sim_depo_evt.sh [-n] [-r] <run> <evt>
#   -n   no noise (sim.signal_pipelines)
#   -r   also save the NF output (raw<N>) frames
# Input:  events/<run6>_<evt>.json (scripts/e1_make_events.py): depo_file, anodes, seed
#         <depo_file> = work/<run6>_<evt>/e1-depos-in.tar.bz2 (un-drifted depos)
# Output: work/<run6>_<evt>/sim-frames-anode<N>.tar.bz2 (gauss/wiener + masks),
#         work/<run6>_<evt>/sim-depos.tar.bz2 (drifted depos + priors),
#         wct_sim_<run6>_<evt>.log, time_sim.txt (abtest/timecmd.py: rc/wall_s/maxrss_kb), sim-provenance.txt
# Env: WCFM_SETARCH=1 (setarch x86_64 -R), WCT_TCMALLOC=off
set -e
WCFM_DIR=$(cd "$(dirname "$0")" && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCFM_DIR}:${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data

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
[ $# -lt 2 ] && { echo "Usage: $0 [-n] [-r] <run> <evt>" >&2; exit 1; }
RUN=$1; EVT=$2
[ "$EVT" = "all" ] && { echo "no 'all' mode: give one event" >&2; exit 1; }
RUN_STRIPPED=$(echo "$RUN" | sed 's/^0*//'); [ -z "$RUN_STRIPPED" ] && RUN_STRIPPED=0
RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")
[ "$EVT" -ge 101 ] || { echo "events < 101 belong to run_sim_evt.sh (gun); refusing" >&2; exit 1; }

EVJSON="$WCFM_DIR/events/${RUN_PADDED}_${EVT}.json"
[ -f "$EVJSON" ] || { echo "no event file $EVJSON" >&2; exit 1; }
WORKDIR="$WCFM_DIR/work/${RUN_PADDED}_${EVT}"
mkdir -p "$WORKDIR"
LOG="$WORKDIR/wct_sim_${RUN_PADDED}_${EVT}.log"
ANODES=$(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["anodes"]))' "$EVJSON")
SEED=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["seed"])' "$EVJSON")
DEPOIN=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["depo_file"])' "$EVJSON")
DEPOIN="$WCFM_DIR/$DEPOIN"
[ -f "$DEPOIN" ] || { echo "no input depo file $DEPOIN" >&2; exit 1; }
echo "event $RUN/$EVT: anodes $ANODES seed $SEED depos $DEPOIN noise=$NOISE"
cd "$WCFM_DIR"
rm -f "$LOG" "$WORKDIR"/sim-frames-*anode*.tar.bz2 "$WORKDIR/sim-depos.tar.bz2"
CFG="$WORKDIR/.wct-sim.json"
wcsonnet --tla-str "depo_inname=$DEPOIN" --tla-code "anode_indices=$ANODES" \
    --tla-str "output_prefix=$WORKDIR/sim-frames" --tla-str "depo_outname=$WORKDIR/sim-depos.tar.bz2" \
    --tla-code "seed=$SEED" --tla-code "noise=$NOISE" --tla-code "save_raw=$SAVE_RAW" \
    -o "$CFG" wct-sim-depofile-nf-sp.jsonnet
[ -s "$CFG" ] || { echo "wcsonnet failed" >&2; exit 1; }
{
    echo "toolkit=$(git -C $WCT_BASE/toolkit rev-parse --short HEAD)"
    echo "wcp=$(git -C $WCT_BASE/wcp-porting-img rev-parse --short HEAD)"
    echo "event_json=$EVJSON"
    echo "depo_input=$DEPOIN"
    echo "config=wct-sim-depofile-nf-sp.jsonnet"
    echo "wires=$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(sorted({n["data"]["filename"] for n in c if n.get("type")=="WireSchemaFile"})[0])' "$CFG")"
    echo "libimg=$(ls -la --time-style=full-iso $WCT_BASE/local/lib/libWireCellImg.so | awk '{print $6" "$7}')"
    echo "simulated=$(date -Is)"
} > "$WORKDIR/sim-provenance.txt"
rc=0
python3 "$WCT_BASE/wcp-porting-img/abtest/timecmd.py" "$WORKDIR/time_sim.txt" env $WC_PRELOAD $SETARCH wire-cell -l stderr -l "${LOG}:debug" -L debug -c "$CFG" || rc=$?
rm -f "$CFG"
echo "sim rc=$rc -> $WORKDIR"
exit $rc
