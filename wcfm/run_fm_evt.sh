#!/bin/bash
# Run the FM feature stage (wcfm/docs/04, F3) for one workspace event, one wire-cell process per anode.
# Usage: ./run_fm_evt.sh [-a anode] [-D cpu|gpu] [-M model.ts] [-F] [-O suffix] <run> <evt|all>
#   -a N   only this anode ident (default: every anode with a sim-frames file)
#   -D dev TorchService device: cpu (default) or gpu
#   -M ts  model path resolved through WIRECELL_PATH (default fm/dune10kt-1x2x6/kd_uni_mbv3_a1_inf01.ts)
#   -F     store features as f4 instead of IEEE-half bits (the parity-gate arm)
#   -O S   write into work/<run6>_<evt><S>/ (reads the frames from work/<run6>_<evt>/)
# Input:  work/<run6>_<evt>/sim-frames-anode<N>.tar.bz2 (run_sim_evt.sh)
# Output: work/<run6>_<evt><S>/fm-features-anode<N>.tar.gz, wct_fm_<run6>_<evt>_a<N>.log,
#         time_fm_a<N>.txt (rc/wall/maxrss), gpu_fm_a<N>.txt (peak VRAM delta, -D gpu), fm-provenance.txt
# Env: WCFM_MAX_JOBS (batch cap, default 4), WCFM_LIBPIN (dir prepended to LD_LIBRARY_PATH: a private
#      or pinned library set), OMP_NUM_THREADS / TORCH threads via WCFM_TORCH_THREADS (default 8)
set -e
WCFM_DIR=$(cd "$(dirname "$0")" && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCFM_DIR}:${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data
[ -n "$WCFM_LIBPIN" ] && export LD_LIBRARY_PATH=${WCFM_LIBPIN}:${LD_LIBRARY_PATH}
export OMP_NUM_THREADS=${WCFM_TORCH_THREADS:-8}
. "$WCFM_DIR/_runlib.sh"

TCMALLOC_SO=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
WC_PRELOAD=""
[ "${WCT_TCMALLOC:-on}" = "on" ] && [ -f "$TCMALLOC_SO" ] && WC_PRELOAD="LD_PRELOAD=$TCMALLOC_SO"

ANODE=""; DEVICE=cpu; MODEL=fm/dune10kt-1x2x6/kd_uni_mbv3_a1_inf01.ts; HALF=true; SUFFIX=""
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -a) ANODE="$2"; shift 2 ;;
        -D) DEVICE="$2"; shift 2 ;;
        -M) MODEL="$2"; shift 2 ;;
        -F) HALF=false; shift ;;
        -O) SUFFIX="$2"; shift 2 ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"
[ $# -lt 2 ] && { echo "Usage: $0 [-a anode] [-D cpu|gpu] [-M model.ts] [-F] [-O suffix] <run> <evt|all>" >&2; exit 1; }
RUN=$1; EVT=$2
RUN_STRIPPED=$(echo "$RUN" | sed 's/^0*//'); [ -z "$RUN_STRIPPED" ] && RUN_STRIPPED=0
RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")

MODEL_FILE=$(python3 -c 'import os,sys
for d in os.environ["WIRECELL_PATH"].split(":"):
    p=os.path.join(d, sys.argv[1])
    if os.path.exists(p): print(p); break' "$MODEL")
[ -f "$MODEL_FILE" ] || { echo "model $MODEL not found on WIRECELL_PATH" >&2; exit 1; }
MODEL_SHA=$(sha256sum "$MODEL_FILE" | cut -d' ' -f1)

process_event() {
    local RUN=$1 EVT=$2
    local INDIR="$WCFM_DIR/work/${RUN_PADDED}_${EVT}"
    local WORKDIR="$WCFM_DIR/work/${RUN_PADDED}_${EVT}${SUFFIX}"
    mkdir -p "$WORKDIR"
    local -a ANODES
    if [ -n "$ANODE" ]; then ANODES=("$ANODE")
    else mapfile -t ANODES < <(ls "$INDIR"/sim-frames-anode*.tar.bz2 2>/dev/null | sed -E 's|.*anode([0-9]+)\.tar\.bz2|\1|' | sort -n); fi
    [ ${#ANODES[@]} -eq 0 ] && { echo "no sim-frames-anode*.tar.bz2 in $INDIR" >&2; return 1; }
    echo "event $RUN/$EVT: anodes ${ANODES[*]} device=$DEVICE model=$MODEL half=$HALF -> $WORKDIR"
    cd "$WCFM_DIR"
    local ai CFG rc=0
    for ai in "${ANODES[@]}"; do
        CFG="$WORKDIR/.wct-fm-a${ai}.json"
        wcsonnet -A "input_prefix=$INDIR/sim-frames" -S "anode_indices=[$ai]" -A "output_dir=$WORKDIR" \
            -A "fm_model=$MODEL" -A "fm_model_sha=$MODEL_SHA" -A "fm_device=$DEVICE" -S "store_half=$HALF" \
            -o "$CFG" wct-fm-features.jsonnet
        [ -s "$CFG" ] || { echo "wcsonnet failed for anode $ai" >&2; return 1; }
    done
    {
        echo "toolkit=$(git -C $WCT_BASE/toolkit rev-parse --short HEAD)"
        echo "wcp=$(git -C $WCT_BASE/wcp-porting-img rev-parse --short HEAD)"
        echo "model=$MODEL"
        echo "model_sha256=$MODEL_SHA"
        echo "device=$DEVICE"
        echo "store_half=$HALF"
        echo "libpin=${WCFM_LIBPIN:-none}"
        echo "torch_threads=$OMP_NUM_THREADS"
        echo "fm_extracted=$(date -Is)"
    } > "$WORKDIR/fm-provenance.txt"
    for ai in "${ANODES[@]}"; do
        CFG="$WORKDIR/.wct-fm-a${ai}.json"
        local LOG="$WORKDIR/wct_fm_${RUN_PADDED}_${EVT}_a${ai}.log"
        rm -f "$LOG" "$WORKDIR/fm-features-anode${ai}.tar.gz"
        local GPUPID=""
        if [ "$DEVICE" != cpu ]; then
            local GPUCSV="$WORKDIR/.gpu_fm_a${ai}.csv"
            nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits > "$GPUCSV.base"
            ( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > "$GPUCSV" 2>/dev/null &
            GPUPID=$!
        fi
        python3 "$WCT_BASE/wcp-porting-img/abtest/timecmd.py" "$WORKDIR/time_fm_a${ai}.txt" \
            env $WC_PRELOAD wire-cell -l stderr -l "${LOG}:debug" -L debug -c "$CFG" || rc=$?
        if [ -n "$GPUPID" ]; then
            kill $GPUPID 2>/dev/null; wait $GPUPID 2>/dev/null || true
            python3 - "$GPUCSV" "$WORKDIR/gpu_fm_a${ai}.txt" <<'EOF'
import sys, collections
base = {l.split(',')[0].strip(): int(l.split(',')[1]) for l in open(sys.argv[1] + '.base') if ',' in l}
peak = collections.defaultdict(int)
for l in open(sys.argv[1]):
    if ',' not in l: continue
    i, m = l.split(','); peak[i.strip()] = max(peak[i.strip()], int(m))
with open(sys.argv[2], 'w') as f:
    for i in sorted(peak): f.write(f"gpu{i}_base_MiB={base.get(i, 0)}\ngpu{i}_peak_MiB={peak[i]}\ngpu{i}_delta_MiB={peak[i] - base.get(i, 0)}\n")
EOF
            rm -f "$GPUCSV" "$GPUCSV.base"
        fi
        rm -f "$CFG"
        echo "anode $ai fm rc=$rc $(cat "$WORKDIR/time_fm_a${ai}.txt" | tr '\n' ' ') $( [ -f "$WORKDIR/gpu_fm_a${ai}.txt" ] && grep delta "$WORKDIR/gpu_fm_a${ai}.txt" | tr '\n' ' ')"
        [ $rc -ne 0 ] && return $rc
    done
    echo "FM features done -> $WORKDIR"
    return $rc
}

if [ "$EVT" = "all" ]; then
    WCFM_MAX_JOBS=${WCFM_MAX_JOBS:-4}
    batch_init
    mapfile -t _events < <(ls -d "$WCFM_DIR/work/${RUN_PADDED}_"*/sim-frames-anode*.tar.bz2 2>/dev/null | sed -E "s|.*/${RUN_PADDED}_([0-9]+)/.*|\1|" | sort -n -u)
    [ ${#_events[@]} -eq 0 ] && { echo "no simulated events for run $RUN" >&2; exit 1; }
    echo "Found ${#_events[@]} event(s): ${_events[*]}  (parallel jobs: $BATCH_MAX)"
    for _e in "${_events[@]}"; do
        batch_wait_slot
        ( process_event "$RUN" "$_e" ) > "$WCFM_DIR/work/.batch_fm_${RUN_PADDED}_${_e}${SUFFIX}.log" 2>&1 &
        BATCH_PIDS[$!]=$_e
        echo "  [start] evt=$_e"
    done
    batch_drain
    batch_summary; exit $?
fi
process_event "$RUN" "$EVT"
