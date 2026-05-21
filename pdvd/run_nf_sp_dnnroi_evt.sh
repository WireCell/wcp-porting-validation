#!/bin/bash
# Standalone NF+SP+DNN-ROI runner for ProtoDUNE-VD (no art/LArSoft).
#
# Mirrors run_nf_sp_evt.sh but loads wct-nf-sp-dnnroi.jsonnet so the
# 4-channel multi-plane DNN-ROI subgraph is wired in after SP.  Uses a
# TorchScript model under wire-cell-data/dnnroi/pdvd/.
#
# Usage:
#   ./run_nf_sp_dnnroi_evt.sh [-a anode] [-r reality] [-D cpu|gpu]
#                             [-M model.ts] [-X debugbase] <run> <evt>
#   ./run_nf_sp_dnnroi_evt.sh                 # list available runs
#
# Output (under work/<RUN_PADDED>_<EVT>/):
#   protodune-sp-dnnroi-frames-anode{N}.tar.bz2  - post-DNN frame
#   wct_nfspdnn_<RUN>_<EVT><tag>.log             - wire-cell log
#   time_<RUN>_<EVT><tag>.txt                    - peak RSS (VmHWM)
#   gpu_mem_<RUN>_<EVT><tag>.csv                 - nvidia-smi VRAM trace

set -e

PDVD_DIR=$(cd "$(dirname "$0")" && pwd)

WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}
# Make libtorch + libWireCellPytorch.so findable at runtime.
export LD_LIBRARY_PATH=${WCT_BASE}/libtorch-shim/lib:${WCT_BASE}/local/lib:${LD_LIBRARY_PATH}

. "$PDVD_DIR/_runlib.sh"

usage() {
    cat <<'EOF'
Usage: ./run_nf_sp_dnnroi_evt.sh [options] <run> <evt>
       ./run_nf_sp_dnnroi_evt.sh          # list available runs

Options:
  -a <anode>     Anode index 0-7 to process.  Default: all 8 (anode_indices
                 [0..7] in one wire-cell process).  Bottom CRP: 0-3 (crp1-4,
                 in the training corpus); top CRP: 4-7 (out-of-domain).
  -r <reality>   'data' (default): inserts the 512->500 ns Resampler on the
                 bottom anodes (n<4) before NF.  'sim': skips it.
  -D <device>    'cpu' (default) or 'gpu' for TorchService.  The QAT INT8
                 model is CPU-only; -P int8 forces -D cpu.
  -P <preset>    'fp32' (default) or 'int8'.  Selects the production
                 deployable for the current 6-ch PDVD campaign:
                   fp32 -> dnnroi/pdvd/pipe_distill_nestedunet_6ch.ts
                           (best FP32 KD; held-out Dice 0.7816)
                   int8 -> dnnroi/pdvd/pipe_qat_nestedunet_6ch_ep0_int8.ts
                           (best INT8 QAT; Dice 0.7797, -0.27% vs FP32)
                 -M overrides the preset's model path.
  -M <model>     TorchScript model path (resolved via WIRECELL_PATH).
                 Overrides -P.  Default: see -P fp32 above.
  -X <basename>  If set, the C++ DNN node dumps {basename}_anode{N}_call{K}.pt
                 (model input + output + meta) for each call.  Use with
                 DNN_ROI_SP/scripts/verify_wirecell_dnn.py.
  -h             Show this help.

Input:  input_data/<run_dir>/<evt_dir>/protodune-orig-frames-anode{0..7}.tar.bz2
Output: work/<RUN_PADDED>_<EVT>/protodune-sp-dnnroi-frames-anode{N}.tar.bz2
EOF
}

ANODE=""
REALITY="data"
DEVICE_EXPLICIT=0
DEVICE="cpu"
PRESET="fp32"
MODEL=""
MODEL_EXPLICIT=0
DEBUG_BASE=""
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -a) ANODE="$2"; shift 2 ;;
        -a*) ANODE="${1#-a}"; shift ;;
        -r) REALITY="$2"; shift 2 ;;
        -r*) REALITY="${1#-r}"; shift ;;
        -D) DEVICE="$2"; DEVICE_EXPLICIT=1; shift 2 ;;
        -D*) DEVICE="${1#-D}"; DEVICE_EXPLICIT=1; shift ;;
        -P) PRESET="$2"; shift 2 ;;
        -P*) PRESET="${1#-P}"; shift ;;
        -M) MODEL="$2"; MODEL_EXPLICIT=1; shift 2 ;;
        -M*) MODEL="${1#-M}"; MODEL_EXPLICIT=1; shift ;;
        -X) DEBUG_BASE="$2"; shift 2 ;;
        -X*) DEBUG_BASE="${1#-X}"; shift ;;
        -*) echo "unknown option: $1" >&2; usage; exit 1 ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"

if [ "$MODEL_EXPLICIT" = "0" ]; then
    case "$PRESET" in
        fp32) MODEL="dnnroi/pdvd/pipe_distill_nestedunet_6ch.ts" ;;
        int8)
            MODEL="dnnroi/pdvd/pipe_qat_nestedunet_6ch_ep0_int8.ts"
            if [ "$DEVICE_EXPLICIT" = "0" ]; then
                DEVICE="cpu"
            elif [ "$DEVICE" != "cpu" ]; then
                echo "[err] -P int8 requires -D cpu (INT8 graph is CPU-only)" >&2
                exit 1
            fi
            ;;
        *) echo "[err] -P must be 'fp32' or 'int8' (got '$PRESET')" >&2; exit 1 ;;
    esac
fi

if [ $# -eq 0 ]; then
    list_runs; exit 0
fi
if [ $# -lt 2 ]; then
    echo "missing <run> and/or <evt>" >&2
    usage; exit 1
fi
RUN=$1
EVT=$2

RUN_STRIPPED=$(echo "$RUN" | sed 's/^0*//')
[ -z "$RUN_STRIPPED" ] && RUN_STRIPPED=0
RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")

find_evtdir() {
    local base="$PDVD_DIR/input_data"
    for rname in "run${RUN}" "run${RUN_PADDED}" "run${RUN_STRIPPED}"; do
        local rdir="$base/$rname"
        [ -d "$rdir" ] || continue
        for ename in "evt${EVT}" "evt_${EVT}"; do
            local cand="$rdir/$ename"
            if [ -d "$cand" ] && [ -n "$(ls -A "$cand" 2>/dev/null)" ]; then
                echo "$cand"; return 0
            fi
        done
        if ls "$rdir/protodune-orig-frames-anode"*.tar.bz2 >/dev/null 2>&1; then
            echo "$rdir"; return 0
        fi
    done
    return 1
}

EVTDIR=$(find_evtdir) || { echo "[err] no event dir for run=$RUN evt=$EVT" >&2; exit 2; }
echo "Event dir: $EVTDIR"

if [ -n "$ANODE" ]; then
    ANODE_CODE="[$ANODE]"
    TAG_SUFFIX="_a${ANODE}"
    if ! ls "$EVTDIR/protodune-orig-frames-anode${ANODE}.tar.bz2" >/dev/null 2>&1; then
        echo "[err] missing $EVTDIR/protodune-orig-frames-anode${ANODE}.tar.bz2" >&2
        exit 2
    fi
else
    ANODE_CODE="[0,1,2,3,4,5,6,7]"
    TAG_SUFFIX=""
fi

WORKDIR="$PDVD_DIR/work/${RUN_PADDED}_${EVT}"
mkdir -p "$WORKDIR"
LOG="$WORKDIR/wct_nfspdnn_${RUN_PADDED}_${EVT}${TAG_SUFFIX}.log"
TIME_LOG="$WORKDIR/time_${RUN_PADDED}_${EVT}${TAG_SUFFIX}.txt"
GPU_CSV="$WORKDIR/gpu_mem_${RUN_PADDED}_${EVT}${TAG_SUFFIX}.csv"
echo "Work dir:    $WORKDIR"
echo "reality:     ${REALITY}"
echo "device:      ${DEVICE}"
echo "model:       ${MODEL}"
echo "anodes:      ${ANODE_CODE}"
echo "Log:         $LOG"

# Resolve debug-dump basename to an absolute path under WORKDIR if relative.
DBG_TLA=()
if [ -n "$DEBUG_BASE" ]; then
    case "$DEBUG_BASE" in
        /*) DBG_ABS="$DEBUG_BASE" ;;
        *)  DBG_ABS="$WORKDIR/$DEBUG_BASE" ;;
    esac
    mkdir -p "$(dirname "$DBG_ABS")"
    DBG_TLA=(--tla-str dnnroi_debugfile="$DBG_ABS")
    echo "Debug dump:  ${DBG_ABS}_anode{N}_call*.pt"
fi

cd "$PDVD_DIR"
rm -f "$LOG" "$TIME_LOG" "$GPU_CSV"

# Pre-run baseline for VRAM, so the user sees the "delta" not just absolute.
GPU_BASELINE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
                          -i 0 2>/dev/null | head -1 | tr -d ' ')
echo "GPU baseline VRAM (MiB, GPU 0): ${GPU_BASELINE:-?}" | tee -a "$GPU_CSV"
nvidia-smi --query-gpu=index,timestamp,memory.used --format=csv,noheader,nounits \
           -lms 100 -i 0 >> "$GPU_CSV" 2>/dev/null &
NVSMI_PID=$!

# Run wire-cell in the background so we can poll /proc/<pid>/status:VmHWM
# (peak resident-set size) for the lifetime of the process.
RC=0
wire-cell \
    -l stderr \
    -l "${LOG}:debug" \
    -L debug \
    --tla-str orig_prefix="${EVTDIR}/protodune-orig-frames" \
    --tla-str sp_prefix="${WORKDIR}/protodune-sp-dnnroi-frames" \
    --tla-str reality="${REALITY}" \
    --tla-code anode_indices="${ANODE_CODE}" \
    --tla-str dnnroi_model="${MODEL}" \
    --tla-str dnnroi_device="${DEVICE}" \
    "${DBG_TLA[@]}" \
    -c wct-nf-sp-dnnroi.jsonnet &
WC_PID=$!

WC_PEAK_KB=0
while kill -0 $WC_PID 2>/dev/null; do
    if [ -r /proc/$WC_PID/status ]; then
        HWM=$(awk '/^VmHWM:/ {print $2}' /proc/$WC_PID/status 2>/dev/null)
        if [ -n "$HWM" ] && [ "$HWM" -gt "$WC_PEAK_KB" ]; then
            WC_PEAK_KB=$HWM
        fi
    fi
    sleep 0.2
done
wait $WC_PID || RC=$?

kill $NVSMI_PID 2>/dev/null
wait $NVSMI_PID 2>/dev/null

echo "VmHWM_kB=$WC_PEAK_KB" > "$TIME_LOG"
if [ "$WC_PEAK_KB" -gt 0 ]; then
    CPU_RSS_GIB=$(awk -v kb="$WC_PEAK_KB" 'BEGIN{printf "%.2f", kb/1024/1024}')
    echo "[mem] CPU peak RSS:  ${CPU_RSS_GIB} GiB (${WC_PEAK_KB} kB, VmHWM)"
fi
if [ -s "$GPU_CSV" ]; then
    GPU_PEAK=$(awk -F, 'NR>1 {gsub(" ",""); if($3+0>m)m=$3+0} END{print m+0}' "$GPU_CSV")
    if [ -n "$GPU_BASELINE" ] && [ -n "$GPU_PEAK" ]; then
        DELTA=$((GPU_PEAK - GPU_BASELINE))
        echo "[mem] GPU peak VRAM: ${GPU_PEAK} MiB  (delta over ${GPU_BASELINE} MiB baseline = ${DELTA} MiB)"
    fi
fi

echo "DNN-ROI done -> $WORKDIR"
exit $RC
