#!/bin/bash
# Standalone NF+SP+DNN-ROI runner for ProtoDUNE-HD (no art/LArSoft).
#
# Mirrors run_nf_sp_evt.sh but loads wct-nf-sp-dnnroi.jsonnet so the
# multi-plane DNN-ROI subgraph is wired in after SP.  Uses the model at
# wire-cell-data/dnnroi/pdhd/CP43.ts.
#
# Usage:
#   ./run_nf_sp_dnnroi_evt.sh [-a anode] [-g elecGain] [-r reality]
#                             [-D cpu|gpu] [-M model.ts] [-m pp|mp]
#                             [-n 3|6] <run> <evt>
#
# Output: work/<RUN_PADDED>_<EVT>/
#   - protodunehd-sp-dnnroi-frames-anode{N}.tar.bz2 (post-DNN/L1SP frame —
#     the single canonical archive consumed by downstream scripts)

set -e

PDHD_DIR=$(cd "$(dirname "$0")" && pwd)

WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}

# Make libtorch + libWireCellPytorch.so findable at runtime.
export LD_LIBRARY_PATH=${WCT_BASE}/libtorch-shim/lib:${WCT_BASE}/local/lib:${LD_LIBRARY_PATH}

usage() {
    cat <<'EOF'
Usage: ./run_nf_sp_dnnroi_evt.sh [options] <run> <evt>

Options:
  -a <anode>     Anode index (0-3). Default: 0 (recommended for first run;
                 the model was trained on APA0 data).
  -g <elecGain>  FE amplifier gain in mV/fC. Default: 14.
  -r <reality>   'data' (default) or 'sim'.
  -D <device>    'cpu' (default) or 'gpu' for TorchService.
  -M <model>     TorchScript model path (resolved via WIRECELL_PATH).
                 Default: dnnroi/pdhd/CP43.ts
  -n <3|6>       Input channels the model expects. 3 (default) = original
                 CP43.ts; 6 = the 6-channel KD/QAT models. Selects the
                 input tag set and input_scale (6-ch models bake per-channel
                 normalization into the .ts, so they run with input_scale=1).
  -m <mode>      DNN-ROI wiring mode: 'pp' (per-plane sequential, default)
                 or 'mp' (stacked multi-plane, legacy).  Per-plane halves
                 peak activation memory by feeding U and V to the model
                 in two (1, 3, 800, 1500) calls instead of one stacked
                 (1, 3, 1600, 1500) call.
  -L <on|off>    Run L1SPFilterPD after DNN-ROI (default: on).  When on,
                 the DNN output is fed to L1SP as the signal channel and
                 raw ADC is preserved through the chain; the final frame
                 carries L1SP-corrected gauss%d / wiener%d alongside
                 raw%d.  When off, the post-DNN frame is written directly
                 (carries dnnsp%d* tags only).
  -X <basename>  If set, the C++ DNN node dumps {basename}_anode{N}_call{K}.pt
                 (containing model input + output + meta) for each call.
                 Use with scripts/verify_wirecell_dnn.py in DNN_ROI_SP.
  -h             Show this help.

Output (under work/<RUN_PADDED>_<EVT>/):
  protodunehd-sp-dnnroi-frames-anode{N}.tar.bz2  - post-DNN frame
  wct_nfspdnn_<RUN>_<EVT>_a<N>.log               - wire-cell log
  time_<RUN>_<EVT>_a<N>.txt                      - /usr/bin/time -v output
                                                   (CPU peak RSS, etc.)
  gpu_mem_<RUN>_<EVT>_a<N>.csv                   - nvidia-smi VRAM trace
                                                   (sampled at 100 ms)
EOF
}

ANODE="0"
ELEC_GAIN="14"
REALITY="data"
DEVICE="cpu"
MODEL="dnnroi/pdhd/CP43.ts"
MODE="pp"
NCHAN="3"
L1SP="on"
DEBUG_BASE=""

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -a) ANODE="$2"; shift 2 ;;
        -g) ELEC_GAIN="$2"; shift 2 ;;
        -r) REALITY="$2"; shift 2 ;;
        -D) DEVICE="$2"; shift 2 ;;
        -M) MODEL="$2"; shift 2 ;;
        -m) MODE="$2"; shift 2 ;;
        -n) NCHAN="$2"; shift 2 ;;
        -L) L1SP="$2"; shift 2 ;;
        -X) DEBUG_BASE="$2"; shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown option: $1" >&2; usage; exit 1 ;;
        *) break ;;
    esac
done

case "$MODE" in
    pp|mp) ;;
    *) echo "[err] -m must be 'pp' or 'mp' (got '$MODE')" >&2; exit 1 ;;
esac

case "$NCHAN" in
    3|6) ;;
    *) echo "[err] -n must be '3' or '6' (got '$NCHAN')" >&2; exit 1 ;;
esac

case "$L1SP" in
    on)  L1SP_TLA="true" ;;
    off) L1SP_TLA="false" ;;
    *) echo "[err] -L must be 'on' or 'off' (got '$L1SP')" >&2; exit 1 ;;
esac

if [ $# -lt 2 ]; then
    echo "missing <run> and/or <evt>" >&2
    usage; exit 1
fi
RUN=$1
EVT=$2

RUN_STRIPPED=$(echo "$RUN" | sed 's/^0*//')
[ -z "$RUN_STRIPPED" ] && RUN_STRIPPED=0
RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")

# Resolve event dir using the same heuristic as run_nf_sp_evt.sh.
find_evtdir() {
    local base="$PDHD_DIR/input_data"
    for rname in "run${RUN}" "run${RUN_PADDED}" "run${RUN_STRIPPED}"; do
        local rdir="$base/$rname"
        [ -d "$rdir" ] || continue
        for ename in "evt${EVT}" "evt_${EVT}"; do
            local cand="$rdir/$ename"
            if [ -d "$cand" ] && [ -n "$(ls -A "$cand" 2>/dev/null)" ]; then
                echo "$cand"; return 0
            fi
        done
        if ls "$rdir/protodunehd-orig-frames-anode"*.tar.bz2 >/dev/null 2>&1; then
            echo "$rdir"; return 0
        fi
    done
    return 1
}

EVTDIR=$(find_evtdir) || { echo "[err] no event dir for run=$RUN evt=$EVT" >&2; exit 2; }
echo "Event dir: $EVTDIR"

if ! ls "$EVTDIR/protodunehd-orig-frames-anode${ANODE}.tar.bz2" >/dev/null 2>&1; then
    echo "[err] missing $EVTDIR/protodunehd-orig-frames-anode${ANODE}.tar.bz2" >&2
    exit 2
fi

WORKDIR="$PDHD_DIR/work/${RUN_PADDED}_${EVT}"
mkdir -p "$WORKDIR"
LOG="$WORKDIR/wct_nfspdnn_${RUN_PADDED}_${EVT}_a${ANODE}.log"
TIME_LOG="$WORKDIR/time_${RUN_PADDED}_${EVT}_a${ANODE}.txt"
GPU_CSV="$WORKDIR/gpu_mem_${RUN_PADDED}_${EVT}_a${ANODE}.csv"
echo "Work dir:    $WORKDIR"
echo "elecGain:    ${ELEC_GAIN} mV/fC"
echo "reality:     ${REALITY}"
echo "device:      ${DEVICE}"
echo "model:       ${MODEL}"
echo "mode:        ${MODE}"
echo "nchan:       ${NCHAN}"
echo "L1SP:        ${L1SP}"
echo "Log:         $LOG"
echo "Time log:    $TIME_LOG"
echo "GPU CSV:     $GPU_CSV"

# Resolve debug-dump basename to an absolute path under WORKDIR if relative.
DBG_TLA=()
if [ -n "$DEBUG_BASE" ]; then
    case "$DEBUG_BASE" in
        /*) DBG_ABS="$DEBUG_BASE" ;;
        *)  DBG_ABS="$WORKDIR/$DEBUG_BASE" ;;
    esac
    mkdir -p "$(dirname "$DBG_ABS")"
    DBG_TLA=(--tla-str dnnroi_debugfile="$DBG_ABS")
    echo "Debug dump:  ${DBG_ABS}_anode${ANODE}_call*.pt"
fi

cd "$PDHD_DIR"
rm -f "$LOG" "$TIME_LOG" "$GPU_CSV"

# Pre-run baseline for VRAM, so the user can see "delta" not just absolute.
GPU_BASELINE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
                          -i 0 2>/dev/null | head -1 | tr -d ' ')
echo "GPU baseline VRAM (MiB, GPU 0): ${GPU_BASELINE:-?}" | tee -a "$GPU_CSV"
# Sample VRAM every 100 ms during the wire-cell run.
nvidia-smi --query-gpu=index,timestamp,memory.used --format=csv,noheader,nounits \
           -lms 100 -i 0 >> "$GPU_CSV" 2>/dev/null &
NVSMI_PID=$!

# Run wire-cell in the background so we can read its /proc/<pid>/status
# (GNU /usr/bin/time -v is not installed on this host).  VmHWM tracks the
# peak resident-set size for the lifetime of the process — better than
# sampling because it captures the high-water mark exactly.
RC=0
wire-cell \
    -l stderr \
    -l "${LOG}:debug" \
    -L debug \
    -V "elecGain=${ELEC_GAIN}" \
    --tla-str orig_prefix="${EVTDIR}/protodunehd-orig-frames" \
    --tla-str sp_prefix="${WORKDIR}/protodunehd-sp-dnnroi-frames" \
    --tla-str reality="${REALITY}" \
    --tla-code anode_indices="[${ANODE}]" \
    --tla-str dnnroi_model="${MODEL}" \
    --tla-str dnnroi_device="${DEVICE}" \
    --tla-str dnnroi_mode="${MODE}" \
    --tla-code dnnroi_nchan="${NCHAN}" \
    --tla-code use_l1sp_dnn="${L1SP_TLA}" \
    "${DBG_TLA[@]}" \
    -c wct-nf-sp-dnnroi.jsonnet &
WC_PID=$!

# Poll the wire-cell VmHWM (peak RSS) until the process exits.
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

# Stop the GPU sampler.
kill $NVSMI_PID 2>/dev/null
wait $NVSMI_PID 2>/dev/null

# Save the headline RSS into the TIME_LOG for posterity.
echo "VmHWM_kB=$WC_PEAK_KB" > "$TIME_LOG"

if [ "$WC_PEAK_KB" -gt 0 ]; then
    CPU_RSS_GIB=$(awk -v kb="$WC_PEAK_KB" 'BEGIN{printf "%.2f", kb/1024/1024}')
    echo "[mem] CPU peak RSS:  ${CPU_RSS_GIB} GiB (${WC_PEAK_KB} kB, from /proc/<pid>/status:VmHWM)"
fi
# Peak from the CSV (column 3 = memory.used).  Skip the baseline header line.
if [ -s "$GPU_CSV" ]; then
    GPU_PEAK=$(awk -F, 'NR>1 {gsub(" ",""); if($3+0>m)m=$3+0} END{print m+0}' "$GPU_CSV")
    if [ -n "$GPU_BASELINE" ] && [ -n "$GPU_PEAK" ]; then
        DELTA=$((GPU_PEAK - GPU_BASELINE))
        echo "[mem] GPU peak VRAM: ${GPU_PEAK} MiB  (delta over ${GPU_BASELINE} MiB baseline = ${DELTA} MiB)"
    elif [ -n "$GPU_PEAK" ]; then
        echo "[mem] GPU peak VRAM: ${GPU_PEAK} MiB"
    fi
fi

echo "DNN-ROI done -> $WORKDIR"
exit $RC
