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
  -D <device>    'cpu' (default) or 'gpu' for TorchService.  The QAT INT8
                 model is CPU-only; -P int8 forces -D cpu.
  -P <preset>    'fp32' (default) or 'int8'.  Selects the production
                 deployable for the current 6-ch PDHD campaign:
                   fp32 -> dnnroi/pdhd/pipe_distill_transformer_6ch.ts
                           (best FP32 KD; held-out Dice 0.9107)
                   int8 -> dnnroi/pdhd/pipe_qat_transformer_6ch_int8.ts
                           (best INT8 QAT; Dice 0.8900)
                 Both presets set -n 6 by default; -M and -n still override.
  -M <model>     TorchScript model path (resolved via WIRECELL_PATH).
                 Overrides -P.  Pass `-M dnnroi/pdhd/CP43.ts -n 3` to run
                 the legacy 3-ch model.
  -n <3|6>       Input channels the model expects. Overrides -P's default
                 (6).  3 = legacy CP43.ts.  6-channel models bake
                 per-channel normalization into the .ts, so they run with
                 input_scale=1.
  -m <mode>      DNN-ROI wiring mode: 'pp' (per-plane sequential, default)
                 or 'mp' (stacked multi-plane, legacy).  Per-plane halves
                 peak activation memory by feeding U and V to the model
                 in two (1, 3, 800, 1500) calls instead of one stacked
                 (1, 3, 1600, 1500) call.
  -N <heur|dnn>  L1SP tagger flavour when -L on (default: dnn).
                 'heur' = legacy 5-arm asymmetry heuristic.
                 'dnn'  = round-3 TorchScript model
                          (wire-cell-data/l1sp/pdhd/l1sp_dnn_pdhd_v1.ts);
                          polarity stays heuristic.  Threshold defaults
                          to 0.99 (p99.9 of the round-3 training
                          corpus); override via --tla-code
                          l1sp_pd_dnn_threshold=<x> if needed.  See
                          l1sp_dl_tagger/experiments/stage_a_pu_round3/
                          deploy_round3.md.
  -L <on|off>    Run L1SPFilterPD after DNN-ROI (default: on).  When on,
                 the DNN output is fed to L1SP as the signal channel and
                 raw ADC is preserved through the chain; the final frame
                 carries L1SP-corrected gauss%d / wiener%d alongside
                 raw%d.  When off, the post-DNN frame is written directly
                 (carries dnnsp%d* tags only).
  -X <basename>  If set, the C++ DNN node dumps {basename}_anode{N}_call{K}.pt
                 (containing model input + output + meta) for each call.
                 Use with scripts/verify_wirecell_dnn.py in DNN_ROI_SP.
  -T <thresh>    DNN sigmoid binarization threshold passed as
                 --tla-code dnnroi_mask_thresh=<val>.  Default: 0.2.
  -w <wf_dir>    Enable L1SP per-ROI waveform dump (requires -L on).  Writes
                 one NPZ per ROI under <wf_dir>/<RUN_PADDED>_<EVT>/apa<N>_*/.
                 Auto-enables dump-all-rois unless overridden by -A.
  -A <on|off>    Override dump-all-rois (default: on when -w is set, off
                 otherwise).  When on, every ROI is dumped, not just the
                 L1SP-triggered ones.
  -O <suffix>    Append <suffix> to the work-dir name so A/B comparison
                 runs can live side-by-side without overwriting each
                 other.  Default: '' (work/<RUN>_<EVT>/).  Example:
                 '-O _noL1SP' writes to work/<RUN>_<EVT>_noL1SP/.
  -Z <dir>       Write per-ROI L1SP-DNN debug NPZ under <dir>/.  Each
                 anode writes one dnn_<tag>_<call>_<ident>.npz file
                 with arrays: channel, plane, roi_start, roi_end,
                 polarity, fired, score, wave (N,2,nbin), scalars
                 (N,29), threshold[0], window_ticks[0].  Consumed by
                 code/inference/diagnose_l1sp_dnn.py.  Requires
                 -N dnn; ignored otherwise.  Relative paths resolve
                 under the work dir.
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
DEVICE_EXPLICIT=0
PRESET="fp32"
MODEL=""
MODEL_EXPLICIT=0
MODE="pp"
NCHAN=""
NCHAN_EXPLICIT=0
L1SP="on"
L1SP_MODE="dnn"
DEBUG_BASE=""
MASK_THRESH="0.2"
WF_DUMP_DIR=""
DUMP_ALL_ROIS=""
DUMP_ALL_EXPLICIT=0
WORK_SUFFIX=""
DNN_DEBUG_DIR=""

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -a) ANODE="$2"; shift 2 ;;
        -g) ELEC_GAIN="$2"; shift 2 ;;
        -r) REALITY="$2"; shift 2 ;;
        -D) DEVICE="$2"; DEVICE_EXPLICIT=1; shift 2 ;;
        -P) PRESET="$2"; shift 2 ;;
        -M) MODEL="$2"; MODEL_EXPLICIT=1; shift 2 ;;
        -m) MODE="$2"; shift 2 ;;
        -n) NCHAN="$2"; NCHAN_EXPLICIT=1; shift 2 ;;
        -L) L1SP="$2"; shift 2 ;;
        -N) L1SP_MODE="$2"; shift 2 ;;
        -X) DEBUG_BASE="$2"; shift 2 ;;
        -T) MASK_THRESH="$2"; shift 2 ;;
        -w) WF_DUMP_DIR="$2"; shift 2 ;;
        -A) DUMP_ALL_ROIS="$2"; DUMP_ALL_EXPLICIT=1; shift 2 ;;
        -O) WORK_SUFFIX="$2"; shift 2 ;;
        -Z) DNN_DEBUG_DIR="$2"; shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown option: $1" >&2; usage; exit 1 ;;
        *) break ;;
    esac
done

if [ "$MODEL_EXPLICIT" = "0" ]; then
    case "$PRESET" in
        fp32) MODEL="dnnroi/pdhd/pipe_distill_transformer_6ch.ts" ;;
        int8)
            MODEL="dnnroi/pdhd/pipe_qat_transformer_6ch_int8.ts"
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
if [ "$NCHAN_EXPLICIT" = "0" ]; then
    case "$MODEL" in
        *CP43.ts) NCHAN="3" ;;
        *)        NCHAN="6" ;;
    esac
fi

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

# L1SP mode TLA — empty string preserves the legacy heuristic path,
# 'dnn' switches L1SPFilterPD to the round-2 TorchScript model.
# Only meaningful when L1SP is on.
case "$L1SP_MODE" in
    heur) L1SP_PD_MODE_TLA="" ;;
    dnn)  L1SP_PD_MODE_TLA="dnn" ;;
    *) echo "[err] -N must be 'heur' or 'dnn' (got '$L1SP_MODE')" >&2; exit 1 ;;
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

WORKDIR="$PDHD_DIR/work/${RUN_PADDED}_${EVT}${WORK_SUFFIX}"
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
echo "L1SP mode:   ${L1SP_MODE}"
echo "mask_thresh: ${MASK_THRESH}"

# L1SP dump-mode wiring (mirrors run_nf_sp_evt.sh -w):
#   -w sets dump_mode='dump' + wf_dump_path; auto-enables dump_all_rois.
L1SP_DUMP_TLA=()
if [ -n "$WF_DUMP_DIR" ]; then
    if [ "$L1SP" != "on" ]; then
        echo "[err] -w requires -L on (L1SP must run to emit dumps)" >&2; exit 1
    fi
    case "$WF_DUMP_DIR" in
        /*) WF_ABS="$WF_DUMP_DIR" ;;
        *)  WF_ABS="$PDHD_DIR/$WF_DUMP_DIR" ;;
    esac
    WF_EVT_DIR="$WF_ABS/${RUN_PADDED}_${EVT}"
    mkdir -p "$WF_EVT_DIR"
    if [ "$DUMP_ALL_EXPLICIT" = "0" ]; then DUMP_ALL_ROIS="on"; fi
    case "$DUMP_ALL_ROIS" in
        on)  DUMP_ALL_TLA="true" ;;
        off) DUMP_ALL_TLA="false" ;;
        *)   echo "[err] -A must be 'on' or 'off'" >&2; exit 1 ;;
    esac
    # NB: do NOT set l1sp_pd_mode='dump' here.  Per-ROI waveform NPZ writes
    # happen in process mode as a side effect of LASSO write-back; 'dump'
    # mode skips the LASSO fit and writes only the calibration NPZ.  This
    # mirrors run_nf_sp_evt.sh -w (process + waveform dump).
    L1SP_DUMP_TLA=(--tla-str l1sp_pd_wf_dump_path="$WF_EVT_DIR" \
                   --tla-code l1sp_pd_dump_all_rois="$DUMP_ALL_TLA")
    echo "L1SP dump:   $WF_EVT_DIR (dump_all_rois=$DUMP_ALL_ROIS)"
elif [ "$DUMP_ALL_EXPLICIT" = "1" ]; then
    echo "[warn] -A given without -w; ignoring (nothing to dump)" >&2
fi

echo "Log:         $LOG"
echo "Time log:    $TIME_LOG"
echo "GPU CSV:     $GPU_CSV"

# Resolve L1SP-DNN per-ROI debug dir to an absolute path under WORKDIR if relative.
L1SP_DNN_DBG_TLA=()
if [ -n "$DNN_DEBUG_DIR" ]; then
    case "$DNN_DEBUG_DIR" in
        /*) DNN_DBG_ABS="$DNN_DEBUG_DIR" ;;
        *)  DNN_DBG_ABS="$WORKDIR/$DNN_DEBUG_DIR" ;;
    esac
    mkdir -p "$DNN_DBG_ABS"
    L1SP_DNN_DBG_TLA=(--tla-str l1sp_pd_dnn_debug_path="$DNN_DBG_ABS")
    echo "L1SP-DNN debug: $DNN_DBG_ABS"
fi

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
    --tla-str l1sp_pd_mode="${L1SP_PD_MODE_TLA}" \
    --tla-code dnnroi_mask_thresh="${MASK_THRESH}" \
    "${DBG_TLA[@]}" \
    "${L1SP_DUMP_TLA[@]}" \
    "${L1SP_DNN_DBG_TLA[@]}" \
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
