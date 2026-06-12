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
  -T <thresh>    DNN sigmoid binarization threshold passed as
                 --tla-code dnnroi_mask_thresh=<val>.  Default: 0.2.
  -L <on|off>    Wire L1SPFilterPD after DNN-ROI (default: on; pass
                 -L off to disable).  When on, the post-DNN
                 gauss/wiener feeds L1SPFilterPD via the
                 protodunevd/l1sp_after_dnnroi.jsonnet envelope.  Final
                 frame archive then carries L1SP-corrected gauss%d /
                 wiener%d alongside raw%d.
  -N <process|heur|dnn|hybrid>
                 L1SP mode when -L is on (default: process).
                   'process' = legacy 5-arm heuristic + LASSO (no model).
                   'heur'    = alias for 'process' (PDHD-style naming).
                   'dnn'     = call PDVD L1SP DNN on every ROI; LASSO if
                               score >= dnn_threshold (per-CRP).
                   'hybrid'  = loose 5-arm heuristic gates DNN inference.
                               DNN runs only on heur-positive ROIs.
                               Polarity from heur.  Track-veto is auto-
                               disabled in this mode (the DNN is the FP
                               suppressor).  Pair with --loose-heur for
                               best recovery and the per-CRP DNN thresh.
                               See experiments/stage_a_pu_round2_pdvd/
                               deploy_round2.md.
  --loose-heur   Loosen L1SP heuristic pre-filters for the DNN chain.
                 Defaults are PDVD-optimized per Phase A2 sweep
                 (experiments/stage_a_pu_round2_pdvd/deploy_round2.md §"Phase A2"):
                   l1_gmax_min        1500 -> 300
                   l1_min_length        30 -> 5      (PDHD uses 10; PDVD
                                                      DNN-ROIs are shorter)
                   l1_energy_frac_thr 0.66 -> 0.20
                 Recommended for -N hybrid.  Coverage of trad-positives
                 across all 22 PDVD events: 76.0 % (vs 72.8 % at PDHD-loose).
                 Override individual values with --gmax-min/--min-length/
                 --energy-frac.
  --l1sp-thresh-bottom <val>
                 DNN sigmoid threshold for bottom CRP (apa<4).  Default:
                 jsonnet 0.94 (single fallback); deploy_round2.md picks
                 0.16 from PR sweep at precision >= 0.70.
  --l1sp-thresh-top <val>
                 DNN sigmoid threshold for top CRP (apa>=4).  Default:
                 jsonnet 0.94; deploy_round2.md picks 0.46.
  --l1sp-thresh <val>
                 Single threshold for both CRPs (overrides both above).
  -Z <dir>       Write per-ROI L1SP-DNN debug NPZ under <dir>/ (resolved
                 relative to the workdir if not absolute).  Required for
                 LASSO-fire verification (see deploy_round2.md §"Phase E").
  -O <suffix>    Append <suffix> to the work-dir name so A/B comparison
                 runs don't clobber each other.  E.g. -O _hybrid writes
                 to work/<RUN>_<EVT>_hybrid/.
  -c <calib_root>
                 Switch L1SP to dump (tagger-only) mode and write per-ROI
                 calib NPZs to <calib_root>/<RUN_PADDED>_<EVT>/apa<N>_*.npz.
                 No LASSO write-back, no DNN dispatch — just the heuristic
                 5-arm decision recorded per ROI for offline analysis.
                 Pair with --gmax-min/--min-length/--energy-frac to dump
                 at non-default pre-filters (useful for the loose-heur
                 optimization sweep described in
                 experiments/stage_a_pu_round2_pdvd/deploy_round2.md
                 §"Phase A2").
  --gmax-min <val>
  --min-length <val>
  --energy-frac <val>
                 Per-parameter pre-filter overrides (override --loose-heur
                 if both given).  Use these to dump at arbitrary triples
                 for the optimization sweep.
  -w <wf_dir>   Enable L1SP per-ROI waveform dump (forces -L on).  Writes
                 one NPZ per ROI under <wf_dir>/<RUN_PADDED>_<EVT>/apa<N>_*/.
                 Auto-enables dump-all-rois unless overridden by -A.
  -A <on|off>   Override dump-all-rois (default: on when -w is set).
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
MASK_THRESH="0.2"
L1SP=""
L1SP_EXPLICIT=0
L1SP_MODE="process"
WF_DUMP_DIR=""
DUMP_ALL_ROIS=""
DUMP_ALL_EXPLICIT=0
DNN_DEBUG_DIR=""
WORK_SUFFIX=""
LOOSE_HEUR=0
CALIB_ROOT=""
GMAX_MIN_OVERRIDE=""
MIN_LENGTH_OVERRIDE=""
ENERGY_FRAC_OVERRIDE=""
L1SP_THRESH_BOTTOM=""
L1SP_THRESH_TOP=""
L1SP_THRESH_SINGLE=""
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
        -T) MASK_THRESH="$2"; shift 2 ;;
        -T*) MASK_THRESH="${1#-T}"; shift ;;
        -L) L1SP="$2"; L1SP_EXPLICIT=1; shift 2 ;;
        -L*) L1SP="${1#-L}"; L1SP_EXPLICIT=1; shift ;;
        -N) L1SP_MODE="$2"; shift 2 ;;
        -N*) L1SP_MODE="${1#-N}"; shift ;;
        -Z) DNN_DEBUG_DIR="$2"; shift 2 ;;
        -Z*) DNN_DEBUG_DIR="${1#-Z}"; shift ;;
        -O) WORK_SUFFIX="$2"; shift 2 ;;
        -O*) WORK_SUFFIX="${1#-O}"; shift ;;
        --loose-heur) LOOSE_HEUR=1; shift ;;
        -c) CALIB_ROOT="$2"; shift 2 ;;
        -c*) CALIB_ROOT="${1#-c}"; shift ;;
        --gmax-min) GMAX_MIN_OVERRIDE="$2"; shift 2 ;;
        --min-length) MIN_LENGTH_OVERRIDE="$2"; shift 2 ;;
        --energy-frac) ENERGY_FRAC_OVERRIDE="$2"; shift 2 ;;
        --l1sp-thresh-bottom) L1SP_THRESH_BOTTOM="$2"; shift 2 ;;
        --l1sp-thresh-top) L1SP_THRESH_TOP="$2"; shift 2 ;;
        --l1sp-thresh) L1SP_THRESH_SINGLE="$2"; shift 2 ;;
        -w) WF_DUMP_DIR="$2"; shift 2 ;;
        -w*) WF_DUMP_DIR="${1#-w}"; shift ;;
        -A) DUMP_ALL_ROIS="$2"; DUMP_ALL_EXPLICIT=1; shift 2 ;;
        -A*) DUMP_ALL_ROIS="${1#-A}"; DUMP_ALL_EXPLICIT=1; shift ;;
        -*) echo "unknown option: $1" >&2; usage; exit 1 ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"

# Normalize -N alias.
case "$L1SP_MODE" in
    heur) L1SP_MODE="process" ;;
    process|dnn|hybrid) ;;
    *) echo "[err] -N must be one of process|heur|dnn|hybrid (got '$L1SP_MODE')" >&2; exit 1 ;;
esac

# L1SP defaults ON (matches run_nf_sp_evt.sh, which always wires L1SP).
# Mode comes from -N (default 'process'); pass -L off to disable.
# -c (calib dump) also auto-enables L1SP and overrides the mode to 'dump'.
if [ "$L1SP_EXPLICIT" = "0" ]; then
    L1SP="on"
fi
if [ -n "$CALIB_ROOT" ]; then
    L1SP="on"
fi

case "$L1SP" in
    on|off) ;;
    *) echo "[err] -L must be 'on' or 'off' (got '$L1SP')" >&2; exit 1 ;;
esac
# -w implies -L on (the envelope must be wired for L1SP to emit dumps).
if [ -n "$WF_DUMP_DIR" ]; then L1SP="on"; fi
case "$L1SP" in
    on)  USE_L1SP_DNN_TLA="true" ;;
    off) USE_L1SP_DNN_TLA="false" ;;
esac

# DNN-mode requires the L1SP envelope to be wired.
if [ "$L1SP" = "off" ] && [ "$L1SP_MODE" != "process" ]; then
    echo "[err] -N $L1SP_MODE requires -L on (the L1SP envelope must be wired)" >&2
    exit 1
fi

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

WORKDIR="$PDVD_DIR/work/${RUN_PADDED}_${EVT}${WORK_SUFFIX}"
mkdir -p "$WORKDIR"
LOG="$WORKDIR/wct_nfspdnn_${RUN_PADDED}_${EVT}${TAG_SUFFIX}.log"
TIME_LOG="$WORKDIR/time_${RUN_PADDED}_${EVT}${TAG_SUFFIX}.txt"
GPU_CSV="$WORKDIR/gpu_mem_${RUN_PADDED}_${EVT}${TAG_SUFFIX}.csv"
echo "Work dir:    $WORKDIR"
echo "reality:     ${REALITY}"
echo "device:      ${DEVICE}"
echo "model:       ${MODEL}"
echo "anodes:      ${ANODE_CODE}"
echo "mask_thresh: ${MASK_THRESH}"
echo "L1SP:        ${L1SP} (mode=${L1SP_MODE})"
echo "Log:         $LOG"

# L1SP per-ROI waveform dump wiring.  -w sets the dump dir; the envelope
# emits per-ROI NPZ in 'process' mode (LASSO writeback side effect — same
# convention as PDHD's run_nf_sp_dnnroi_evt.sh -w).  -w forces mode=process.
L1SP_DUMP_TLA=()
if [ -n "$WF_DUMP_DIR" ]; then
    case "$WF_DUMP_DIR" in
        /*) WF_ABS="$WF_DUMP_DIR" ;;
        *)  WF_ABS="$PDVD_DIR/$WF_DUMP_DIR" ;;
    esac
    WF_EVT_DIR="$WF_ABS/${RUN_PADDED}_${EVT}"
    mkdir -p "$WF_EVT_DIR"
    if [ "$DUMP_ALL_EXPLICIT" = "0" ]; then DUMP_ALL_ROIS="on"; fi
    case "$DUMP_ALL_ROIS" in
        on)  DUMP_ALL_TLA="true" ;;
        off) DUMP_ALL_TLA="false" ;;
        *)   echo "[err] -A must be 'on' or 'off' (got '$DUMP_ALL_ROIS')" >&2; exit 1 ;;
    esac
    # -w forces 'process' mode (LASSO writeback path); silently override any -N.
    L1SP_MODE="process"
    L1SP_DUMP_TLA=(--tla-str l1sp_pd_wf_dump_path="$WF_EVT_DIR" \
                   --tla-code l1sp_pd_dump_all_rois="$DUMP_ALL_TLA")
    echo "L1SP dump:   $WF_EVT_DIR (dump_all_rois=$DUMP_ALL_ROIS) [forces mode=process]"
elif [ "$DUMP_ALL_EXPLICIT" = "1" ]; then
    echo "[warn] -A given without -w; ignoring (nothing to dump)" >&2
fi

# -c calib_root: switch to mode=dump, write per-ROI scalar NPZ.  Pair with
# --gmax-min/--min-length/--energy-frac to dump at custom pre-filter values
# (used by the Phase A2 optimization sweep).  Must happen BEFORE the
# L1SP_MODE_TLA assembly below so the mode-flag flip takes effect.
L1SP_CALIB_TLA=()
if [ -n "$CALIB_ROOT" ]; then
    case "$CALIB_ROOT" in
        /*) CALIB_ABS="$CALIB_ROOT" ;;
        *)  CALIB_ABS="$PDVD_DIR/$CALIB_ROOT" ;;
    esac
    CALIB_EVT_DIR="$CALIB_ABS/${RUN_PADDED}_${EVT}"
    mkdir -p "$CALIB_EVT_DIR"
    L1SP_MODE="dump"
    L1SP_CALIB_TLA=(--tla-str l1sp_pd_dump_path="$CALIB_EVT_DIR")
    echo "L1SP calib:  $CALIB_EVT_DIR [forces mode=dump]"
fi

# L1SP mode TLA — always passed when L1SP is on (empty when off so the
# envelope's default 'process' applies via use_l1sp_dnn=false).  Built
# AFTER -c so calib mode wins.
L1SP_MODE_TLA=()
if [ "$L1SP" = "on" ]; then
    L1SP_MODE_TLA=(--tla-str l1sp_pd_mode="$L1SP_MODE")
fi

# Loose-heur preset (mirrors PDHD round-4 loose-heur values; see
# experiments/stage_a_pu_round2_pdvd/deploy_round2.md §"Phase A" for the
# PDVD recovery-rate validation that justifies these on PDVD too).
# Per-parameter overrides (--gmax-min/--min-length/--energy-frac) take
# precedence over --loose-heur.
LOOSE_HEUR_TLA=()
if [ "$LOOSE_HEUR" = "1" ]; then
    # PDVD-optimized preset (Phase A2: min_length 10 -> 5 vs PDHD-loose
    # because PDVD DNN-ROIs are shorter, so the length pre-filter is the
    # dominant rejection knob).
    GMAX_MIN_RESOLVED="${GMAX_MIN_OVERRIDE:-300.0}"
    MIN_LENGTH_RESOLVED="${MIN_LENGTH_OVERRIDE:-5}"
    ENERGY_FRAC_RESOLVED="${ENERGY_FRAC_OVERRIDE:-0.20}"
else
    GMAX_MIN_RESOLVED="$GMAX_MIN_OVERRIDE"
    MIN_LENGTH_RESOLVED="$MIN_LENGTH_OVERRIDE"
    ENERGY_FRAC_RESOLVED="$ENERGY_FRAC_OVERRIDE"
fi
if [ -n "$GMAX_MIN_RESOLVED" ]; then
    LOOSE_HEUR_TLA+=(--tla-code l1sp_pd_gmax_min="$GMAX_MIN_RESOLVED")
fi
if [ -n "$MIN_LENGTH_RESOLVED" ]; then
    LOOSE_HEUR_TLA+=(--tla-code l1sp_pd_min_length="$MIN_LENGTH_RESOLVED")
fi
if [ -n "$ENERGY_FRAC_RESOLVED" ]; then
    LOOSE_HEUR_TLA+=(--tla-code l1sp_pd_energy_frac_thr="$ENERGY_FRAC_RESOLVED")
fi
if [ ${#LOOSE_HEUR_TLA[@]} -gt 0 ]; then
    echo "Loose heur:  gmax_min=${GMAX_MIN_RESOLVED:-default} min_length=${MIN_LENGTH_RESOLVED:-default} energy_frac=${ENERGY_FRAC_RESOLVED:-default}"
fi

# Per-CRP DNN threshold overrides.  --l1sp-thresh sets both legs.
L1SP_THRESH_TLA=()
if [ -n "$L1SP_THRESH_SINGLE" ]; then
    L1SP_THRESH_TLA+=(--tla-code l1sp_pd_dnn_threshold="$L1SP_THRESH_SINGLE")
fi
if [ -n "$L1SP_THRESH_BOTTOM" ]; then
    L1SP_THRESH_TLA+=(--tla-code l1sp_pd_dnn_threshold_bottom="$L1SP_THRESH_BOTTOM")
    echo "L1SP thresh: bottom=$L1SP_THRESH_BOTTOM"
fi
if [ -n "$L1SP_THRESH_TOP" ]; then
    L1SP_THRESH_TLA+=(--tla-code l1sp_pd_dnn_threshold_top="$L1SP_THRESH_TOP")
    echo "L1SP thresh: top=$L1SP_THRESH_TOP"
fi

# L1SP-DNN per-ROI debug dump (one NPZ per call with channel, score,
# fired, polarity, waveform, scalars). Required for LASSO-fire verification.
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
    --tla-code dnnroi_mask_thresh="${MASK_THRESH}" \
    --tla-code use_l1sp_dnn="${USE_L1SP_DNN_TLA}" \
    "${DBG_TLA[@]}" \
    "${L1SP_DUMP_TLA[@]}" \
    "${L1SP_MODE_TLA[@]}" \
    "${LOOSE_HEUR_TLA[@]}" \
    "${L1SP_THRESH_TLA[@]}" \
    "${L1SP_DNN_DBG_TLA[@]}" \
    "${L1SP_CALIB_TLA[@]}" \
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
