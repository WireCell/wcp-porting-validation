#!/bin/bash
# Per-event SBND pattern-recognition (PR) job — standalone, self-contained.  -h for help.
# Usage: ./run_pr_evt.sh [mc|data] [-N n] [-p names] <idx|all>
#        ./run_pr_evt.sh [mc|data] [-N n]            # list available events
#
# Loads the post-QL point-cloud tree written by run_ql_evt.sh -save-pctree
# (work/ql_evt<ID>/pctree-evt<ID>.tar.gz) and runs the PR-tail visitors on it
# (wct-pr-perevt.jsonnet), writing work/pr_evt<ID>/mabc-pr.zip plus a re-saved
# tree work/pr_evt<ID>/pctree-pr-evt<ID>.tar.gz for the round-trip gate.
# With the default empty pipeline this is the identity check: the zip's
# clustering layer must match the Q/L job's mabc-all-apa.zip clustering layer.
# See ../docs/sbnd-pattern-recognition.md.
#
# Prerequisite:  ./run_ql_evt.sh <mode> -save-pctree <idx>

set -e

SBND_DIR=$(cd "$(dirname "$0")" && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WCT_BASE}/wire-cell-data/sbnd/photodet:${WIRECELL_PATH}

. "$SBND_DIR/_runlib.sh"

JSONNET="$SBND_DIR/wct-pr-perevt.jsonnet"
# The SCN DL vertex runs python embedded in wire-cell.  Unlike the uBooNE
# qlport job (whose WireCellRoot plugin pulls libpython in with global
# symbol visibility via ROOT), this job loads no ROOT -- python C-extensions
# (_ctypes etc.) then fail with "undefined symbol: PyTuple_Type" unless
# libpython is preloaded globally.  Applied only when DL is requested.
PYLIB=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.11.so.1.0
# Same LAr TLAs as run_ql_evt.sh (anode/params objects identical to the Q/L job).
DL=6.2; DT=9.8; LIFETIME=6; DRIFTSPEED=1.563

usage() {
    cat <<EOF
Per-event SBND pattern recognition — runs on the post-QL pctree tarball.

Usage: $(basename "$0") [mc|data] [-N n] [-p names] <idx|all>
       $(basename "$0") [mc|data] [-N n]            # list available events

  mc|data   input set (default mc)
  idx       1-based event index (same numbering as run_ql_evt.sh);
            'all' runs every event with a saved pctree (parallel, cap nproc)
  -p        comma-separated PR pipeline visitor names (default '' = empty
            pipeline = round-trip identity gate). Names resolve in clus_pr's
            cm_by_name (cfg/pgrapher/experiment/sbnd/clus.jsonnet),
            e.g. -p switch_scope
  -stm      shorthand for the STM tagger chain:
            -p switch_scope,steiner,fiducialutils,tagger_check_stm
            (uses sbnd_track_fitting.json; grep TaggerCheckSTM in the log)
  -nu       shorthand for the neutrino-PR chain:
            -p switch_scope,steiner,fiducialutils,tagger_check_stm,tagger_check_neutrino
            with the per-mode beam window (grep TaggerCheckNeutrino in the log;
            Bee layers track_fit/shower_track/vertices + mc particle flow)
  -bw l,h   beam window [l,h) in us on cluster_t0 (matched flash time); overrides
            the per-mode default (mc ${BEAM_WINDOW_MC}, data ${BEAM_WINDOW_DATA} -- placeholders,
            see the md for calibration provenance)
  -dnn      -nu plus the SCN DL vertex with the uBooNE-trained weights
            (UNTUNED on SBND -- functional demo only; needs sparseconvnet
            importable in the job python, see the md)

Requires: run_ql_evt.sh <mode> -save-pctree <idx> first
          (work/ql_evt<ID>/pctree-evt<ID>.tar.gz).
Output:   work/pr_evt<ID>/mabc-pr.zip and pctree-pr-evt<ID>.tar.gz
EOF
}

MODE=mc
PIPELINE=""
NU=0
BEAM_WINDOW=""
DL_WEIGHTS=""
# Per-mode beam-window defaults in us on cluster_t0 (= matched flash time,
# trigger-offset-corrected -- NOT the raw opflash time convention of
# flash_t0_lan_reco2.py).  Calibrated from the 7 saved pctrees: MC BNB bundle
# +1.257 us (evt12); data in-time matched bundles +1.38/+1.69/+1.77 us
# (evts 686/1698/1258).  See the md for provenance.
BEAM_WINDOW_MC="0.5,2.0"
BEAM_WINDOW_DATA="0.5,2.5"
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -N) SBND_SAMPLE="$2"; shift 2 ;;
        -N*) SBND_SAMPLE="${1#-N}"; shift ;;
        mc|data) MODE="$1"; shift ;;
        -p) PIPELINE="$2"; shift 2 ;;
        -stm|--stm) PIPELINE="switch_scope,steiner,fiducialutils,tagger_check_stm"; shift ;;
        -nu|--nu) NU=1; PIPELINE="switch_scope,steiner,fiducialutils,tagger_check_stm,tagger_check_neutrino"; shift ;;
        -dnn|--dnn) NU=1; PIPELINE="switch_scope,steiner,fiducialutils,tagger_check_stm,tagger_check_neutrino"
                    DL_WEIGHTS="uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth"; shift ;;
        -bw) BEAM_WINDOW="$2"; shift 2 ;;
        -p*) PIPELINE="${1#-p}"; shift ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"

if [ "$NU" = 1 ] && [ -z "$BEAM_WINDOW" ]; then
    case "$MODE" in
        mc)   BEAM_WINDOW="$BEAM_WINDOW_MC" ;;
        data) BEAM_WINDOW="$BEAM_WINDOW_DATA" ;;
    esac
fi
BEAM_WINDOW_CODE="[${BEAM_WINDOW:-0,0}]"

case "$MODE" in
    mc)   REALITY=sim ;;
    data) REALITY=data ;;
esac

sbnd_check_sample "$MODE" || exit 1
[ -f "$JSONNET" ] || { echo "ERROR: missing jsonnet: $JSONNET" >&2; exit 1; }

load_events "$MODE" || exit 1
EVENT_IDS=("${SBND_EVENTS[@]}")

if [ $# -eq 0 ]; then
    echo "Sample: input-${SBND_SAMPLE}evt-$MODE   (${#EVENT_IDS[@]} events)"
    echo "Events for mode '$MODE' (idx -> EVT_ID):"
    for i in "${!EVENT_IDS[@]}"; do printf "  %2d -> %s\n" $((i + 1)) "${EVENT_IDS[$i]}"; done
    exit 0
fi

# jsonnet list literal from the comma-separated -p value.
PIPELINE_CODE="[]"
if [ -n "$PIPELINE" ]; then
    PIPELINE_CODE="[$(echo "$PIPELINE" | sed "s/[^,]\+/'&'/g")]"
fi

process_event() {
    local IDX=$1
    local EVT_ID="${EVENT_IDS[$((IDX - 1))]}"
    [ -n "$EVT_ID" ] || { echo "ERROR: invalid idx $IDX (1..${#EVENT_IDS[@]})" >&2; return 1; }

    local QLDIR="$SBND_DIR/work/ql_evt${EVT_ID}"
    local PCT="$QLDIR/pctree-evt${EVT_ID}.tar.gz"
    local PRDIR="$SBND_DIR/work/pr_evt${EVT_ID}"
    local LOG="$PRDIR/wct_pr_evt${EVT_ID}.log"

    if [ ! -s "$PCT" ]; then
        echo "[skip] evt $EVT_ID: missing $PCT — run ./run_ql_evt.sh $MODE -save-pctree $IDX first" >&2
        return 2
    fi

    rm -rf "$PRDIR"; mkdir -p "$PRDIR"

    echo "[evt $EVT_ID] PR (pipeline=$PIPELINE_CODE) $PCT -> $PRDIR/mabc-pr.zip"
    rm -f "$LOG"
    [ -n "$DL_WEIGHTS" ] && export LD_PRELOAD="$PYLIB"
    wire-cell \
        -l stderr -l "${LOG}:debug" -L debug \
        --tla-str  "input=$PCT" \
        --tla-code "anode_indices=[0,1]" \
        --tla-str  "output_dir=$PRDIR" \
        --tla-code "run=0" --tla-code "subrun=0" --tla-code "event=${EVT_ID}" \
        --tla-str  "reality=$REALITY" \
        --tla-code "DL=$DL" --tla-code "DT=$DT" \
        --tla-code "lifetime=$LIFETIME" --tla-code "driftSpeed=$DRIFTSPEED" \
        --tla-code "pipeline_names=$PIPELINE_CODE" \
        --tla-str  "trackfitting_config=$SBND_DIR/sbnd_track_fitting.json" \
        --tla-str  "save_tensors=$PRDIR/pctree-pr-evt${EVT_ID}.tar.gz" \
        --tla-str  "dl_weights=$DL_WEIGHTS" \
        --tla-code "beam_window_us=$BEAM_WINDOW_CODE" \
        -c "$JSONNET"
    echo "[evt $EVT_ID] done -> $PRDIR/mabc-pr.zip"
}

mkdir -p "$SBND_DIR/work"
IDX="$1"
if [ "$IDX" = "all" ]; then
    batch_init
    echo "Mode $MODE: ${#EVENT_IDS[@]} events. Parallel jobs: $BATCH_MAX"
    for i in $(seq 1 "${#EVENT_IDS[@]}"); do
        _evtid="${EVENT_IDS[$((i - 1))]}"
        _blog="$SBND_DIR/work/.batch_pr_evt${_evtid}.log"
        batch_wait_slot
        ( process_event "$i" ) > "$_blog" 2>&1 &
        BATCH_PIDS[$!]=$_evtid
        echo "  [start] idx=$i evt=$_evtid  log: $_blog"
    done
    batch_drain
    batch_summary
else
    process_event "$IDX"
fi
