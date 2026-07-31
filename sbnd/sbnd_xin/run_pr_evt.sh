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
DL=4.0; DT=8.8; LIFETIME=35; DRIFTSPEED=1.563  # DL/DT = SBND diffusion (cm^2/s), sbndcode wcsimsp_sbnd.fcl (docs/66);
                                                      # LIFETIME = SBND simparams (35 ms).  Inert in the
                                                      # reco chain -- see docs/64 sec 4.

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
  -stm-fit  persist the per-pass STM track fits (doc 40): cluster PCs
            stm_fit/stm_pass/stm_eval, a Bee 'stm_fit' layer in mabc-pr.zip,
            and tracking-stm.root (appends stm_magnify to the pipeline).
            DEFAULT OFF = byte-identical legacy outputs.  Env: SBND_STM_FIT=1.
  -tgm      shorthand for the cosmic-tagger chain (TGM then STM):
            -p switch_scope,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm
            with the per-mode beam window (in-window bundles are never TGM-tagged;
            grep TaggerCheckTGM in the log)
  -nu       shorthand for the full PR chain:
            -p switch_scope,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm,tagger_check_neutrino
            with the per-mode beam window (grep TaggerCheckNeutrino in the log;
            Bee layers track_fit/shower_track/vertices + mc particle flow)
  -bw l,h   beam window [l,h) in us on cluster_t0 (matched flash time); overrides
            the per-mode default (mc ${BEAM_WINDOW_MC}, data ${BEAM_WINDOW_DATA} = the
            experiment window, same as SBND Q/L beam_pref).  Since doc 56 this
            window also GATES which bundles the taggers evaluate at all
            (beam_window_only, default on) -- not just TGM's beam protection.
  -dnn      -nu plus the SCN DL vertex (now the DEFAULT for any pipeline that
            includes tagger_check_neutrino, so this is just the -nu shorthand
            with DL spelled out).  Needs sparseconvnet importable in the job
            python; the libpython preload is handled here.  Weights are still
            uBooNE-TRAINED (SBND retraining = docs/pr/2 gap G3).
  -no-dnn   geometric vertex instead of the DL one (SBND_DL_VTX=0).  This is the
            arm every identity gate must use -- the DL vertex is never a gate arm.

Requires: run_ql_evt.sh <mode> -save-pctree <idx> first
          (work/ql_evt<ID>/pctree-evt<ID>.tar.gz).
Output:   work/pr_evt<ID>/mabc-pr.zip and pctree-pr-evt<ID>.tar.gz
EOF
}

MODE=mc
PIPELINE=""
# Persist per-pass STM track fits + tracking-stm.root (doc 40).
# DEFAULT OFF: opt in with -stm-fit / SBND_STM_FIT=1.
STM_FIT="${SBND_STM_FIT:-0}"
NU=0
BEAM_WINDOW=""
# SCN (DL) neutrino vertex.  DEFAULT ON since 2026-07-30 (docs/pr/4): the
# geometric vertex put evt 18253/1/172230's vertex at the far end of a proton
# track; the DL vertex moved it 9.7 cm onto the true interaction point.  The
# weights stay uBooNE-TRAINED (docs/pr/2 gap G3 open).  Only meaningful with
# tagger_check_neutrino in the pipeline -- inert otherwise.
# Turn off with -no-dnn (or SBND_DL_VTX=0) for the geometric-vertex arm, which
# is also what every identity gate must use (CLAUDE.md M4).
DL_WEIGHTS_DEFAULT="uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth"
DL_WEIGHTS="$([ "${SBND_DL_VTX:-1}" = 0 ] && echo "" || echo "$DL_WEIGHTS_DEFAULT")"
# Per-mode beam-window defaults in us on cluster_t0 (= matched flash time,
# trigger-offset-corrected -- NOT the raw opflash time convention of
# flash_t0_lan_reco2.py).  Originally guessed from the 7 saved pctrees
# (MC BNB bundle +1.257 us evt12; data in-time matched bundles +1.38/+1.69/+1.77
# us, evts 686/1698/1258) as placeholders 0.5,2.0 / 0.5,2.5.
#
# Now the EXPERIMENT window 0.2,2.2, the same value used by SBND Q/L matching
# (beam_pref_tlow/thigh) and by run_nusel_evt.sh.  This matters since doc 56:
# beam_window_only defaults ON, so the window no longer only tunes TGM's in-beam
# protection -- it decides which bundles get a steiner graph and a tagger verdict
# at all.  The old placeholders would have silently dropped every in-beam bundle
# below 0.5 us: 6 of the 27 in the 30-event scan (0.520 .. 0.735 us).
BEAM_WINDOW_MC="0.2,2.2"
BEAM_WINDOW_DATA="0.2,2.2"
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -N) SBND_SAMPLE="$2"; shift 2 ;;
        -N*) SBND_SAMPLE="${1#-N}"; shift ;;
        mc|data) MODE="$1"; shift ;;
        -p) PIPELINE="$2"; shift 2 ;;
        -stm|--stm) PIPELINE="switch_scope,steiner,fiducialutils,tagger_check_stm"; shift ;;
        -tgm|--tgm) NU=1; PIPELINE="switch_scope,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm"; shift ;;
        -nu|--nu) NU=1; PIPELINE="switch_scope,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm,tagger_check_neutrino"; shift ;;
        -dnn|--dnn) NU=1; PIPELINE="switch_scope,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm,tagger_check_neutrino"
                    DL_WEIGHTS="$DL_WEIGHTS_DEFAULT"; shift ;;
        -no-dnn|--no-dnn) DL_WEIGHTS=""; shift ;;
        -bw) BEAM_WINDOW="$2"; shift 2 ;;
        -stm-fit|--stm-fit) STM_FIT=1; shift ;;
        -no-stm-fit|--no-stm-fit) STM_FIT=0; shift ;;
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
# -stm-fit appends the Magnify-tracking ROOT dump to the tagger pipeline.
if [ "$STM_FIT" = 1 ] && [ -n "$PIPELINE" ]; then
    PIPELINE="$PIPELINE,stm_magnify"
fi

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
    # Preload only when the DL vertex can actually run, i.e. DL weights are set
    # AND tagger_check_neutrino is in this pipeline.  The -stm / -tgm / bare -p
    # arms must keep the exact process environment they had before the doc-pr/4
    # default flip -- they are A/B comparison arms.
    # if-form, not `[ -n .. ] && ..`: under `set -e` the && list would return 1
    # and abort the run whenever the condition is false.
    case "$PIPELINE_CODE" in
        *"'tagger_check_neutrino'"*) NEED_DL_PRELOAD=1 ;;
        *) NEED_DL_PRELOAD=0 ;;
    esac
    if [ -n "$DL_WEIGHTS" ] && [ "$NEED_DL_PRELOAD" = 1 ]; then
        export LD_PRELOAD="$PYLIB"
    else
        unset LD_PRELOAD
    fi
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
        --tla-code "save_stm_fit=$([ "$STM_FIT" = 1 ] && echo true || echo false)" \
        `# TGM/FC knobs pinned to the PRE-ADOPTION values.  The canonical config` \
        `# adopted the production operating point as its defaults on 2026-07-27` \
        `# (doc 64), and this per-event debug/A-B runner passes only a subset of` \
        `# the PR job's TLAs, so without these pins its -tgm / -nu / -dnn demos` \
        `# would silently switch to merge-aware TGM + the wider FV margins.  That` \
        `# change is unvalidated for this runner, so it stays byte-identical here.` \
        `# DELETE this block to follow production (that is what run_nusel_evt.sh` \
        `# passes); -no-nucand etc. do not exist in this runner.` \
        --tla-code "tgm_neutrino_candidate=false" \
        --tla-code "tgm_chord_charge=false" \
        --tla-str  "tgm_chord_mode=chord" \
        --tla-code "tgm_component_extremes=false" \
        --tla-code "tgm_component_rescue=false" \
        --tla-code "tgm_rescue_chord=false" \
        --tla-code "tgm_main_pair=false" \
        --tla-str  "tgm_main_pair_mode=path" \
        --tla-code "tgm_fv_zmax_margin=3" \
        --tla-code "tgm_fv_zmax_margin_interior=0" \
        --tla-code "tgm_fv_x_margin=2" \
        --tla-code "tgm_fv_y_margin=2.5" \
        -c "$JSONNET"
    # A failed SCN import is only a WARN and the code quietly reverts to the
    # geometric vertex -- an rc=0 run with different physics.  Never let that
    # pass unnoticed (docs/pr/4).
    if [ "$NEED_DL_PRELOAD" = 1 ] && [ -n "$DL_WEIGHTS" ] && grep -q "DL vertex failed" "$LOG" 2>/dev/null; then
        echo "[evt $EVT_ID] *** WARNING: DL vertex requested but FAILED; this run used the" >&2
        echo "                geometric vertex.  See: grep 'DL vertex failed' $LOG" >&2
    fi
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
