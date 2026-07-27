#!/bin/bash
# Per-event SBND charge-light (Q/L) matching — standalone, self-contained.  -h for help.
# Usage: ./run_ql_evt.sh [mc|data] [-N n] <idx|all> [-a anode]
#        ./run_ql_evt.sh [mc|data] [-N n]            # list available events
#   mode:  mc (default) | data
#   -N:    event-sample size (default 10); e.g. -N 100 uses input-100evt-<mode>
#   idx:   1-based event index into the chosen sample/mode; all = every event (parallel)
#   -a:    restrict to one anode (0 or 1)
#
# Self-contained: reads the toolkit's OWN per-event imaging output
#   work/evt<ID>/icluster-apa{0,1}-{active,masked}.npz   (from ./run_img_evt.sh)
# plus that event's opflash (split from input_files/input-<N>evt-<mode>/), runs
# per-APA clustering + Q/L matching, and writes work/ql_evt<ID>/mabc-all-apa.zip
# (img + clustering + 2-view dead-area + op/Q-L layers).
#
# Prerequisite:  ./run_img_evt.sh <mode> [-N n] <idx>  (produces the per-event active+masked npz)
# Workflow:      run_img_evt.sh <mode>  ->  run_ql_evt.sh <mode>
#
# Both mc and data are wired: per-event imaging comes from
# input_files/input-<N>evt-<mode>/frames-dnn.tar.bz2 via run_img_evt.sh, and the
# opflash is split from input_files/input-<N>evt-<mode>/opflash_apa{0,1}.tar.gz.
# NB: opflash members are keyed by event id, so an event lacking opflash in the
# sample (e.g. parts of the 100evt set) is skipped with a clear message.

set -e

SBND_DIR=$(cd "$(dirname "$0")" && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WCT_BASE}/wire-cell-data/sbnd/photodet:${WIRECELL_PATH}

. "$SBND_DIR/_runlib.sh"

SEMIMODEL=semi-analytical-sbnd.json
JSONNET="$SBND_DIR/wct-clus-matching-perevt.jsonnet"
# Q/L drift / diffusion (documented values; same as run_clust_QL_evt.sh).
DL=6.5781; DT=13.1349; LIFETIME=6; DRIFTSPEED=1.563   # DL/DT = SBND physical diffusion (cm^2/s)

usage() {
    cat <<EOF
Per-event SBND charge-light (Q/L) matching — standalone, self-contained.

Usage: $(basename "$0") [mc|data] [-N n] [-a anode] <idx|all>
       $(basename "$0") [mc|data] [-N n]            # list available events

  mc|data   input set (default mc)
  idx       1-based event index into the chosen sample/mode (see no-arg listing);
            'all' matches every event in parallel (cap nproc, SBND_MAX_JOBS=N)
  -a        restrict to one anode (0 or 1)
  -calib    also dump work/ql_evt<ID>/calib-evt<ID>.json for the ql_scan
            hand-scan viewer (matched zip output is unchanged)
  -cathode-diag  log the cathode-crossing TPC0/TPC1 offset three-vector
            diagnostic (grep QLCATHODE in the run log; output unchanged)
  -auto-mask  RE-ASSERT the per-event dynamic dead-PMT auto-mask (masks a PMT
            that is dead in THIS event while its live neighbours fire; grep
            QLAUTOMASK in the run log).  The auto-mask is already ON
            unconditionally in the production config (cfg/.../sbnd/qlmatching.jsonnet
            match_data), so this flag does NOT toggle it -- without it the key is
            simply omitted and production is inherited.  To genuinely disable it,
            pass auto_mask=false to that module directly (doc 64)
  -beam-pref  re-assert the beam-window flash preference overlay. The
            preference is ON in the production config since the round-2
            adoption (weight 0.5, rescue 0.2, gate ks 0.3 / pred 2%; doc 22),
            so this flag only matters together with BEAMPREF_WEIGHT /
            BEAMPREF_RESCUE env overrides to scan a different operating point
  -no-main-flag  do NOT stamp flag_main_cluster on every matched bundle main
            (QLMatching flag_matched_mains).  ON BY DEFAULT for this chain:
            without it only the mains that decompose_cluster_groups SPLIT carry
            the flag, so a compact single-component match is skipped by
            TaggerCheckTGM/STM/FC and reads "no-bundle" in the nusel table
            (evt286021, 1.158 us beam flash -> 141-pt cluster, 437 PE predicted).
            The C++/jsonnet defaults stay FALSE -- only this runner opts in, so
            every other config and detector is unaffected.
            Env: SBND_QL_MAIN_FLAG=0.
  -save-pctree  also write the post-QL point-cloud tree to
            work/ql_evt<ID>/pctree-evt<ID>.tar.gz (TensorDM tar; input of the
            pattern-recognition job; off by default => byte-identical)
  -save-rcid  persist the flash-merge per-blob provenance (real_cluster_id /
            real_cluster_main perblob arrays) through the pctree tarball, so
            the PR job can tell which points were the bundle's main cluster
            (TaggerCheckTGM main mode "real", doc 38).  Only meaningful WITH
            -save-pctree.  Off by default => byte-identical tarball.
            Env: SBND_QL_SAVE_RCID=1.
  -no-rcid-global  doc 53: DISABLE the real_cluster_id re-stamp, which is ON by
                default (C++ default true).  On, real_cluster_id is one globally
                unique ident epoch; off, it is a mix of two dense 1..N epochs and
                31% of values name two clusters.  Group membership is unchanged
                either way => the un-merge and TGM are verdict-neutral, so pass
                this ONLY for A/B archaeology.  Env: SBND_QL_RCID_GLOBAL=0.
                (-rcid-global is accepted and also implies -save-rcid.)
  -save-assoc   doc 52: also record the isolated grouping's main+associated
                partition (assoc_cluster_id / assoc_cluster_main per blob),
                carried across every later merge and saved into the pctree
                tarball, so the PR job can un-merge it (run_nusel_evt.sh
                -unmerge-assoc) and fit the main alone.  Use with -save-pctree.
                Env: SBND_SAVE_ASSOC=1.
  -no-realign   A/B ARCHAEOLOGY ONLY: reproduce the pre-fix behavior where
                QLMatching's decompose/recompose left every perblob array
                misaligned with the permuted blob order (doc 52 §12; the fix
                realign_perblob is ON by C++ default since §12.8).
                Env: SBND_REALIGN=0.
  -trace-bee  DIAGNOSTIC: dump one Bee "clustering" layer per clustering step
            (tr<NN>_<Type>, per-APA zips AND mabc-all-apa.zip), so a merge can
            be attributed to the pass that made it.  Match pieces by point
            coordinates -- cluster ids are renumbered after every step.  Adds
            ~20 layers per zip; use a FRESH work root.  Off => byte-identical.
            Env: SBND_TRACE_BEE=1.  See sbnd_xin/docs/51.
  -lm       LM (light-mismatch) tagger (QLMatching lm_tagger).  Judges every
            FINAL matched bundle by per-drift-side KS shape + pred/meas
            normalization (the photon library's unmodeled cathode leakage
            light means the far side is never judged alone); stamps cluster
            scalar "lm_flag" (0 pass / 1 low-energy / 2 light mismatch) read
            by nusel_extract.py's lm column, dumps per-bundle lm* keys into
            the -calib JSON, and logs "LM verdict" lines.  Off by default =>
            byte-identical (C++/jsonnet defaults stay FALSE).
            Env: SBND_QL_LM=1.

Requires: run_img_evt.sh first (work/evt<ID>/icluster-apa{0,1}-{active,masked}.npz).
Opflash comes from input_files/input-<N>evt-<mode>/opflash_apa{0,1}.tar.gz (keyed
by event id; events with no opflash in the sample are skipped).
Output: work/ql_evt<EVT_ID>/mabc-all-apa.zip
EOF
    sbnd_common_help
}

# --- Args ---
MODE=mc
ANODE=""
# Joint matching is the default: both TPCs go into ONE QLMatching node (which
# matches each APA and merges, replacing the separate PointTreeMerging). Pass
# --per-apa (or SBND_JOINT=0) to run the legacy two-node per-APA path instead.
# Both are byte-identical today; the joint node is where the joint algorithm lands.
JOINT=true
[ "${SBND_JOINT:-}" = "0" ] && JOINT=false
# -calib: also dump the per-event Q/L hand-scan calibration JSON
# (work/ql_evt<ID>/calib-evt<ID>.json) for the ql_scan viewer. Off by default;
# the matched mabc-all-apa.zip output is byte-identical with or without it.
CALIB=""
# CALIB_SUFFIX: optional suffix inserted before .json in the calib-dump filename
# (e.g. CALIB_SUFFIX=.nl -> calib-evt<ID>.nl.json), so an NL rerun does not clobber
# the linear dump the hand-scan was based on. Default empty = calib-evt<ID>.json.
CALIB_SUFFIX="${CALIB_SUFFIX:-}"
# PMT_NL: enable the per-PMT predicted-light non-linearity correction (threaded into
# the matching jsonnet as --tla-code pmt_nl). true (default, SBND going forward) maps
# predicted PE into the saturated space. PMT_NL=false = identity (OFF baseline).
PMT_NL="${PMT_NL:-true}"
# -cathode-diag: log the cathode-crossing TPC0/TPC1 offset three-vector diagnostic
# (grep "QLCATHODE" in the run log). Off by default; matched output is unchanged.
CATHODE=""
# -save-pctree: also write the post-QL point-cloud tree (all-APA MABC output:
# live+dead trees, cluster_t0/flash annotations, opflash PC) to
# work/ql_evt<ID>/pctree-evt<ID>.tar.gz -- the persistent intermediate format
# consumed by the downstream pattern-recognition job (run_pr_evt.sh).  Off by
# default (production byte-identical; the sink stays a dump_mode no-op).
# See sbnd/docs/sbnd-pattern-recognition.md.
SAVEPCT=""
# -auto-mask: enable the per-event dynamic dead-PMT auto-mask (QLMatching auto_mask).
# Off by default (production byte-identical); masks a PMT that is dead in THIS event
# while its live neighbours fire (a run-dead channel absent from the static ch_mask).
AUTOMASK="false"
# -beam-pref: beam-window flash preference (QLMatching beam_pref). Off by default
# (production byte-identical); when on, a flash in the (0.2, 2.2) us BNB window is
# exempt from the rival-consistent cull and its LASSO columns are shrunk less, so
# the beam flash competes for (and tends to win) its clusters. See
# docs/22_ql-beam-flash-preference.md (reco1 evts 246579/116962 case study).
BEAMPREF="false"
# BEAMPREF_WEIGHT / BEAMPREF_RESCUE: beam-preference operating point (inert unless
# -beam-pref). weight = LASSO L1 multiplier for beam-window bundles (validated 0.5;
# 0.2 over-collects), rescue = empty-flash rescue steal guard scale. Env-overridable
# for scans, e.g. BEAMPREF_WEIGHT=0.35 ./run_ql_evt.sh data all -beam-pref -calib.
BEAMPREF_WEIGHT="${BEAMPREF_WEIGHT:-0.5}"
BEAMPREF_RESCUE="${BEAMPREF_RESCUE:-0.2}"
# flag_matched_mains (QLMatching): stamp flag_main_cluster on EVERY matched bundle
# main, not only on the ones decompose_cluster_groups split.  DEFAULT ON for this
# chain -- the knob-off path leaves compact single-component matches unflagged and
# therefore invisible to TaggerCheckTGM/STM/FC and to the nusel bundle table, which
# is not the behavior we want from the selection.  The C++ and jsonnet defaults are
# still false; this runner passes main_flag=true explicitly, so nothing outside this
# chain changes.  -no-main-flag / SBND_QL_MAIN_FLAG=0 restores the legacy set.
MAINFLAG="${SBND_QL_MAIN_FLAG:-1}"
# LM (light-mismatch) tagger (QLMatching lm_tagger): per-drift-side KS +
# normalization verdict on every final matched bundle, stamped as cluster
# scalar "lm_flag" + calib-dump lm* keys.  OFF by default (C++/jsonnet defaults
# false => byte-identical); -lm / SBND_QL_LM=1 opts in.  See docs/34_lm-tagger.md.
QL_LM="${SBND_QL_LM:-0}"
# Persist flash-merge per-blob provenance through the pctree tarball (doc 38).
# OFF by default: opt in with -save-rcid / SBND_QL_SAVE_RCID=1.
SAVE_RCID="${SBND_QL_SAVE_RCID:-0}"
# -rcid-global (doc 53): re-stamp real_cluster_id into ONE globally unique ident
# epoch at save time, instead of the legacy mix of the epoch examine_bundles
# recorded and the epoch enumerate_idents has since installed (31% of values
# name two clusters).  Group membership is unchanged, so the un-merge and TGM
# are unaffected; what changes is that the value becomes a valid event-wide key.
# Implies -save-rcid.  Env: SBND_QL_RCID_GLOBAL=1.
RCID_GLOBAL="${SBND_QL_RCID_GLOBAL:-1}"
if [ "$RCID_GLOBAL" = 1 ]; then SAVE_RCID=1; fi
# Per-step Bee trace for merge attribution.  OFF by default (diagnostic only).
TRACE_BEE="${SBND_TRACE_BEE:-0}"
# -save-assoc: doc 52.  clustering_isolated records the main + associated
# partition it creates into the per-blob pair assoc_cluster_id /
# assoc_cluster_main, merge_clusters carries it across every later merge, and
# MABC homogenizes it into the pctree tarball, so the PR job can undo the
# grouping (run_nusel_evt.sh -unmerge-assoc) and fit the main alone.
# Only meaningful together with -save-pctree.
SAVE_ASSOC="${SBND_SAVE_ASSOC:-0}"
# -no-realign: doc 52 §12.8, A/B archaeology ONLY.  QLMatching realign_perblob
# is ON by C++ default (recompose keeps the perblob rows aligned with the
# permuted blob order); this reproduces the pre-fix misaligned behavior.
# Env: SBND_REALIGN=0.
REALIGN="${SBND_REALIGN:-}"
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -N) SBND_SAMPLE="$2"; shift 2 ;;
        -N*) SBND_SAMPLE="${1#-N}"; shift ;;
        mc|data) MODE="$1"; shift ;;
        -auto-mask|--auto-mask) AUTOMASK="true"; shift ;;   # before -a* (it starts with -a)
        -beam-pref|--beam-pref) BEAMPREF="true"; shift ;;
        -a) ANODE="$2"; shift 2 ;;
        -a*) ANODE="${1#-a}"; shift ;;
        -s|--per-apa|--separate) JOINT=false; shift ;;
        -calib|--calib) CALIB=1; shift ;;
        -cathode-diag|--cathode-diag) CATHODE=1; shift ;;
        -save-pctree|--save-pctree) SAVEPCT=1; shift ;;
        -save-rcid|--save-rcid) SAVE_RCID=1; shift ;;
        -trace-bee|--trace-bee) TRACE_BEE=1; shift ;;
        -save-assoc|--save-assoc) SAVE_ASSOC=1; shift ;;
        -rcid-global|--rcid-global) RCID_GLOBAL=1; SAVE_RCID=1; shift ;;
        -no-rcid-global|--no-rcid-global) RCID_GLOBAL=0; shift ;;
        -no-realign|--no-realign) REALIGN=0; shift ;;
        -no-main-flag|--no-main-flag) MAINFLAG=0; shift ;;
        -lm|--lm) QL_LM=1; shift ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"

case "$MODE" in
    mc)   REALITY=sim ;;
    data) REALITY=data ;;
esac

sbnd_check_sample "$MODE" || exit 1
INPUT_DIR=$(sbnd_input_dir "$MODE")
[ -f "$JSONNET" ]   || { echo "ERROR: missing jsonnet: $JSONNET" >&2; exit 1; }

# Event-id list/order from the mode's frames-dnn archive (mode- and
# sample-agnostic; same order downstream pipelines stream events, so idx is
# stable).  The 100evt-mc set carries duplicate frames; load_events dedups.
load_events "$MODE" || exit 1
EVENT_IDS=("${SBND_EVENTS[@]}")

if [ $# -eq 0 ]; then
    echo "Sample: input-${SBND_SAMPLE}evt-$MODE   (${#EVENT_IDS[@]} events)"
    echo "Events for mode '$MODE' (idx -> EVT_ID):"
    for i in "${!EVENT_IDS[@]}"; do printf "  %2d -> %s\n" $((i + 1)) "${EVENT_IDS[$i]}"; done
    exit 0
fi

ANODE_CODE="[0,1]"
[ -n "$ANODE" ] && ANODE_CODE="[$ANODE]"

process_event() {
    local IDX=$1
    local EVT_ID="${EVENT_IDS[$((IDX - 1))]}"
    [ -n "$EVT_ID" ] || { echo "ERROR: invalid idx $IDX (1..${#EVENT_IDS[@]})" >&2; return 1; }

    local IMGDIR="$SBND_WORK_ROOT/evt${EVT_ID}"     # per-event imaging output (run_img_evt.sh)
    local QLDIR="$SBND_WORK_ROOT/ql_evt${EVT_ID}"    # isolated Q/L workspace + output
    local LOG="$QLDIR/wct_ql_evt${EVT_ID}.log"

    # Require the toolkit's per-event imaging output (active + masked, both anodes).
    local n kind f
    for n in 0 1; do
        for kind in active masked; do
            f="$IMGDIR/icluster-apa${n}-${kind}.npz"
            if [ ! -s "$f" ]; then
                echo "ERROR: missing $f — run ./run_img_evt.sh $IDX first" >&2
                return 2
            fi
        done
    done

    # Fresh isolated workspace: symlink the imaging npz, stage the per-event opflash.
    rm -rf "$QLDIR"; mkdir -p "$QLDIR"
    for n in 0 1; do
        ln -s "$IMGDIR/icluster-apa${n}-active.npz" "$QLDIR/icluster-apa${n}-active.npz"
        ln -s "$IMGDIR/icluster-apa${n}-masked.npz" "$QLDIR/icluster-apa${n}-masked.npz"
    done

    # Split this event's opflash from the bundled archive (members are uniquely
    # suffixed by event ident: opflash_tensorset_<ID>_* and opflash_tensor_<ID>_*).
    for n in 0 1; do
        local src="$INPUT_DIR/opflash_apa${n}.tar.gz"
        [ -s "$src" ] || { echo "ERROR: missing opflash: $src" >&2; return 1; }
        # Skip events that have no opflash in this sample (e.g. parts of the
        # 100evt set): without flashes Q/L matching cannot run.
        if ! tar tzf "$src" | grep -q "^opflash_tensorset_${EVT_ID}_"; then
            echo "[skip] evt $EVT_ID: no opflash for apa${n} in $(basename "$src")" >&2
            return 2
        fi
        local stage="$QLDIR/.opflash_stage_apa${n}"
        mkdir -p "$stage"
        tar xzf "$src" -C "$stage" --wildcards "opflash_tensorset_${EVT_ID}_*" "opflash_tensor_${EVT_ID}_*"
        ( cd "$stage" && tar czf "$QLDIR/opflash_apa${n}.tar.gz" opflash_tensorset_${EVT_ID}_* opflash_tensor_${EVT_ID}_* )
        rm -rf "$stage"
    done

    # Bee run/subrun numbers.  The reco1 extractions (run_reco1_dump.sh) carry the
    # art run/subrun alongside the event in each opflash tensor-set metadata, so
    # the Bee JSONs can show the full (run, subrun, event) triplet instead of
    # 0/0/<evt>.  Read them here rather than plumbing them through the cluster
    # files: this is the minimal, default-inert layer.  Samples whose metadata
    # lacks the keys (yuhw's larsoft dumps: only frame_apply_at_caf) keep the
    # historical run=0 subrun=0 => unchanged Bee output for every older sample.
    local RUN_NO=0 SUBRUN_NO=0 _md
    _md=$(tar xzOf "$QLDIR/opflash_apa0.tar.gz" "opflash_tensorset_${EVT_ID}_metadata.json" 2>/dev/null) || _md=''
    if [ -n "$_md" ]; then
        local _rse
        _rse=$(printf '%s' "$_md" | python3 -c \
            'import json,sys; d=json.load(sys.stdin); print(int(d.get("run",0)), int(d.get("subrun",0)))' \
            2>/dev/null) && [ -n "$_rse" ] && read -r RUN_NO SUBRUN_NO <<< "$_rse"
    fi

    # Optional hand-scan calibration dump (one per-event JSON, both TPCs).
    local CALIB_TLA=()
    [ -n "$CALIB" ] && CALIB_TLA=(--tla-str "calib_dump=$QLDIR/calib-evt${EVT_ID}${CALIB_SUFFIX}.json")
    # Optional cathode-crossing offset diagnostic (logs QLCATHODE lines to $LOG).
    local CATHODE_TLA=()
    [ -n "$CATHODE" ] && CATHODE_TLA=(--tla-str "cathode_diag=on")
    # Optional persistent post-QL point-cloud tree (PR-job intermediate file).
    local SAVEPCT_TLA=()
    [ -n "$SAVEPCT" ] && SAVEPCT_TLA=(--tla-str "save_tensors=$QLDIR/pctree-evt${EVT_ID}.tar.gz")

    echo "[evt $EVT_ID] rse=($RUN_NO, $SUBRUN_NO, $EVT_ID)"
    echo "[evt $EVT_ID] Q/L matching (anodes $ANODE_CODE, joint=$JOINT${CALIB:+, calib}${CATHODE:+, cathode-diag}${SAVEPCT:+, save-pctree}) -> $QLDIR/mabc-all-apa.zip"
    rm -f "$LOG"
    wire-cell \
        -l stderr -l "${LOG}:debug" -L debug \
        --tla-str  "input=$QLDIR" \
        --tla-code "anode_indices=${ANODE_CODE}" \
        --tla-str  "output_dir=$QLDIR" \
        --tla-code "run=${RUN_NO}" --tla-code "subrun=${SUBRUN_NO}" --tla-code "event=${EVT_ID}" \
        --tla-str  "reality=$REALITY" \
        --tla-str  "semimodel_file=$SEMIMODEL" \
        --tla-code "DL=$DL" --tla-code "DT=$DT" \
        --tla-code "lifetime=$LIFETIME" --tla-code "driftSpeed=$DRIFTSPEED" \
        --tla-code "joint=$JOINT" \
        --tla-code "pmt_nl=$PMT_NL" \
        --tla-code "auto_mask=$AUTOMASK" \
        --tla-code "beam_pref=$BEAMPREF" \
        --tla-code "beam_pref_weight=$BEAMPREF_WEIGHT" \
        --tla-code "beam_pref_rescue=$BEAMPREF_RESCUE" \
        --tla-code "main_flag=$([ "$MAINFLAG" = 1 ] && echo true || echo false)" \
        --tla-code "lm=$([ "$QL_LM" = 1 ] && echo true || echo false)" \
        --tla-code "save_rcid=$([ "$SAVE_RCID" = 1 ] && echo true || echo false)" \
        --tla-code "trace_bee=$([ "$TRACE_BEE" = 1 ] && echo true || echo false)" \
        --tla-code "save_assoc=$([ "$SAVE_ASSOC" = 1 ] && echo true || echo false)" \
        --tla-code "rcid_global=$([ "$RCID_GLOBAL" = 0 ] && echo false || echo null)" \
        --tla-code "realign=$([ "$REALIGN" = 0 ] && echo false || echo null)" \
        "${CALIB_TLA[@]}" \
        "${CATHODE_TLA[@]}" \
        "${SAVEPCT_TLA[@]}" \
        -c "$JSONNET"
    echo "[evt $EVT_ID] done -> $QLDIR/mabc-all-apa.zip${CALIB:+ (+ calib-evt${EVT_ID}.json)}"
}

mkdir -p "$SBND_WORK_ROOT"
IDX="$1"
if [ "$IDX" = "all" ]; then
    batch_init
    echo "Mode $MODE: ${#EVENT_IDS[@]} events. Parallel jobs: $BATCH_MAX"
    for i in $(seq 1 "${#EVENT_IDS[@]}"); do
        _evtid="${EVENT_IDS[$((i - 1))]}"
        _blog="$SBND_WORK_ROOT/.batch_ql_evt${_evtid}.log"
        batch_wait_slot
        ( process_event "$i" ) > "$_blog" 2>&1 &
        BATCH_PIDS[$!]=$_evtid
        echo "  [start] idx=$i evt=$_evtid  log: $_blog"
    done
    batch_drain
    batch_summary
    exit $?
else
    process_event "$IDX"
    exit $?
fi
