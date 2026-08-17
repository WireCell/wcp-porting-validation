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

JSONNET="$SBND_DIR/wct-clus-matching-perevt.jsonnet"
# NOTE (doc 68): the SBND production operating point is NOT in this script.  It
# lives in the job's TLA defaults, cfg/pgrapher/experiment/sbnd/
# wct-clus-matching-perevt.jsonnet -- including semimodel_file, the LAr
# DL/DT/lifetime/driftSpeed set, joint, pmt_nl, main_flag, lm, save_rcid and
# save_assoc.  This runner passes only what is per-event (paths, run/subrun/
# event, an anode restriction) plus explicit overrides.  Do not reintroduce a
# default value here: two copies of the operating point is what doc 68 removed.

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
            (QLMatching flag_matched_mains).  ON in the config: without it only
            the mains that decompose_cluster_groups SPLIT carry the flag, so a
            compact single-component match is skipped by TaggerCheckTGM/STM/FC
            and reads "no-bundle" in the nusel table (evt286021, 1.158 us beam
            flash -> 141-pt cluster, 437 PE predicted).
            Env: SBND_QL_MAIN_FLAG=0.
  -save-pctree  also write the post-QL point-cloud tree to
            work/ql_evt<ID>/pctree-evt<ID>.tar.gz (TensorDM tar; input of the
            pattern-recognition job; off by default => byte-identical)
  -save-rcid  re-assert the flash-merge per-blob provenance (real_cluster_id /
            real_cluster_main perblob arrays) in the pctree tarball, which lets
            the PR job tell which points were the bundle's main cluster
            (TaggerCheckTGM main mode "real", doc 38).  ON in the config since
            doc 68; only meaningful WITH -save-pctree.  See -no-save-rcid.
            Env: SBND_QL_SAVE_RCID=1.
  -no-rcid-global  doc 53: DISABLE the real_cluster_id re-stamp, which is ON by
                C++ default.  On, real_cluster_id is one globally unique ident
                epoch; off, it is a mix of two dense 1..N epochs and 31% of
                values name two clusters.  Group membership is unchanged either
                way => the un-merge and TGM are verdict-neutral, so pass this
                ONLY for A/B archaeology.  Env: SBND_QL_RCID_GLOBAL=0.
                (-rcid-global is accepted and is a no-op: 1 means "inherit".)
  -save-assoc   doc 52: re-assert the isolated grouping's main+associated
                partition (assoc_cluster_id / assoc_cluster_main per blob),
                carried across every later merge and saved into the pctree
                tarball, so the PR job can un-merge it (its default pipeline
                runs unmerge_assoc) and fit the main alone.  ON in the config
                since doc 68; use with -save-pctree.  Env: SBND_SAVE_ASSOC=1.
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
  -no-lm    turn OFF the LM (light-mismatch) tagger (QLMatching lm_tagger),
            which is ON in the config since doc 64.  It judges every FINAL
            matched bundle by per-drift-side KS shape + pred/meas normalization
            (the photon library's unmodeled cathode leakage light means the far
            side is never judged alone); stamps cluster scalar "lm_flag"
            (0 pass / 1 low-energy / 2 light mismatch) read by
            nusel_extract.py's lm column, dumps per-bundle lm* keys into the
            -calib JSON, and logs "LM verdict" lines.  -lm re-asserts it.
            Env: SBND_QL_LM=0.
  -no-save-rcid / -no-save-assoc
            drop the flash-merge (doc 38) / isolated-grouping (doc 52) per-blob
            provenance from the -save-pctree tarball.  Both ON in the config
            since doc 68; the PR job's default pipeline reads them
            (unmerge_bundle mode "real", unmerge_assoc).
            Env: SBND_QL_SAVE_RCID=0 / SBND_SAVE_ASSOC=0.

NOTE (doc 68): the production operating point lives in the job config
(cfg/pgrapher/experiment/sbnd/wct-clus-matching-perevt.jsonnet), not in this
script.  Every flag above is an OVERRIDE; with no flags you get production.

Requires: run_img_evt.sh first (work/evt<ID>/icluster-apa{0,1}-{active,masked}.npz).
Opflash comes from input_files/input-<N>evt-<mode>/opflash_apa{0,1}.tar.gz (keyed
by event id; events with no opflash in the sample are skipped).
Output: work/ql_evt<EVT_ID>/mabc-all-apa.zip
EOF
    sbnd_common_help
}

# --- Args ---
# Every knob variable below is EMPTY by default, meaning "pass no TLA" => the
# config default (= the production operating point) applies.  1 forces true,
# 0 forces false.  See the doc-68 note at the top of this file.
MODE=mc
ANODE=""
# --per-apa / SBND_JOINT=0: legacy two-node per-APA graph (one QLMatching per
# APA -> PointTreeMerging -> all-APA MABC) instead of the production joint node.
JOINT="${SBND_JOINT:-}"
# -calib: also dump the per-event Q/L hand-scan calibration JSON
# (work/ql_evt<ID>/calib-evt<ID>.json) for the ql_scan viewer. Off by default;
# the matched mabc-all-apa.zip output is byte-identical with or without it.
CALIB=""
# CALIB_SUFFIX: optional suffix inserted before .json in the calib-dump filename
# (e.g. CALIB_SUFFIX=.nl -> calib-evt<ID>.nl.json), so an NL rerun does not clobber
# the linear dump the hand-scan was based on. Default empty = calib-evt<ID>.json.
CALIB_SUFFIX="${CALIB_SUFFIX:-}"
# PMT_NL=0: turn OFF the per-PMT predicted-light non-linearity correction (the
# identity/OFF baseline).  ON in the config; unset here = inherit.
PMT_NL="${PMT_NL:-}"
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
# -auto-mask: RE-ASSERT the per-event dynamic dead-PMT auto-mask (QLMatching
# auto_mask), which the production config already enables unconditionally.
AUTOMASK=""
# -beam-pref: RE-ASSERT the beam-window flash preference overlay (QLMatching
# beam_pref), also already ON in the production config since the round-2
# adoption. Only useful together with BEAMPREF_WEIGHT / BEAMPREF_RESCUE to scan
# a different operating point. See docs/22_ql-beam-flash-preference.md.
BEAMPREF=""
# BEAMPREF_WEIGHT / BEAMPREF_RESCUE: beam-preference operating-point OVERRIDES
# (inert unless -beam-pref; empty = keep the config's validated 0.5 / 0.2).
# weight = LASSO L1 multiplier for beam-window bundles (0.2 over-collects),
# rescue = empty-flash rescue steal guard scale. For scans, e.g.
# BEAMPREF_WEIGHT=0.35 ./run_ql_evt.sh data all -beam-pref -calib.
BEAMPREF_WEIGHT="${BEAMPREF_WEIGHT:-}"
BEAMPREF_RESCUE="${BEAMPREF_RESCUE:-}"
# -no-main-flag / SBND_QL_MAIN_FLAG=0: legacy flag_matched_mains set (only the
# mains decompose_cluster_groups SPLIT carry flag_main_cluster, so a compact
# single-component match is invisible to TaggerCheckTGM/STM/FC and reads
# "no-bundle" in the nusel table).  ON in the config.
MAINFLAG="${SBND_QL_MAIN_FLAG:-}"
# -no-lm / SBND_QL_LM=0: pre-LM baseline (QLMatching lm_tagger off).  ON in the
# config since doc 64.  See docs/34_lm-tagger.md.
QL_LM="${SBND_QL_LM:-}"
# -no-save-rcid / SBND_QL_SAVE_RCID=0: drop the flash-merge per-blob provenance
# from the pctree tarball (doc 38).  ON in the config since doc 68 -- the PR
# job's default unmerge_bundle runs in "real" mode, which reads these arrays.
SAVE_RCID="${SBND_QL_SAVE_RCID:-}"
# -no-rcid-global (doc 53): do NOT re-stamp real_cluster_id into ONE globally
# unique ident epoch at save time, leaving the legacy mix of the epoch
# examine_bundles recorded and the epoch enumerate_idents has since installed
# (31% of values name two clusters).  Group membership is unchanged, so the
# un-merge and TGM are verdict-neutral either way.  ON by C++ default; only the
# OFF case is expressible as a TLA.  Env: SBND_QL_RCID_GLOBAL=0.
RCID_GLOBAL="${SBND_QL_RCID_GLOBAL:-}"
# Per-step Bee trace for merge attribution.  OFF in the config (diagnostic only).
TRACE_BEE="${SBND_TRACE_BEE:-}"
# -save-assoc: doc 52.  clustering_isolated records the main + associated
# partition it creates into the per-blob pair assoc_cluster_id /
# assoc_cluster_main, merge_clusters carries it across every later merge, and
# MABC homogenizes it into the pctree tarball, so the PR job can undo the
# grouping (its default pipeline runs unmerge_assoc) and fit the main alone.
# ON in the config since doc 68; -no-save-assoc / SBND_SAVE_ASSOC=0 drops them,
# which makes that visitor a warning + no-op.
SAVE_ASSOC="${SBND_SAVE_ASSOC:-}"
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
        -auto-mask|--auto-mask) AUTOMASK=1; shift ;;   # before -a* (it starts with -a)
        -beam-pref|--beam-pref) BEAMPREF=1; shift ;;
        -a) ANODE="$2"; shift 2 ;;
        -a*) ANODE="${1#-a}"; shift ;;
        -s|--per-apa|--separate) JOINT=0; shift ;;
        -joint|--joint) JOINT=1; shift ;;
        -calib|--calib) CALIB=1; shift ;;
        -cathode-diag|--cathode-diag) CATHODE=1; shift ;;
        -save-pctree|--save-pctree) SAVEPCT=1; shift ;;
        -save-rcid|--save-rcid) SAVE_RCID=1; shift ;;
        -no-save-rcid|--no-save-rcid) SAVE_RCID=0; shift ;;
        -trace-bee|--trace-bee) TRACE_BEE=1; shift ;;
        -save-assoc|--save-assoc) SAVE_ASSOC=1; shift ;;
        -no-save-assoc|--no-save-assoc) SAVE_ASSOC=0; shift ;;
        -rcid-global|--rcid-global) RCID_GLOBAL=1; shift ;;
        -no-rcid-global|--no-rcid-global) RCID_GLOBAL=0; shift ;;
        -no-realign|--no-realign) REALIGN=0; shift ;;
        -main-flag|--main-flag) MAINFLAG=1; shift ;;
        -no-main-flag|--no-main-flag) MAINFLAG=0; shift ;;
        -lm|--lm) QL_LM=1; shift ;;
        -no-lm|--no-lm) QL_LM=0; shift ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"

# --- Override TLAs -------------------------------------------------------
# Built once: an EMPTY knob emits no TLA, so the job's own default (= the SBND
# production operating point, doc 68) stands.  Only an explicit 1/0 speaks.
KNOB_TLA=()
knob_bool() {   # knob_bool <tla-name> <value>
    case "$2" in
        1) KNOB_TLA+=(--tla-code "$1=true") ;;
        0) KNOB_TLA+=(--tla-code "$1=false") ;;
    esac
}
knob_bool joint      "$JOINT"
knob_bool pmt_nl     "$PMT_NL"
knob_bool auto_mask  "$AUTOMASK"
knob_bool main_flag  "$MAINFLAG"
knob_bool lm         "$QL_LM"
knob_bool save_rcid  "$SAVE_RCID"
knob_bool trace_bee  "$TRACE_BEE"
knob_bool save_assoc "$SAVE_ASSOC"
# rcid_global / realign are C++-default-TRUE tri-states whose config default is
# null (= inherit), so only the OFF case is expressible; 1 means "inherit", not
# "emit true", and must stay silent to keep the compiled config unchanged.
[ "$RCID_GLOBAL" = 0 ] && KNOB_TLA+=(--tla-code "rcid_global=false")
[ "$REALIGN" = 0 ]     && KNOB_TLA+=(--tla-code "realign=false")
# -beam-pref re-asserts the overlay; its numbers are only read then.
if [ "$BEAMPREF" = 1 ]; then
    KNOB_TLA+=(--tla-code "beam_pref=true")
    [ -n "$BEAMPREF_WEIGHT" ] && KNOB_TLA+=(--tla-code "beam_pref_weight=$BEAMPREF_WEIGHT")
    [ -n "$BEAMPREF_RESCUE" ] && KNOB_TLA+=(--tla-code "beam_pref_rescue=$BEAMPREF_RESCUE")
fi

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

# Both anodes is the job default; only a -a restriction needs a TLA.
ANODE_CODE="[0,1]"
ANODE_TLA=()
if [ -n "$ANODE" ]; then
    ANODE_CODE="[$ANODE]"
    ANODE_TLA=(--tla-code "anode_indices=${ANODE_CODE}")
fi

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
    # Cathode BUNDLE rescue (doc pr/14): joins a cathode crosser whose halves
    # sit in different flash bundles (flash-reco absorbing-window defect).
    # SBND production default is ON since pr/14 §7.4 validation (owner decision
    # 2026-08-01); unset inherits that config default.  SBND_CATHODE_RESCUE=0
    # forces the pre-pr/14 legacy path (byte-identical to before the knob),
    # =1 forces on explicitly.
    local CRESCUE_TLA=()
    case "${SBND_CATHODE_RESCUE:-}" in
        1) CRESCUE_TLA=(--tla-code "cathode_rescue=true") ;;
        0) CRESCUE_TLA=(--tla-code "cathode_rescue=false") ;;
    esac
    # Unmatched-cluster adoption pass of the rescue (doc pr/17): a flashless
    # cluster geometrically continuing a beam-window cluster across the
    # cathode is merged into the beam bundle (56463 veto-ON: the rejoined nu
    # is unmatched and invisible downstream).  SBND production default is ON
    # since doc pr/17 (validated: fires 1/1000 mcp1k, 0/48 nueCC48); unset
    # inherits that config default.  SBND_RESCUE_UNMATCHED=0 forces the
    # pre-pr/17 legacy path (byte-identical), =1 forces on explicitly.
    case "${SBND_RESCUE_UNMATCHED:-}" in
        1) CRESCUE_TLA+=(--tla-code "cathode_rescue_unmatched=true") ;;
        0) CRESCUE_TLA+=(--tla-code "cathode_rescue_unmatched=false") ;;
    esac
    # Cathode bundle rescue ROUND 2 (docs/73): four independent openings of a
    # measured blocker behind the 10 in-beam events doc 72 §A found still cut at
    # the cathode.  ALL FOUR are SBND config default FALSE -- unset inherits
    # that, so an ordinary run is unaffected.  =1 turns one on for an A/B arm.
    #   SBND_RESCUE_IN_BEAM     class A (2 evts): far half may be in-beam
    #   SBND_RESCUE_GEOM_FIRST  class B (6 evts): drop the dt0 window behind a
    #                           tightened geometry (the widest-reaching knob)
    #   SBND_RESCUE_PIERCE      class C (2 evts): cathode-piercing agreement
    #                           replaces the conn angle where conn is drift-
    #                           dominated or too short to define a direction
    #   SBND_RESCUE_PIERCE_CUT  <cm> operating point of that test (default 8)
    #   SBND_RESCUE_DEST_BEAM   round-2 pairs adopt the BEAM bundle instead of
    #                           the length-based a/b/c/d rule
    case "${SBND_RESCUE_IN_BEAM:-}" in
        1) CRESCUE_TLA+=(--tla-code "rescue_in_beam_far=true") ;;
        0) CRESCUE_TLA+=(--tla-code "rescue_in_beam_far=false") ;;
    esac
    case "${SBND_RESCUE_GEOM_FIRST:-}" in
        1) CRESCUE_TLA+=(--tla-code "rescue_geom_first=true") ;;
        0) CRESCUE_TLA+=(--tla-code "rescue_geom_first=false") ;;
    esac
    case "${SBND_RESCUE_PIERCE:-}" in
        1) CRESCUE_TLA+=(--tla-code "rescue_pierce_test=true") ;;
        0) CRESCUE_TLA+=(--tla-code "rescue_pierce_test=false") ;;
    esac
    [ -n "${SBND_RESCUE_PIERCE_CUT:-}" ] && \
        CRESCUE_TLA+=(--tla-code "rescue_pierce_cut=${SBND_RESCUE_PIERCE_CUT}*10")  # cm -> wc units
    case "${SBND_RESCUE_DEST_BEAM:-}" in
        1) CRESCUE_TLA+=(--tla-code "rescue_dest_beam_for_new=true") ;;
        0) CRESCUE_TLA+=(--tla-code "rescue_dest_beam_for_new=false") ;;
    esac
    # Round 3 (docs/73 sec 12): the beam-side donor must BE its bundle's
    # matched main (evt 51128: a 3.8 cm associated fragment displaced the
    # real 57.7 cm main).  SBND config default FALSE; unset inherits that.
    case "${SBND_RESCUE_BEAM_MAIN:-}" in
        1) CRESCUE_TLA+=(--tla-code "rescue_beam_main_only=true") ;;
        0) CRESCUE_TLA+=(--tla-code "rescue_beam_main_only=false") ;;
    esac
    # Separate vertex veto (doc pr/15): per-APA separate() un-splits a
    # neutrino-vertex "V" (run 18255 evt 56463, nu cut in two at its vertex).
    # SBND production default is ON; unset inherits that config default.
    # SBND_SEP_VVETO=0 forces the pre-pr/15 legacy path (byte-identical to
    # before the knob), =1 forces on explicitly.
    local VVETO_TLA=()
    case "${SBND_SEP_VVETO:-}" in
        1) VVETO_TLA=(--tla-code "sep_vertex_veto=true") ;;
        0) VVETO_TLA=(--tla-code "sep_vertex_veto=false") ;;
    esac
    # Neutrino-stage iso-band guard (doc pr/18): the per-APA neutrino stage may
    # not merge an isochronous band with a non-band cluster spanning > 20 cm of
    # drift, even on touch (run 18255 evt 10550: separate correctly split the
    # nu candidate off the cosmic band, neutrino re-merged them at 0.31 cm).
    # SBND production default is ON; unset inherits that config default.
    # SBND_NU_ISO_GUARD=0 forces the pre-pr/18 legacy path (byte-identical to
    # before the knob), =1 forces on explicitly.
    local ISOGUARD_TLA=()
    case "${SBND_NU_ISO_GUARD:-}" in
        1) ISOGUARD_TLA=(--tla-code "nu_iso_band_guard=true") ;;
        0) ISOGUARD_TLA=(--tla-code "nu_iso_band_guard=false") ;;
    esac
    # doc pr/66: record each nu_iso_band_guard refusal as per-blob provenance
    # so the all-APA clustering chain -- which has no iso-band guard of its
    # own -- declines to re-merge the exact pair the per-APA chain already
    # refused.  SBND PRODUCTION ON, owner flip 2026-08-12; unset inherits that
    # default.  SBND_NU_BAND_VETO=0 forces the legacy path (byte-identical
    # pre-flip config), =1 forces on (a no-op today since the config default
    # is already on).
    local BANDVETO_TLA=()
    case "${SBND_NU_BAND_VETO:-}" in
        1) BANDVETO_TLA=(--tla-code "nu_band_veto=true") ;;
        0) BANDVETO_TLA=(--tla-code "nu_band_veto=false") ;;
    esac
    # doc pr/19 campaign pair (SBND config default OFF pending validation):
    # iso_cathode_guard = per-APA clustering_isolated declines the 80 cm
    # small->big absorb for near-cathode smalls; adopt_nu_fragments = all-APA
    # rescue pass 3 adopts the freed flashless fragments into a beam-window
    # cluster (run 18253 evt 444187).  Unset inherits the config default;
    # =1 forces on, =0 forces the legacy path (byte-identical).
    local OC_TLA=()
    case "${SBND_ISO_CATHODE_GUARD:-}" in
        1) OC_TLA=(--tla-code "iso_cathode_guard=true") ;;
        0) OC_TLA=(--tla-code "iso_cathode_guard=false") ;;
    esac
    case "${SBND_ADOPT_NU_FRAG:-}" in
        1) OC_TLA+=(--tla-code "adopt_nu_fragments=true") ;;
        0) OC_TLA+=(--tla-code "adopt_nu_fragments=false") ;;
    esac
    # doc pr/20 Part I P1: write the per-blob "real_cluster_was_main" array on
    # the all-APA flash-time merge (which member was a matched bundle MAIN
    # before the merge demoted it).  Unset inherits the config default (false);
    # =1 forces on, =0 forces the legacy path (byte-identical).
    case "${SBND_SAVE_WASMAIN:-}" in
        1) OC_TLA+=(--tla-code "save_bundle_main_provenance=true") ;;
        0) OC_TLA+=(--tla-code "save_bundle_main_provenance=false") ;;
    esac

    echo "[evt $EVT_ID] rse=($RUN_NO, $SUBRUN_NO, $EVT_ID)"
    local _ov=""
    [ ${#KNOB_TLA[@]} -gt 0 ] && _ov=", overrides: ${KNOB_TLA[*]}"
    echo "[evt $EVT_ID] Q/L matching (anodes $ANODE_CODE${CALIB:+, calib}${CATHODE:+, cathode-diag}${SAVEPCT:+, save-pctree}${_ov}) -> $QLDIR/mabc-all-apa.zip"
    rm -f "$LOG"
    wire-cell \
        -l stderr -l "${LOG}:debug" -L debug \
        --tla-str  "input=$QLDIR" \
        "${ANODE_TLA[@]}" \
        --tla-str  "output_dir=$QLDIR" \
        --tla-code "run=${RUN_NO}" --tla-code "subrun=${SUBRUN_NO}" --tla-code "event=${EVT_ID}" \
        --tla-str  "reality=$REALITY" \
        "${CALIB_TLA[@]}" \
        "${CATHODE_TLA[@]}" \
        "${SAVEPCT_TLA[@]}" \
        "${CRESCUE_TLA[@]}" \
        "${VVETO_TLA[@]}" \
        "${ISOGUARD_TLA[@]}" \
        "${BANDVETO_TLA[@]}" \
        "${OC_TLA[@]}" \
        "${KNOB_TLA[@]}" \
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
