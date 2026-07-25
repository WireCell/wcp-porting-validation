#!/bin/bash
# Per-event SBND neutrino-selection chain — standalone, self-contained.  -h for help.
# Usage: ./run_nusel_evt.sh [mc|data] [-N n] [-bw l,h] <idx|all>
#        ./run_nusel_evt.sh [mc|data] [-N n]           # list available events
#
# Chain (see docs/23_nusel-tgm-stm-chain.md):
#   1. Q/L matching with the persisted point-cloud tree
#      (./run_ql_evt.sh <mode> -save-pctree <idx>; skipped when
#      work/ql_evt<ID>/pctree-evt<ID>.tar.gz already exists);
#   2. the PR tagger job on the loaded tree:
#      switch_scope -> steiner -> fiducialutils -> tagger_check_tgm ->
#      tagger_check_stm -> tagger_check_fc
#      (wct-pr-perevt.jsonnet, beam window on cluster_t0;
#       check_neutrino_candidate ON by default -- see -no-nucand);
#   3. the per-bundle label table (nusel_extract.py): one row per matched
#      bundle (= main cluster) with flash time / PE, bundle size, TGM/STM/FC
#      verdicts and a label, plus a row per beam-window flash that matched
#      no bundle.
#
# Honors SBND_WORK_ROOT / SBND_INPUT_DIR / SBND_SAMPLE like run_ql_evt.sh, so
# it runs directly on reprocessing trees, e.g. the MCP2025C reco1 sample:
#   SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
#   SBND_WORK_ROOT=$PWD/work-mcp10 ./run_nusel_evt.sh data all
#
# Prerequisite: ./run_img_evt.sh <mode> <idx> (per-event imaging npz).
# Output: work/nusel_evt<ID>/{wct_nusel_evt<ID>.log, mabc-pr.zip, nusel-evt<ID>.tsv}
#         'all' also merges: work/nusel-table.tsv (per bundle) and
#         work/nusel-events.tsv (per event).

set -e

# pwd -P: sbnd_xin is reachable through a symlink (toolkit/sbnd_xin); the
# PR jsonnet's relative import '../particle_dataset.jsonnet' only resolves
# from the REAL location (wcp-porting-img/sbnd/), so canonicalize here.
SBND_DIR=$(cd "$(dirname "$0")" && pwd -P)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WCT_BASE}/wire-cell-data/sbnd/photodet:${WIRECELL_PATH}

. "$SBND_DIR/_runlib.sh"

JSONNET="$SBND_DIR/wct-pr-perevt.jsonnet"
# Same LAr TLAs as run_ql_evt.sh / run_pr_evt.sh (identical anode/params objects).
DL=6.2; DT=9.8; LIFETIME=6; DRIFTSPEED=1.563
# The tagger pipeline.  fiducialutils MUST precede the taggers (they silently
# no-op without it); TGM before STM (STM skips TGM-flagged mains).
# tagger_check_fc is LAST: it evaluates every in-scope main regardless of the
# TGM/STM verdicts (so position does not change its coverage), and running it
# after them keeps their inputs free of the PCA/hough/steiner-boundary caches
# cluster_fc_check populates.  Verified: TGM/STM verdicts on the 10-event
# MCP2025C sample are identical with and without it (docs/25_fc-flag.md).
PIPELINE="switch_scope,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm,tagger_check_fc"

usage() {
    cat <<EOF
Per-event SBND neutrino selection: Q/L bundles -> TGM/STM taggers -> label table.

Usage: $(basename "$0") [mc|data] [-N n] [-bw l,h] [-save-pr-tree] <idx|all>
       $(basename "$0") [mc|data] [-N n]            # list available events

  mc|data       input set (default mc)
  idx           1-based event index (same numbering as run_ql_evt.sh);
                'all' runs every event (parallel, cap nproc / SBND_MAX_JOBS)
  -bw l,h       beam window [l,h) in us on cluster_t0 (= matched flash time,
                trigger-referenced).  Default ${BEAM_WINDOW}.  Used for the TGM
                beam protection AND the table's in_beam / label columns.
  -nucand       enable the ported check_neutrino_candidate veto in
                tagger_check_tgm: in-beam-window bundles may then be tagged
                TGM when the Dijkstra path-topology veto clears them.
                ON BY DEFAULT for this chain since doc 26.
  -chord        merge-aware TGM: sets BOTH tgm_chord_charge (require charge
                along the chord between the two extreme points) AND
                tgm_component_extremes (find the extremes per connected
                component and union them, so a real through-goer inside a
                merged bundle can still form its own pair).  Guards
                against the QL flash-time merge grafting a detached fragment
                into the tagged cluster (evt285185 clus 20: a 14-pt speck 608 cm
                away made a 173 cm track look through-going).  C++/jsonnet
                default stays FALSE; only this flag opts in.
                Env: SBND_TGM_CHORD=1.
                The guard measures support in "path" mode by default: the two
                extremes must be joined by a piecewise charge path through the
                cluster's own points with no jump > 30 cm.  The original
                straight-chord sampling ("chord", docs 29) falsely rejected
                curved tracks (evt285185 cluster 16: a continuous 480 cm
                top->anode crosser bows 10 cm off its chord -- doc 31).
                Env: SBND_TGM_CHORD_MODE=chord restores the doc-29 behavior.
  -rescue       component rescue in tagger_check_tgm: a component shorter than
                component_min_length (10 cm) still donates its extreme points
                when it is path-connected (30 cm-step charge path) to a
                component that passed the length cut.  Recovers a genuine
                track END that fragments into a small piece behind small gaps
                (evt286681 cluster 7: the last 2.5 cm before the top wall);
                detached merge-grafted specks stay dropped (path-disconnected).
                Requires -chord (uses its component decomposition).
                Env: SBND_TGM_RESCUE=1.
  -rescue-chord rescued-end pairs must ALSO pass the straight-chord support
                test even in path mode.  Rescue's target -- a genuine track
                end fragmented behind small gaps -- lies ON the straight line
                to the track's other end and passes; but path mode alone lets
                a rescued wall-touching speck pair across TWO merged cosmics
                through an L-shaped charge detour (evt288727 cluster 6: two
                touching cosmics, one entering downstream + one entering the
                bottom, faked a TGM -- doc 33).  Requires -rescue.
                Env: SBND_TGM_RESCUE_CHORD=1.
  -fvz <cm>     downstream-z (z ~ 500 cm face) inset of the TGM/FC fiducial
                box in cm (default 3 = legacy).  Shared by tagger_check_tgm
                and tagger_check_fc.  Env: SBND_TGM_FVZ_MARGIN=<cm>.
  -fvzi <cm>    downstream-z inset used by check_tgm's CASE-A INTERIOR
                support tests (chord midpoints + waypoint re-check) when > 0
                (default 0 = off; interior tests then share -fvz).  Makes
                the -fvz widening endpoint-only so a corner clipper running
                ALONG the downstream wall inside the widened band keeps its
                midpoint support (evt287517 cluster 16 / evt289805 cluster 9
                -- doc 35).  TGM only; FC and the endpoint exit tests keep
                -fvz.  Env: SBND_TGM_FVZ_INTERIOR=<cm>.
  -fvx <cm>     drift-x (|x| ~ 200 cm faces) inset of the TGM/FC fiducial
                box in cm, both faces symmetric (default 2 = legacy).
                Shared by tagger_check_tgm and tagger_check_fc.
                Env: SBND_TGM_FVX_MARGIN=<cm>.
  -fvy <cm>     vertical-y (|y| ~ 200 cm faces) inset, both faces symmetric
                (default 2.5 = legacy).  Shared by tagger_check_tgm and
                tagger_check_fc.  Env: SBND_TGM_FVY_MARGIN=<cm>.
  -stm-fit      persist the per-pass STM track fits (doc 40): cluster PCs
                stm_fit/stm_pass/stm_eval, a Bee 'stm_fit' layer in
                mabc-pr.zip, and tracking-stm.root (SbndMagnifyTrackingVisitor
                appended to the pipeline) for Magnify-tracking-SBND.
                DEFAULT OFF = byte-identical legacy outputs.
                Env: SBND_STM_FIT=1.
  -unmerge      restore the prototype "main cluster + associated clusters"
                data product before the taggers (doc 45): split each
                flash-merged bundle back into its pre-merge main (retained,
                keeps flag_main_cluster and matched_flash_gid) and one
                flag_associated_cluster cluster per co-merged member.  The
                'unmerge_bundle' visitor is inserted between switch_scope and
                steiner.  Without it TaggerCheckSTM fits a bundle of detached
                cosmics as ONE track (showcase-stmfit-mc-evt18 cluster 80) and
                check_other_clusters() has no companions left to count.  Uses
                the exact per-blob provenance (this runner passes -save-rcid
                to a Q/L step it launches); falls back per cluster to the
                longest-component proxy on old tarballs.
                DEFAULT OFF = byte-identical legacy pipeline.
                Env: SBND_UNMERGE=1.
  -unmerge-comp like -unmerge but always use the longest-connected-component
                proxy to pick the main.  Env: SBND_UNMERGE_MODE=component.
  -main-pair-real  like -main-pair, but identify the main EXACTLY via the
                per-blob real_cluster_main flash-merge provenance instead of
                the largest-component proxy (doc 38).  Needs a pctree saved
                with run_ql_evt.sh -save-rcid (this runner passes -save-rcid
                to a Q/L step it launches when this flag is on); falls back
                to the proxy on old tarballs.
                Env: SBND_TGM_MAIN_PAIR_MODE=real.
  -main-pair    a pair may tag TGM only when at least one end lies in the
                cluster's MAIN charge component (largest 30 cm path
                component; a cathode crosser is one component).  A merged-in
                fragment that is itself through-going otherwise tags the
                whole bundle on its own pair, which the chord guard
                deliberately allows and the -nucand veto cannot protect
                (it walks the pair's own path): evt289343 main 9, in-beam
                bundle tagged TGM by a 26 cm corner-clipping cosmic fragment
                450 cm from the main track (doc 36).  Off by default =>
                byte-identical.  Env: SBND_TGM_MAIN_PAIR=1.
  -lm           LM (light-mismatch) tagger in the Q/L step (QLMatching
                lm_tagger, doc 34): per-drift-side KS shape + pred/meas
                normalization verdict on every final matched bundle, stamped
                as cluster scalar "lm_flag" -> the table's lm column, with a
                label demotion nu-candidate -> LM for an in-beam bundle whose
                matched flash its charge cannot explain (evt286021 main 8).
                Passes -lm -calib to run_ql_evt.sh when the Q/L step runs;
                an EXISTING pctree is reused untouched, so use a fresh work
                root for an LM pass.  Off by default => byte-identical.
                Env: SBND_QL_LM=1.
  -no-nucand    disable it (pre-doc-26 conservative never-tag-in-beam).
                Env: SBND_TGM_NUCAND=0.
                NB: the C++/jsonnet default stays FALSE -- only this runner
                opts in, so every other config and detector is unaffected.
  -save-pr-tree also re-save the post-PR tree to
                work/nusel_evt<ID>/pctree-pr-evt<ID>.tar.gz (NB: tagger flags
                set on only some clusters do NOT survive re-serialization; the
                table always takes verdicts from the log)

Table: one row per QUALIFYING BUNDLE = a cluster that is (a) flagged
flag_main_cluster and (b) in scope (passes switch_scope's active-volume
filter) -- exactly the population the taggers evaluate.  Main-flagged
out-of-volume shards are reported on stderr, not tabulated.  Flashes are
deduplicated across APAs first (one physical flash is reconstructed once per
TPC), so a beam flash seen in both TPCs counts once.
  run subrun event main_id flash_gid flash_apa flash_grp flash_time_us
  flash_pe flash_pe_grp in_beam n_bundle npts_main npts_bundle len_main_cm
  tgm stm fc label
  fc: fully-contained verdict (TaggerCheckFC).  Orthogonal to the cosmic
      taggers -- it does NOT enter 'label', matching the prototype where FC is
      an independent eval variable, not a veto.
  label: TGM | STM | nu-candidate (in-window, untagged) | not-tagged | no-bundle

Requires per-event imaging (./run_img_evt.sh) in \$SBND_WORK_ROOT/evt<ID>/.
The Q/L step reruns automatically when the pctree tarball is missing.
EOF
    sbnd_common_help
}

MODE=mc
BEAM_WINDOW="0.2,2.2"
SAVEPRT=""
# check_neutrino_candidate veto in tagger_check_tgm (in-beam-window bundles may
# then be tagged TGM).  DEFAULT ON for this chain as of doc 26 -- the knob-off
# path left in-beam bundles untaggable by construction, which is not the
# behavior we want from the selection.  The C++ and jsonnet defaults are still
# false; this runner passes tgm_neutrino_candidate=true explicitly, so nothing
# outside this chain changes.  -no-nucand / SBND_TGM_NUCAND=0 restores it.
NUCAND="${SBND_TGM_NUCAND:-1}"
# Chord-charge guard in tagger_check_tgm.  DEFAULT OFF: opt in with -chord /
# SBND_TGM_CHORD=1 until the flipped verdicts are hand-scanned.
CHORD="${SBND_TGM_CHORD:-0}"
# Support-measurement mode for the guard: "path" (piecewise charge path,
# doc 31) or "chord" (straight-chord sampling, doc 29).  Only used when the
# guard is on.
CHORD_MODE="${SBND_TGM_CHORD_MODE:-path}"
# Component rescue in tagger_check_tgm (path-connected short components keep
# their extremes).  DEFAULT OFF: opt in with -rescue / SBND_TGM_RESCUE=1.
RESCUE="${SBND_TGM_RESCUE:-0}"
# Straight-chord check on rescued-end pairs (doc 33).  DEFAULT OFF: opt in
# with -rescue-chord / SBND_TGM_RESCUE_CHORD=1.
RESCUE_CHORD="${SBND_TGM_RESCUE_CHORD:-0}"
# Downstream-z inset of the TGM/FC fiducial box, cm.  DEFAULT 3 = legacy.
FVZ_MARGIN="${SBND_TGM_FVZ_MARGIN:-3}"
# Interior-support downstream-z inset for check_tgm CASE-A, cm.  DEFAULT 0 =
# off (interior tests share FVZ_MARGIN).
FVZ_INTERIOR="${SBND_TGM_FVZ_INTERIOR:-0}"
# Drift-x / vertical-y insets of the TGM/FC fiducial box, cm, both faces
# symmetric.  DEFAULTS 2 / 2.5 = legacy.
FVX_MARGIN="${SBND_TGM_FVX_MARGIN:-2}"
FVY_MARGIN="${SBND_TGM_FVY_MARGIN:-2.5}"
# TGM pairs must touch the cluster's main charge component (doc 36).
# DEFAULT OFF: opt in with -main-pair / SBND_TGM_MAIN_PAIR=1.
MAIN_PAIR="${SBND_TGM_MAIN_PAIR:-0}"
# How -main-pair identifies the main: "path" = largest-component proxy
# (doc 36), "real" = per-blob flash-merge provenance (doc 38, needs a
# -save-rcid pctree).  -main-pair-real sets both.
MAIN_PAIR_MODE="${SBND_TGM_MAIN_PAIR_MODE:-path}"
# LM (light-mismatch) tagger in the Q/L step (QLMatching lm_tagger, doc 34).
# DEFAULT OFF: opt in with -lm / SBND_QL_LM=1.  Only affects a Q/L step this
# runner LAUNCHES (pctree missing); an existing pctree is reused as-is, so mix
# -lm with pre-LM trees deliberately or start a fresh work root.
QL_LM="${SBND_QL_LM:-0}"
# Persist per-pass STM track fits + tracking-stm.root (doc 40).
# DEFAULT OFF: opt in with -stm-fit / SBND_STM_FIT=1.
STM_FIT="${SBND_STM_FIT:-0}"
# Restore the prototype main+associated data product before the taggers by
# splitting each flash-merged bundle back into its pre-merge members (doc 45).
# DEFAULT OFF: opt in with -unmerge / SBND_UNMERGE=1.  The visitor is inserted
# between switch_scope and steiner (it MUST precede steiner_pc creation).
UNMERGE="${SBND_UNMERGE:-0}"
# How -unmerge identifies the main: "real" = per-blob flash-merge provenance
# (exact, needs a -save-rcid pctree -- this runner passes it), "component" =
# longest connected component (proxy).  -unmerge-comp selects the proxy.
UNMERGE_MODE="${SBND_UNMERGE_MODE:-real}"
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -N) SBND_SAMPLE="$2"; shift 2 ;;
        -N*) SBND_SAMPLE="${1#-N}"; shift ;;
        mc|data) MODE="$1"; shift ;;
        -bw) BEAM_WINDOW="$2"; shift 2 ;;
        -save-pr-tree|--save-pr-tree) SAVEPRT=1; shift ;;
        -nucand|--nucand) NUCAND=1; shift ;;
        -no-nucand|--no-nucand) NUCAND=0; shift ;;
        -chord|--chord) CHORD=1; shift ;;
        -no-chord|--no-chord) CHORD=0; shift ;;
        -rescue|--rescue) RESCUE=1; shift ;;
        -no-rescue|--no-rescue) RESCUE=0; shift ;;
        -rescue-chord|--rescue-chord) RESCUE_CHORD=1; shift ;;
        -no-rescue-chord|--no-rescue-chord) RESCUE_CHORD=0; shift ;;
        -fvz|--fvz) FVZ_MARGIN="$2"; shift 2 ;;
        -fvzi|--fvzi) FVZ_INTERIOR="$2"; shift 2 ;;
        -fvx|--fvx) FVX_MARGIN="$2"; shift 2 ;;
        -fvy|--fvy) FVY_MARGIN="$2"; shift 2 ;;
        -main-pair|--main-pair) MAIN_PAIR=1; shift ;;
        -main-pair-real|--main-pair-real) MAIN_PAIR=1; MAIN_PAIR_MODE=real; shift ;;
        -no-main-pair|--no-main-pair) MAIN_PAIR=0; MAIN_PAIR_MODE=path; shift ;;
        -lm|--lm) QL_LM=1; shift ;;
        -no-lm|--no-lm) QL_LM=0; shift ;;
        -stm-fit|--stm-fit) STM_FIT=1; shift ;;
        -no-stm-fit|--no-stm-fit) STM_FIT=0; shift ;;
        -unmerge|--unmerge) UNMERGE=1; shift ;;
        -unmerge-comp|--unmerge-comp) UNMERGE=1; UNMERGE_MODE=component; shift ;;
        -no-unmerge|--no-unmerge) UNMERGE=0; UNMERGE_MODE=real; shift ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"

# -unmerge inserts the bundle un-merge between switch_scope and steiner.  The
# order matters twice over: it must run AFTER switch_scope (which re-applies
# the T0 correction and carves the out-of-volume shards, and carries the
# per-blob provenance across that carve) and BEFORE steiner (separate() does
# not carry node-local PCs, so a steiner_pc built from the pre-split blob set
# would survive on the retained main).
if [ "$UNMERGE" = 1 ]; then
    PIPELINE="${PIPELINE/switch_scope,/switch_scope,unmerge_bundle,}"
fi
# -stm-fit appends the Magnify-tracking ROOT dump to the tagger pipeline.
if [ "$STM_FIT" = 1 ]; then
    PIPELINE="$PIPELINE,stm_magnify"
fi

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

process_event() {
    local IDX=$1
    local EVT_ID="${EVENT_IDS[$((IDX - 1))]}"
    [ -n "$EVT_ID" ] || { echo "ERROR: invalid idx $IDX (1..${#EVENT_IDS[@]})" >&2; return 1; }

    local QLDIR="$SBND_WORK_ROOT/ql_evt${EVT_ID}"
    local PCT="$QLDIR/pctree-evt${EVT_ID}.tar.gz"
    local NUDIR="$SBND_WORK_ROOT/nusel_evt${EVT_ID}"
    local LOG="$NUDIR/wct_nusel_evt${EVT_ID}.log"

    # 1. Q/L matching with the persisted tree (idempotent: reuse an existing
    # pctree; run_ql_evt.sh is deterministic so a rerun reproduces the same
    # matching, only adding the tarball).
    if [ ! -s "$PCT" ]; then
        local QL_FLAGS=(-save-pctree)
        # -lm: LM tagger + calib dump in the Q/L step (the dump carries the
        # per-bundle lm* metrics the tuning/hand-scan read; doc 34).
        [ "$QL_LM" = 1 ] && QL_FLAGS+=(-lm -calib)
        # -main-pair-real (doc 38) and -unmerge in "real" mode (doc 45) both
        # need the per-blob provenance in the pctree.
        if [ "$MAIN_PAIR_MODE" = real ] || { [ "$UNMERGE" = 1 ] && [ "$UNMERGE_MODE" = real ]; }; then
            QL_FLAGS+=(-save-rcid)
        fi
        echo "[evt $EVT_ID] pctree missing — running Q/L step (${QL_FLAGS[*]})"
        "$SBND_DIR/run_ql_evt.sh" "$MODE" "${QL_FLAGS[@]}" "$IDX" || return 1
        [ -s "$PCT" ] || { echo "ERROR: Q/L step did not produce $PCT" >&2; return 1; }
    fi

    rm -rf "$NUDIR"; mkdir -p "$NUDIR"

    # Run/subrun for the table + Bee labels, from the reco1 opflash metadata
    # (same fallback semantics as run_ql_evt.sh: absent keys -> 0/0).
    local RUN_NO=0 SUBRUN_NO=0 _md
    _md=$(tar xzOf "$QLDIR/opflash_apa0.tar.gz" "opflash_tensorset_${EVT_ID}_metadata.json" 2>/dev/null) || _md=''
    if [ -n "$_md" ]; then
        local _rse
        _rse=$(printf '%s' "$_md" | python3 -c \
            'import json,sys; d=json.load(sys.stdin); print(int(d.get("run",0)), int(d.get("subrun",0)))' \
            2>/dev/null) && [ -n "$_rse" ] && read -r RUN_NO SUBRUN_NO <<< "$_rse"
    fi

    local SAVEPRT_TLA=""
    [ -n "$SAVEPRT" ] && SAVEPRT_TLA="$NUDIR/pctree-pr-evt${EVT_ID}.tar.gz"

    # 2. PR tagger job.  Run from NUDIR so the dump-mode TensorFileSink's
    # trash-pr.tar.gz (if any) lands here, not in the source tree.
    echo "[evt $EVT_ID] rse=($RUN_NO, $SUBRUN_NO, $EVT_ID) taggers ($PIPELINE, bw=[$BEAM_WINDOW] us, chord=$CHORD mode=$CHORD_MODE rescue=$RESCUE rescue_chord=$RESCUE_CHORD main_pair=$MAIN_PAIR/$MAIN_PAIR_MODE fvz=$FVZ_MARGIN fvzi=$FVZ_INTERIOR fvx=$FVX_MARGIN fvy=$FVY_MARGIN lm=$QL_LM stmfit=$STM_FIT unmerge=$UNMERGE/$UNMERGE_MODE)"
    rm -f "$LOG"
    (
        cd "$NUDIR"
        wire-cell \
            -l stderr -l "${LOG}:debug" -L debug \
            --tla-str  "input=$PCT" \
            --tla-code "anode_indices=[0,1]" \
            --tla-str  "output_dir=$NUDIR" \
            --tla-code "run=${RUN_NO}" --tla-code "subrun=${SUBRUN_NO}" --tla-code "event=${EVT_ID}" \
            --tla-str  "reality=$REALITY" \
            --tla-code "DL=$DL" --tla-code "DT=$DT" \
            --tla-code "lifetime=$LIFETIME" --tla-code "driftSpeed=$DRIFTSPEED" \
            --tla-code "pipeline_names=[$(echo "$PIPELINE" | sed "s/[^,]\+/'&'/g")]" \
            --tla-str  "trackfitting_config=$SBND_DIR/sbnd_track_fitting.json" \
            --tla-str  "save_tensors=$SAVEPRT_TLA" \
            --tla-str  "dl_weights=" \
            --tla-code "beam_window_us=[$BEAM_WINDOW]" \
            --tla-code "tgm_neutrino_candidate=$([ "$NUCAND" = 1 ] && echo true || echo false)" \
            --tla-code "tgm_chord_charge=$([ "$CHORD" = 1 ] && echo true || echo false)" \
            --tla-str  "tgm_chord_mode=$CHORD_MODE" \
            --tla-code "tgm_component_extremes=$([ "$CHORD" = 1 ] && echo true || echo false)" \
            --tla-code "tgm_component_rescue=$([ "$RESCUE" = 1 ] && echo true || echo false)" \
            --tla-code "tgm_rescue_chord=$([ "$RESCUE_CHORD" = 1 ] && echo true || echo false)" \
            --tla-code "tgm_main_pair=$([ "$MAIN_PAIR" = 1 ] && echo true || echo false)" \
            --tla-str  "tgm_main_pair_mode=$MAIN_PAIR_MODE" \
            --tla-code "tgm_fv_zmax_margin=$FVZ_MARGIN" \
            --tla-code "tgm_fv_zmax_margin_interior=$FVZ_INTERIOR" \
            --tla-code "tgm_fv_x_margin=$FVX_MARGIN" \
            --tla-code "tgm_fv_y_margin=$FVY_MARGIN" \
            --tla-str  "unmerge_bundle_mode=$UNMERGE_MODE" \
            --tla-code "save_stm_fit=$([ "$STM_FIT" = 1 ] && echo true || echo false)" \
            -c "$JSONNET"
    ) || return 1
    rm -f "$NUDIR/trash-pr.tar.gz"

    # 3. Per-bundle label table.
    python3 "$SBND_DIR/nusel_extract.py" \
        --pctree "$PCT" --prbee "$NUDIR/mabc-pr.zip" --prlog "$LOG" \
        --qlbee "$QLDIR/mabc-all-apa.zip" \
        --beam-window "$BEAM_WINDOW" \
        --run "$RUN_NO" --subrun "$SUBRUN_NO" \
        --out "$NUDIR/nusel-evt${EVT_ID}.tsv" || return 1

    echo "[evt $EVT_ID] table -> $NUDIR/nusel-evt${EVT_ID}.tsv"
    column -t "$NUDIR/nusel-evt${EVT_ID}.tsv" | sed 's/^/  /'
}

mkdir -p "$SBND_WORK_ROOT"
IDX="$1"
if [ "$IDX" = "all" ]; then
    batch_init
    echo "Mode $MODE: ${#EVENT_IDS[@]} events. Parallel jobs: $BATCH_MAX"
    for i in $(seq 1 "${#EVENT_IDS[@]}"); do
        _evtid="${EVENT_IDS[$((i - 1))]}"
        _blog="$SBND_WORK_ROOT/.batch_nusel_evt${_evtid}.log"
        batch_wait_slot
        ( process_event "$i" ) > "$_blog" 2>&1 &
        BATCH_PIDS[$!]=$_evtid
        echo "  [start] idx=$i evt=$_evtid  log: $_blog"
    done
    batch_drain
    batch_summary || exit 1

    # Merge the per-event tables (only those produced by this sample's events).
    _tsvs=()
    for _evtid in "${EVENT_IDS[@]}"; do
        _t="$SBND_WORK_ROOT/nusel_evt${_evtid}/nusel-evt${_evtid}.tsv"
        [ -s "$_t" ] && _tsvs+=("$_t")
    done
    if [ "${#_tsvs[@]}" -gt 0 ]; then
        python3 "$SBND_DIR/nusel_extract.py" --merge "${_tsvs[@]}" \
            --out "$SBND_WORK_ROOT/nusel-table.tsv" \
            --events-out "$SBND_WORK_ROOT/nusel-events.tsv"
        echo
        echo "===== per-bundle table: $SBND_WORK_ROOT/nusel-table.tsv ====="
        column -t "$SBND_WORK_ROOT/nusel-table.tsv"
        echo
        echo "===== per-event summary: $SBND_WORK_ROOT/nusel-events.tsv ====="
        column -t "$SBND_WORK_ROOT/nusel-events.tsv"
    fi
else
    process_event "$IDX"
fi
