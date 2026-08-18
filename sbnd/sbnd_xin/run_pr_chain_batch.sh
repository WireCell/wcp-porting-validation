#!/bin/bash
# doc pr/11: batch driver for the FULL 13-stage SBND PR chain -- the one that
# is missing today.  Every existing runner (run_nusel_evt.sh, run_pr_evt.sh,
# run_full1k_nusel.sh) stops at tagger_check_fc; this one appends
# tagger_check_neutrino -> numu_bdt_scorer -> nue_bdt_scorer -> tracking_visitor
# -> tagger_output, so it is the first driver that can produce numu_score /
# nue_score / T_kine at population scale.
#
# Runs on an EXISTING Q/L pctree root (never re-runs Q/L or imaging -- CLAUDE.md
# M11): fork-by-duplication of run_nusel_evt.sh's process_event() (M10 -- that
# production script stays byte-untouched), with:
#   - the DL (SCN) neutrino vertex ON (the production default since e3d46c91;
#     no dl_weights override, unlike run_nusel_evt.sh which always forces
#     geometric).  Needs libpython RTLD_GLOBAL preloaded or the SCN import
#     fails and the DL vertex SILENTLY falls back to geometric (a WARN
#     "DL vertex failed: ..." is the only sign -- checked per event, see below).
#   - OMP_NUM_THREADS=MKL_NUM_THREADS=1: the DL inference is multithreaded: a
#     30-event probe showed 5.7s wall / 5.6s core pinned vs 5.3s wall / 9.5s
#     core unpinned on the same event.  Pinned so wall time is a comparable
#     single-thread latency number across every event and arm.
#   - -stm-fit (stm_magnify / tracking-stm.root) is OMITTED: doc pr/3 confirms
#     save_stm_fit only gates a diagnostic dump, never the STM verdict itself,
#     and skipping it avoids a second UPDATE-mode ROOT writer per event.
#   - RSE (run/subrun/event) is read from the Q/L job's own opflash metadata
#     (opflash_tensorset_<EVT>_metadata.json), exactly like run_nusel_evt.sh --
#     no external per-sample RSE file needed, and it is correct even for
#     samples spanning multiple runs (nueCC48) or reusing event numbers across
#     samples (MC evt 12 appears in both round1-qlmatch and round2-patrec).
#
# Usage:
#   ./run_pr_chain_batch.sh <ql_root> <out_root> <data|sim> [evt ...]
#     ql_root   dir containing ql_evt<ID>/{pctree-evt<ID>.tar.gz,
#               opflash_apa0.tar.gz, mabc-all-apa.zip} -- e.g. work-mcp1kall-d59k
#     out_root  FRESH dir for pr_evt<ID>/ outputs (refuses to reuse a non-empty
#               one that was not created by a prior run of this script -- M13)
#     data|sim  reality TLA
#     evt ...   optional explicit event-id subset; default = every ql_evt<ID>
#               found under ql_root
#
# Env: PR_JOBS (default 6, M5), SBND_WCT_LOGLEVEL (default debug -- needed for
#      the MABC timing / TaggerCheckNeutrino timing substage lines, doc pr/11
#      sec 3/5; perf=true is already on for SBND).
#
# Per-event output: out_root/pr_evt<ID>/{wct_pr_evt<ID>.log, stdout.log, rc.txt,
#   .time.meta (timecmd.py), mabc-pr.zip, tracking-pr.root,
#   pctree-pr-evt<ID>.tar.gz, nusel-evt<ID>.tsv}.
# Batch: out_root/nusel-table.tsv + nusel-events.tsv (merged, same shape as
#   run_nusel_evt.sh all).  Read the SCORES with pr_scores_table.py.
set -u

SX=$(cd "$(dirname "$0")" && pwd -P)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
TK=$WCT_BASE/toolkit
export WIRECELL_PATH=$TK/cfg:$WCT_BASE/wire-cell-data:$WCT_BASE/wire-cell-data/sbnd/photodet:${WIRECELL_PATH:-}
export PYTHONPATH=$TK/pyutil/python:$WCT_BASE/local/python:$WCT_BASE/wire-cell-python:${PYTHONPATH:-}
AB=$SX/../../abtest

SBND_DIR=$SX  # required by _runlib.sh (unused otherwise here: QLROOT/OUTROOT
              # are explicit args, not SBND_WORK_ROOT-derived).
# _runlib.sh (2026-08-05) refuses to source when SBND_WORK_ROOT is unset AND
# the default work/ is absent (M13 guard, doc 71) -- a false positive for
# THIS script, which never reads $SBND_WORK_ROOT.  Give it a harmless
# placeholder rather than weakening the guard for scripts that do use the
# default (run_img_evt.sh, run_ql_evt.sh, run_nusel_evt.sh, run_pr_evt.sh).
SBND_WORK_ROOT=${SBND_WORK_ROOT:-unused-see-QLROOT-OUTROOT-args}
. "$SX/_runlib.sh"

usage() {
    cat <<EOF
Usage: $0 <ql_root> <out_root> <data|sim> [evt ...]
  ql_root   dir with ql_evt<ID>/{pctree,opflash,mabc-all-apa.zip}
  out_root  fresh output dir (pr_evt<ID>/ per event)
  data|sim  reality TLA
  evt ...   optional event-id subset (default: every ql_evt<ID> in ql_root)
Env: PR_JOBS (default 6), SBND_WCT_LOGLEVEL (default debug)
EOF
}

[ $# -ge 3 ] || { usage; exit 1; }
QLROOT=$(cd "$1" 2>/dev/null && pwd -P) || { echo "ERROR: no such ql_root: $1" >&2; exit 1; }
OUTROOT=$2
REALITY=$3
shift 3

case "$REALITY" in
    data|sim) ;;
    *) echo "ERROR: reality must be data|sim, got '$REALITY'" >&2; exit 1 ;;
esac

mkdir -p "$OUTROOT"
OUTROOT=$(cd "$OUTROOT" && pwd -P)

# doc pr/38 round 3: the PR job re-applies switch_scope's pos_offset
# correction, gated on reality -- so the PR reality MUST match the lineage of
# the ql_root that produced the input pctree, or borderline scope / in-window
# / PID decisions flip wholesale (234638 red-chord track, 55715 whole-event
# EM-shower absorb).  Every run stamps its out_root; the check reads the
# ql_root's stamp, falling back to the reality= field of its .batch_* markers
# (rounds that predate the stamp).  Mismatch is fatal unless
# SBND_ALLOW_REALITY_MISMATCH=1 (deliberate cross-lineage A/B only).
echo "$REALITY" > "$OUTROOT/.lineage_reality"
QL_REALITY=""
if [ -f "$QLROOT/.lineage_reality" ]; then
    QL_REALITY=$(cat "$QLROOT/.lineage_reality")
else
    mapfile -t _QLR < <(grep -sho 'reality=[a-z]*' "$QLROOT"/.batch_*.log 2>/dev/null | sort -u | sed 's/reality=//')
    if [ "${#_QLR[@]}" -gt 1 ]; then
        echo "WARN: ql_root has MIXED reality markers (${_QLR[*]}) -- lineage check skipped, verify by hand" >&2
    elif [ "${#_QLR[@]}" -eq 1 ]; then
        QL_REALITY="${_QLR[0]}"
    fi
fi
if [ -n "$QL_REALITY" ] && [ "$QL_REALITY" != "$REALITY" ]; then
    echo "ERROR: reality mismatch: ql_root lineage is '$QL_REALITY' but this run was given '$REALITY'." >&2
    echo "       (doc pr/38 round 3; set SBND_ALLOW_REALITY_MISMATCH=1 only for a deliberate cross-lineage A/B)" >&2
    [ "${SBND_ALLOW_REALITY_MISMATCH:-0}" = "1" ] || exit 1
fi

if [ $# -ge 1 ]; then
    EVENT_IDS=("$@")
else
    mapfile -t EVENT_IDS < <(ls -d "$QLROOT"/ql_evt*/ 2>/dev/null | sed -E 's#.*/ql_evt([0-9]+)/?$#\1#' | sort -n)
fi
[ "${#EVENT_IDS[@]}" -gt 0 ] || { echo "ERROR: no events found under $QLROOT" >&2; exit 1; }

JSONNET="$SX/wct-pr-perevt.jsonnet"
[ -f "$JSONNET" ] || { echo "ERROR: missing jsonnet: $JSONNET" >&2; exit 1; }

# doc 68: the LAr set and the TrackFitting parameter file come from the job's
# own defaults (cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet); only an
# SBND_TRACKFIT_JSON override is named, for the doc-66 diffusion A/B.
TFJSON="${SBND_TRACKFIT_JSON:-}"
TFJSON_TLA=()
[ -n "$TFJSON" ] && TFJSON_TLA=(--tla-str "trackfitting_config=$TFJSON")

# Production NUF pipeline + the 5 neutrino-PR stages (doc pr/2-3; ordering
# matters -- BDTs after tagger_check_neutrino, nue after numu, tagger_output
# after tracking_visitor because it opens tracking-pr.root in UPDATE mode).
# protect_bundle + steiner_refresh (doc pr/23): uboone's second graph
# examination (Protect_Over_Clustering) -- split each beam-bundle cluster at
# graph component boundaries, cathode re-join per the cfg operating point.
# Position (doc pr/23 ordering decision): AFTER the cosmic taggers and BEFORE
# tagger_check_neutrino, with steiner_refresh (replace=false) right after so
# the split clusters' steiner products are rebuilt -- the prototype-faithful
# order (cosmic verdicts on unsplit clusters, wire-cell-prod-stm.cxx:806;
# protect only in the nue executable, wire-cell-prod-nue.cxx:1322).
PIPELINE="switch_scope,unmerge_bundle,unmerge_assoc,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm,tagger_check_fc,protect_bundle,steiner_refresh,tagger_check_neutrino,numu_bdt_scorer,nue_bdt_scorer,tracking_visitor,tagger_output"

# PR_EXTRA_STAGES: comma-separated cm_by_name stages APPENDED to the pipeline
# above.  EMPTY BY DEFAULT => the pipeline string, and therefore every compiled
# config and every output of this driver, is unchanged.  Names resolve in
# clus_pr's cm_by_name (cfg/pgrapher/experiment/sbnd/clus.jsonnet).
#
# Its reason for existing is the PR event display (doc pr/26):
#   PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh <ql_root> <out> sim 388
# appends PrDisplayDump, which writes pr_evt<ID>/calib-pr-evt<ID>.json next to
# the usual outputs.  That stage is read-only, so an arm run with it must hash
# identically to one run without -- which is the doc's gate.
if [ -n "${PR_EXTRA_STAGES:-}" ]; then
    PIPELINE="$PIPELINE,$PR_EXTRA_STAGES"
fi

# doc sbnd_xin/docs/pr/75: the neutrino-vertex hand scan needs the per-event
# vertex scoreboard (compare_main_vertices scores, DL top-K voxels, the seven
# rerank composite terms, the accept route) beside the pr_display dump.  That
# recording lives in TaggerCheckNeutrino -- PrDisplayDump runs after it and
# cannot see its locals -- so it is a separate C++ knob, DEFAULT OFF.
#
# Turned on automatically whenever pr_display is requested, so producing a
# scannable dump stays the one command doc pr/26 documents.  It is pure
# recording (no decision reads it), which is what keeps the doc pr/26 sec 6
# claim -- a pr_display arm hashes identically to a plain one -- intact.
# SBND_VERTEX_SCOREBOARD set explicitly always wins, including =0/false to
# reproduce a pre-pr/75 dump.
case ",${PR_EXTRA_STAGES:-}," in
    *,pr_display,*) : "${SBND_VERTEX_SCOREBOARD:=true}" ;;
esac

# doc sbnd_xin/docs/pr/79 sec 10: harvest requires the scoreboard, so a
# truthy SBND_DL_VTX_HARVEST defaults the board on too (explicit
# SBND_VERTEX_SCOREBOARD still wins).  pr_display alone does NOT enable
# harvest -- it is opt-in, so the pr/26 sec 6 hash-identity claim for
# scoreboard-only dumps keeps holding.
case "${SBND_DL_VTX_HARVEST:-}" in
    true|1) : "${SBND_VERTEX_SCOREBOARD:=true}" ;;
esac

# SBND PRODUCTION DEFAULT ON since the doc pr/23 sec 9 flip (owner 2026-08-02,
# after the sec 8 fresh-tree gate: 0 event_label / nu_evaluated flips in 572
# valfast events).  SBND_PROTECT_BUNDLE=0 removes both stages = the pre-pr/23
# chain (the arm every pre-flip comparison uses).
if [ "${SBND_PROTECT_BUNDLE:-1}" = 0 ]; then
    PIPELINE="${PIPELINE/protect_bundle,steiner_refresh,/}"
fi

# Cathode kink veto (doc pr/20 Part II B0), cm.  EMPTY = emit no TLA = the job
# default null = C++ 0 = OFF = the legacy kink search, so a bare run of this
# script is byte-identical to before the knob existed.
# Env: SBND_CATHODE_KINK_XCUT=<cm> SBND_CATHODE_X=<cm>.
CATH_TLA=()
[ -n "${SBND_CATHODE_KINK_XCUT:-}" ] && CATH_TLA+=(--tla-code "cathode_kink_xcut=${SBND_CATHODE_KINK_XCUT}")
[ -n "${SBND_CATHODE_X:-}" ]         && CATH_TLA+=(--tla-code "cathode_x=${SBND_CATHODE_X}")
# doc pr/89 Arm D2: post-DL adjustment reach, cm.  EMPTY = no TLA = the cfg
# default null = the C++ defaults (vks 5.0 / mvga 15.0) = byte-identical.
# The TLAs already exist and are fully threaded (wct-pr-perevt.jsonnet:1977,
# :1989); these envs only expose them to a batch A/B.
# Env: SBND_VKS_RADIUS=<cm> SBND_MVGA_RADIUS=<cm>.
[ -n "${SBND_VKS_RADIUS:-}" ]  && CATH_TLA+=(--tla-code "vks_radius=${SBND_VKS_RADIUS}")
[ -n "${SBND_MVGA_RADIUS:-}" ] && CATH_TLA+=(--tla-code "mvga_radius=${SBND_MVGA_RADIUS}")
# doc pr/89 Arm C (C2, owner-approved): rule-1 topology term in the DL rerank
# composite.  EMPTY = no TLA = C++ default 0 = term never computed =
# byte-identical.  The offline C1 replay selected weight 3.0, center 0.
# Env: SBND_DL_VTX_TOPO_WEIGHT=<w> SBND_DL_VTX_TOPO_CENTER=<c>.
[ -n "${SBND_DL_VTX_TOPO_WEIGHT:-}" ] && CATH_TLA+=(--tla-code "dl_vtx_topo_weight=${SBND_DL_VTX_TOPO_WEIGHT}")
[ -n "${SBND_DL_VTX_TOPO_CENTER:-}" ] && CATH_TLA+=(--tla-code "dl_vtx_topo_center=${SBND_DL_VTX_TOPO_CENTER}")
# doc pr/47 sec 8 (O1): wide-baseline cathode kink accept, degrees.  EMPTY =
# no TLA = the job default (SBND ON at 25 deg since 2026-08-07).  0 = force
# the C++ OFF path (legacy kink search, byte-identical); "null" also works
# (omits the key entirely).  Skirt/baseline ride the C++ defaults 3/15 cm.
# Env: SBND_CATHODE_WIDE_KINK_ANGLE=<deg|0|null>.
[ -n "${SBND_CATHODE_WIDE_KINK_ANGLE:-}" ] && CATH_TLA+=(--tla-code "cathode_wide_kink_angle=${SBND_CATHODE_WIDE_KINK_ANGLE}")
# doc pr/25 sec 3: long shower-topology demote length, cm.  EMPTY = no TLA =
# the cfg default (null = OFF = byte-identical).  50 is the scan-supported
# operating point; the guard measures segment_track_length(seg,0).
[ -n "${SBND_SHOWER_TOPO_DEMOTE_LEN:-}" ] && CATH_TLA+=(--tla-code "shower_topo_demote_len=${SBND_SHOWER_TOPO_DEMOTE_LEN}")
# doc pr/31 sec 11 (F2, was P2): skip the stage-3
# segment_determine_shower_direction call so a kShowerTopology segment keeps
# the direction segment_is_shower_topology set -- the prototype's state.
# EMPTY = no TLA = the cfg default (false = OFF = byte-identical).  Set to 1
# for the arm that measures it.
[ "${SBND_SHOWER_TOPO_PROTO_DIR:-}" = 1 ] && CATH_TLA+=(--tla-code "shower_topo_proto_dir=true")
# doc pr/32 sec 11: the four stage-4 (neutrino vertex ID) port fixes.  These
# are TRI-STATE on purpose -- unset = no TLA = whatever the cfg says, 1 = force
# on, 0 = force off -- because once the SBND operating point flips them on, the
# gate arm still has to be able to turn them back off without editing cfg/.
#   F1 SBND_VERTEX_DIR_USE_FIT_POINT   conflict + all-showers geometry from the
#                                      continuous fit, not the Steiner snap
#   F2 SBND_SHOWER_TRAJ_RECHECK_PARITY improve_vertex shower-traj recheck:
#                                      stored-flag gates, 10 cm inner test, and
#                                      a clearable kShowerTrajectory
#   F3 SBND_MAIN_VERTEX_REQUIRE_DESC   drop invalid-descriptor candidates
#                                      before compare_main_vertices scores
#   F4 SBND_MAIN_VERTEX_CANDIDATE_FLAG set kMainCandidate (diagnostic only)
for _pr32 in \
    "SBND_VERTEX_DIR_USE_FIT_POINT:vertex_dir_use_fit_point" \
    "SBND_SHOWER_TRAJ_RECHECK_PARITY:shower_traj_recheck_parity" \
    "SBND_MAIN_VERTEX_REQUIRE_DESC:main_vertex_require_descriptor" \
    "SBND_MAIN_VERTEX_CANDIDATE_FLAG:main_vertex_candidate_flag" ; do
    _env=${_pr32%%:*}; _key=${_pr32#*:}; _val=${!_env:-}
    [ "$_val" = 1 ] && CATH_TLA+=(--tla-code "$_key=true")
    [ "$_val" = 0 ] && CATH_TLA+=(--tla-code "$_key=false")
done
unset _pr32 _env _key _val
# doc pr/31 sec 12: the sec 10.12 topology/PID/direction port fixes.  Same
# tri-state contract as the pr/32 loop above (unset = cfg default, 1 = force
# on, 0 = force off).
#   F5 SBND_CONT_MUON_DIR3_30CM       cont-muon dir3 always at 30 cm
#   F6 SBND_TRACK_COMP_EMPTY_ABSTAIN  empty comparison window abstains
#   F3 SBND_SHOWER_TOPO_RESET         clear kShowerTopology/dirsign at entry
#   F1 SBND_RECLASS_PRESERVE_4MOM     preserve 4-momentum at reclassification
#   F4 SBND_DIR_TRACK_MEDIAN_LOCAL    median over the PID's own dQ/dx vector
#   F7 SBND_EXAMINE_SHOWERS_VTX_BY_INDEX  order the examine_all_showers pair
#                                     by graph index (dormant pending pr/30 F4)
for _pr31 in \
    "SBND_CONT_MUON_DIR3_30CM:cont_muon_dir3_30cm" \
    "SBND_TRACK_COMP_EMPTY_ABSTAIN:track_comp_empty_abstain" \
    "SBND_SHOWER_TOPO_RESET:shower_topo_reset" \
    "SBND_RECLASS_PRESERVE_4MOM:reclass_preserve_4mom" \
    "SBND_DIR_TRACK_MEDIAN_LOCAL:dir_track_median_local" \
    "SBND_EXAMINE_SHOWERS_VTX_BY_INDEX:examine_showers_vertex_by_index" ; do
    _env=${_pr31%%:*}; _key=${_pr31#*:}; _val=${!_env:-}
    [ "$_val" = 1 ] && CATH_TLA+=(--tla-code "$_key=true")
    [ "$_val" = 0 ] && CATH_TLA+=(--tla-code "$_key=false")
done
unset _pr31 _env _key _val
# doc pr/34 sec 11: the sec 10 particle-flow (Bee mc tree) port fixes.  Same
# tri-state contract (unset = cfg default, 1 = force on, 0 = force off).
# DISPLAY-ONLY stage: these move mabc-pr.zip::data/0/0-mc.json and nothing
# else -- gate with pr34_cmp.py, NOT pr32_cmp.py (pctree gate is vacuous).
#   F1 SBND_PF_TRACK_MAIN_CLUSTER_ONLY  track BFS keeps main-cluster idents only
#   F2 SBND_PF_SHOWER_VERTEX_BARRIER    BFS never expands through a shower vtx
#   F3 SBND_PF_SHOWER_PARENT_PRECEDENCE parent shower resolves before track seg
#   F4 SBND_PF_PI0_NODE_PER_ID          one pi0 node per id (home = highest-E
#                                       daughter's parent, owner 2026-08-04)
#   F5 SBND_PF_PDG_NAME_PROTOTYPE_FALLBACK  pi0/nuclei/number PDG-name fallback
for _pr34 in \
    "SBND_PF_TRACK_MAIN_CLUSTER_ONLY:pf_track_main_cluster_only" \
    "SBND_PF_SHOWER_VERTEX_BARRIER:pf_shower_vertex_barrier" \
    "SBND_PF_SHOWER_PARENT_PRECEDENCE:pf_shower_parent_precedence" \
    "SBND_PF_PI0_NODE_PER_ID:pf_pi0_node_per_id" \
    "SBND_PF_PDG_NAME_PROTOTYPE_FALLBACK:pf_pdg_name_prototype_fallback" ; do
    _env=${_pr34%%:*}; _key=${_pr34#*:}; _val=${!_env:-}
    [ "$_val" = 1 ] && CATH_TLA+=(--tla-code "$_key=true")
    [ "$_val" = 0 ] && CATH_TLA+=(--tla-code "$_key=false")
done
unset _pr34 _env _key _val
# doc pr/38 (2026-08-05 owner round 2): the two short-lived pr/38 knobs were
# REMOVED -- their corrected behavior (barrier minus shower start vertices +
# orphan root-leaf safety net) now lives under the existing
# SBND_PF_SHOWER_VERTEX_BARRIER / pf_shower_vertex_barrier knob above.
# doc pr/35 sec 11: the sec 10 energy-reconstruction port fix.  Same tri-state
# contract (unset = cfg default, 1 = force on, 0 = force off).  NOT display-
# only: kine_reco_Enu is numu-BDT var 69 and in the nue reader.  Gate with
# pr35_cmp.py on calib-pr-evt<ID>.json (PR_EXTRA_STAGES=pr_display) + the
# nusel TSVs -- pctree does NOT carry KineInfo, pr32/pr34_cmp gate nothing.
#   F1 SBND_KINE_SHOWER_PDG_LIVE  live start-segment PDG at the four
#                                 fill_kine_tree sites (prototype parity)
_val=${SBND_KINE_SHOWER_PDG_LIVE:-}
[ "$_val" = 1 ] && CATH_TLA+=(--tla-code "kine_shower_pdg_live=true")
[ "$_val" = 0 ] && CATH_TLA+=(--tla-code "kine_shower_pdg_live=false")
unset _val
# doc pr/36 sec 11: the sec 10 tagger-stage port fixes.  Same tri-state
# contract (unset = cfg default, 1 = force on, 0 = force off).  The gate is
# TWO artifacts (doc pr/36 sec 10.9), neither alone sufficient: T_tagger in
# tracking-pr.root (leaf compare, covers F3-F6 fields + scores) PLUS the
# tagger block of calib-pr-evt<ID>.json (PR_EXTRA_STAGES=pr_display; the ONLY
# outlet for F1's match_isFC -- it is NOT booked in T_tagger).  Gate with
# pr36_cmp.py; pctree/mabc must never move.
#   F1 SBND_NEUTRINO_CONSISTENT_FV     match_isFC on the STM/TGM/FC fiducial
#   F3 SBND_SP_SCE_CORRECTION          single-photon SCE gate (OFF on SBND,
#                                      owner 2026-08-04; vacuous, no helper)
#   F4 SBND_TAGGER_ORDERED_SEGSETS     index-ordered accumulation sets (M4)
#   F5 SBND_STEM_ENDPOINT_WCPT_PARITY  prototype wcpt stem-endpoint rule
#   F6 SBND_BROKEN_MUON_CLUSTER_ID_COUNT  distinct cluster IDs, not pointers
#   F7 SBND_NEUTRINO_TYPE_BITMASK      verdict bitmask + neutrino_type/I branch
for _pr36 in \
    "SBND_NEUTRINO_CONSISTENT_FV:neutrino_consistent_fv" \
    "SBND_SP_SCE_CORRECTION:sp_sce_correction" \
    "SBND_TAGGER_ORDERED_SEGSETS:tagger_ordered_segment_sets" \
    "SBND_STEM_ENDPOINT_WCPT_PARITY:stem_endpoint_wcpt_parity" \
    "SBND_BROKEN_MUON_CLUSTER_ID_COUNT:broken_muon_cluster_id_count" \
    "SBND_NEUTRINO_TYPE_BITMASK:neutrino_type_bitmask" ; do
    _env=${_pr36%%:*}; _key=${_pr36#*:}; _val=${!_env:-}
    [ "$_val" = 1 ] && CATH_TLA+=(--tla-code "$_key=true")
    [ "$_val" = 0 ] && CATH_TLA+=(--tla-code "$_key=false")
done
unset _pr36 _env _key _val
# doc pr/33 sec 11: the sec 10 EM-shower-clustering port fixes.  Same
# tri-state contract (unset = cfg default, 1 = force on, 0 = force off).
# The stage is upstream of T_kine, the nue tagger pi0 block, and the Bee
# PR layers (mc.json pi0 grouping + shower_track membership), so gate with
# pr33_cmp.py (all trees + calib + per-member mabc + pctree + TSVs;
# PR_EXTRA_STAGES=pr_display for the calib dump).  pctree and the
# imaging-derived mabc members must never move; ssmsp_* movement in
# T_tagger is the F3 scoping tripwire.
#   F1a SBND_DAUGHTER_COUNT_PROTO_MAIN_VERTEX     _tracks callee, proton skip
#   F1b SBND_DAUGHTER_COUNT_PROTO_EXAMINE_SHOWERS _tracks callee, daughter_length
#   F2a SBND_SHOWER_PDG_FROM_START_SEGMENT  start-segment PDG at 4 sites
#   F2b SBND_SHOWER_PDG_FROM_SHOWER_TYPE    shower type at the inverted site
#   F2c SBND_SHOWER_PDG_EXACT_MUON_TEST     exact ==13 muon test at 2 sites
#   F3  SBND_PI0_ID_SHARED_ALLOCATOR        shared pi0-id stream, no collision
#   F4  SBND_SHOWER_FLAG_PDG_ELECTRON       is_shower gains abs(pdg)==11
#   F5  SBND_SHOWER_LESS_ID_TIEBREAK        shower_less id tie-break (house rule)
for _pr33 in \
    "SBND_DAUGHTER_COUNT_PROTO_MAIN_VERTEX:daughter_count_proto_main_vertex" \
    "SBND_DAUGHTER_COUNT_PROTO_EXAMINE_SHOWERS:daughter_count_proto_examine_showers" \
    "SBND_SHOWER_PDG_FROM_START_SEGMENT:shower_pdg_from_start_segment" \
    "SBND_SHOWER_PDG_FROM_SHOWER_TYPE:shower_pdg_from_shower_type" \
    "SBND_SHOWER_PDG_EXACT_MUON_TEST:shower_pdg_exact_muon_test" \
    "SBND_PI0_ID_SHARED_ALLOCATOR:pi0_id_shared_allocator" \
    "SBND_SHOWER_FLAG_PDG_ELECTRON:shower_flag_pdg_electron" \
    "SBND_SHOWER_LESS_ID_TIEBREAK:shower_less_id_tiebreak" ; do
    _env=${_pr33%%:*}; _key=${_pr33#*:}; _val=${!_env:-}
    [ "$_val" = 1 ] && CATH_TLA+=(--tla-code "$_key=true")
    [ "$_val" = 0 ] && CATH_TLA+=(--tla-code "$_key=false")
done
unset _pr33 _env _key _val
# doc pr/24: isochronous first-segment endpoint finding.  SBND PRODUCTION
# DEFAULT ON since the round-3 flip (owner 2026-08-03) -- EMPTY = no TLA = the
# cfg default = ON at the C++ sizing defaults (40 cm min length, 25 cm max
# drift extent, 0.35 frac, 0.02 quantile, 4 cm diagnostic tube radius, 0.12 min
# sheet aspect).  SBND_ISO_ENDPOINT=0 restores the legacy wire-footprint
# boundary endpoints for an A/B; =1 is now a no-op kept so older runner
# invocations still mean what they said.
if [ "${SBND_ISO_ENDPOINT:-1}" = 0 ]; then
    CATH_TLA+=(--tla-code "iso_endpoint=false")
else
    [ "${SBND_ISO_ENDPOINT:-}" = 1 ] && CATH_TLA+=(--tla-code "iso_endpoint=true")
fi
# doc pr/24 round 3 overrides, validation only (both inert unless
# SBND_ISO_ENDPOINT=1).  SBND_ISO_MIN_ASPECT=0 disables the sheet-aspect gate,
# which is how the aspect distribution is harvested from the DEBUG log without
# a rebuild; SBND_ISO_TUBE_R sets the axis-tube radius in cm.
[ -n "${SBND_ISO_MIN_ASPECT:-}" ] && CATH_TLA+=(--tla-code "iso_endpoint_min_aspect=${SBND_ISO_MIN_ASPECT}")
[ -n "${SBND_ISO_TUBE_R:-}" ] && CATH_TLA+=(--tla-code "iso_endpoint_tube_radius=${SBND_ISO_TUBE_R}")
# doc pr/24 round 5 (sec 18): examine_vertices_3's get_local_extension
# recovery step has no check that it actually extends the vertex (rather than
# retracting it toward the far endpoint) -- worst on an iso_endpoint-picked
# isochronous seed.  DEFAULT ON since the round-6 flip (owner 2026-08-06,
# doc pr/24 sec 19.1) -- EMPTY = no TLA = the cfg default = ON.
# SBND_V3_EXT_GUARD=0 restores the legacy (unguarded) get_local_extension for
# an A/B; =1 is now a no-op kept so older runner invocations still mean what
# they said.
if [ "${SBND_V3_EXT_GUARD:-1}" = 0 ]; then
    CATH_TLA+=(--tla-code "v3_extension_guard=false")
else
    [ "${SBND_V3_EXT_GUARD:-}" = 1 ] && CATH_TLA+=(--tla-code "v3_extension_guard=true")
fi
[ -n "${SBND_V3_EXT_MIN_GAIN:-}" ] && CATH_TLA+=(--tla-code "v3_extension_min_gain=${SBND_V3_EXT_MIN_GAIN}")
# doc pr/67: LOG-ONLY trajectory-coverage probe (why the fitted trajectory does
# not cover the image, worst in isochronous topologies).  Default OFF; the key
# is suppressed in the compiled config unless set, so an unset run is
# byte-identical.  SBND_TRAJ_COVER_PROBE=1 turns the diagnostic lines on.
[ "${SBND_TRAJ_COVER_PROBE:-}" = 1 ] && CATH_TLA+=(--tla-code "traj_cover_probe=true")
# docs/73 sec 12 (round 3).  Both SBND config default FALSE; unset inherits
# that, =1/=0 forces for an A/B arm.
#   SBND_NU_FALLBACK_DEMOTED   when NO candidate survives TaggerCheckNeutrino's
#                              primary loop, consider demoted mains (evt 65289)
#   SBND_ESVA_IGNORE_EMPTY_2D  eliminate_short_vertex_activities case 5: the
#                              empty-2D-index sentinel is "no information", not
#                              "covered" (evt 78242 cross-cathode junction)
case "${SBND_NU_FALLBACK_DEMOTED:-}" in
    1) CATH_TLA+=(--tla-code "nu_fallback_demoted_mains=true") ;;
    0) CATH_TLA+=(--tla-code "nu_fallback_demoted_mains=false") ;;
esac
case "${SBND_ESVA_IGNORE_EMPTY_2D:-}" in
    1) CATH_TLA+=(--tla-code "esva_ignore_empty_2d=true") ;;
    0) CATH_TLA+=(--tla-code "esva_ignore_empty_2d=false") ;;
esac
# doc pr/67 counterfactual: override find_proto_vertex's HARDCODED main-cluster
# branch-search round budget (2).  DIAGNOSTIC ONLY -- a value > 0 changes
# reconstruction output by design.  Unset = the hardcoded budget stands.
[ -n "${SBND_PR_FIND_OTHER_ROUNDS:-}" ] && CATH_TLA+=(--tla-code "pr_find_other_rounds=${SBND_PR_FIND_OTHER_ROUNDS}")
# doc pr/39: exclude a shower's own start vertex from the end_point
# farthest-vertex search (prototype map_vtx_segs parity, same rule as
# fill_sets's exclude_start_vertex, extended to calculate_kinematics{,_long_muon}).
# SBND production default since 2026-08-06 (owner: "turn it on for SBND"),
# exactly the v3_extension_guard idiom -- EMPTY = no TLA = the cfg default
# (ON).  SBND_SHOWER_ENDPOINT_EXCLUDE_START_VERTEX=0 restores the legacy
# (unguarded) end_point search for an A/B; =1 is now a no-op kept so older
# runner invocations still mean what they said.
if [ "${SBND_SHOWER_ENDPOINT_EXCLUDE_START_VERTEX:-1}" = 0 ]; then
    CATH_TLA+=(--tla-code "shower_endpoint_exclude_start_vertex=false")
else
    [ "${SBND_SHOWER_ENDPOINT_EXCLUDE_START_VERTEX:-}" = 1 ] && CATH_TLA+=(--tla-code "shower_endpoint_exclude_start_vertex=true")
fi
# Steiner TERMINAL filter fidelity (doc pr/29 D1 + D12).
# **SBND PRODUCTION DEFAULT ON since the owner flip 2026-08-04** -- these are
# port bugs, not tuning: the toolkit terminal filter was tighter than the WCP
# prototype in two independent ways and was discarding 47.7% of all Steiner
# terminals on evt 388.  EMPTY = emit no TLA = the cfg default = BOTH ON, so a
# bare run is production (doc 68).  Set them to 0 for the pre-fix arm:
#   SBND_STEINER_WIRE_TOL=0    drop the prototype's one wire of slack.
#   SBND_STEINER_ADJ_SLICE=0   go back to stepping the adjacent-slice lookup by
#                              1, which (the map key being in ticks, 4 per
#                              slice on SBND) never resolves -- the dead branch.
# Both together = the legacy arm every pre-flip comparison needs.
[ -n "${SBND_STEINER_WIRE_TOL:-}" ] && \
    CATH_TLA+=(--tla-code "steiner_terminal_wire_tol=${SBND_STEINER_WIRE_TOL}")
case "${SBND_STEINER_ADJ_SLICE:-}" in
    1) CATH_TLA+=(--tla-code "steiner_terminal_adjacent_slice=true") ;;
    0) CATH_TLA+=(--tla-code "steiner_terminal_adjacent_slice=false") ;;
esac
# Steiner EDGE-WEIGHT charge fidelity (doc pr/29 D2).  **SBND PRODUCTION
# DEFAULT ON since the owner flip 2026-08-04**, same reasoning as the two
# above: create_steiner_tree is called with disable_dead_mix_cell=false and the
# toolkit dropped the argument before the edge-weight charge calculation, so
# create_enhanced_steiner_graph's `= true` default won.  EMPTY = no TLA = the
# cfg default = ON.  SBND_STEINER_EDGE_DEAD_MIX=0 restores the dropped
# argument's effect for the pre-fix arm.
case "${SBND_STEINER_EDGE_DEAD_MIX:-}" in
    1) CATH_TLA+=(--tla-code "steiner_edge_charge_forward_dead_mix=true") ;;
    0) CATH_TLA+=(--tla-code "steiner_edge_charge_forward_dead_mix=false") ;;
esac
# protect_bundle knob overrides (doc pr/23, validation only).  EMPTY = no TLA
# = the cfg default = the SBND operating point.  The _XCUT/_DYZ/_DIS values
# are in CM, converted via wirecell.jsonnet because the C++ takes INTERNAL
# units (the cathode_kink_xcut cm-vs-internal trap, doc pr/20).  0 disables
# the cathode re-join pass (prototype-faithful).
[ -n "${SBND_PROTECT_GRAPH:-}" ] && \
    CATH_TLA+=(--tla-str "protect_graph_name=${SBND_PROTECT_GRAPH}")
[ -n "${SBND_PROTECT_REJOIN_XCUT:-}" ] && \
    CATH_TLA+=(--tla-code "protect_cathode_rejoin_xcut=(import 'wirecell.jsonnet').cm*${SBND_PROTECT_REJOIN_XCUT}")
[ -n "${SBND_PROTECT_REJOIN_DYZ:-}" ] && \
    CATH_TLA+=(--tla-code "protect_cathode_rejoin_dyz=(import 'wirecell.jsonnet').cm*${SBND_PROTECT_REJOIN_DYZ}")
[ -n "${SBND_PROTECT_REJOIN_DIS:-}" ] && \
    CATH_TLA+=(--tla-code "protect_cathode_rejoin_dis=(import 'wirecell.jsonnet').cm*${SBND_PROTECT_REJOIN_DIS}")
# Direction-agreement fallback for a dyz-only re-join failure (doc pr/25,
# SBND evt 489327): DESIGNED, NOT YET the SBND default (cfg default = 0 =
# disabled).  SBND_PROTECT_REJOIN_PERP in CM (0/unset = fallback off);
# SBND_PROTECT_REJOIN_ANGLE in DEGREES (no cm conversion);
# SBND_PROTECT_REJOIN_DIR_RADIUS in CM; SBND_PROTECT_REJOIN_DIR_NPTS a bare
# point-count integer.
[ -n "${SBND_PROTECT_REJOIN_PERP:-}" ] && \
    CATH_TLA+=(--tla-code "protect_cathode_rejoin_perp=(import 'wirecell.jsonnet').cm*${SBND_PROTECT_REJOIN_PERP}")
[ -n "${SBND_PROTECT_REJOIN_ANGLE:-}" ] && \
    CATH_TLA+=(--tla-code "protect_cathode_rejoin_angle=${SBND_PROTECT_REJOIN_ANGLE}")
[ -n "${SBND_PROTECT_REJOIN_DIR_RADIUS:-}" ] && \
    CATH_TLA+=(--tla-code "protect_cathode_rejoin_dir_radius=(import 'wirecell.jsonnet').cm*${SBND_PROTECT_REJOIN_DIR_RADIUS}")
[ -n "${SBND_PROTECT_REJOIN_DIR_NPTS:-}" ] && \
    CATH_TLA+=(--tla-code "protect_cathode_rejoin_dir_npts=${SBND_PROTECT_REJOIN_DIR_NPTS}")
# Demoted-main flag on the outer un-merge (doc pr/20 Part I P2).  EMPTY = emit
# no TLA = the job default null = C++ false = OFF.  Needs the Q/L stage to have
# run with SBND_SAVE_WASMAIN=1, else the visitor warns and flags nothing.
# Env: SBND_RESTORE_DEMOTED_MAINS=1|0.
case "${SBND_RESTORE_DEMOTED_MAINS:-}" in
    1) CATH_TLA+=(--tla-code "restore_demoted_mains=true") ;;
    0) CATH_TLA+=(--tla-code "restore_demoted_mains=false") ;;
esac
# Legacy-tree guard (doc pr/23 sec 4.2).  cfg default TRUE: with the restore
# on, a pctree with no wasmain array ABORTS the job instead of silently
# running pre-pr/20 behaviour.  Pass 0 to DECLARE an intentional legacy-tree
# run (e.g. the pinned valfast PR-tail hubs).
# Env: SBND_REQUIRE_WASMAIN=1|0.
case "${SBND_REQUIRE_WASMAIN:-}" in
    1) CATH_TLA+=(--tla-code "require_provenance=true") ;;
    0) CATH_TLA+=(--tla-code "require_provenance=false") ;;
esac
# Let TGM/STM/FC evaluate those demoted mains (doc pr/20 Part I P3).  Inert
# unless SBND_RESTORE_DEMOTED_MAINS=1 above put the flag there.
# Env: SBND_EVAL_DEMOTED_MAINS=1|0.
case "${SBND_EVAL_DEMOTED_MAINS:-}" in
    1) CATH_TLA+=(--tla-code "evaluate_demoted_mains=true") ;;
    0) CATH_TLA+=(--tla-code "evaluate_demoted_mains=false") ;;
esac
# Exempt a flag_demoted_main cluster from TaggerCheckTGM's main_pair_rejects
# veto (doc pr/25, SBND evt 320029): with tgm_main_pair on, that guard reads a
# per-blob array that is all-zero on every demoted main by construction, so it
# vetoed every demoted-main pair unconditionally, before any CASE-A/CASE-B
# boundary geometry ran.  DESIGNED, NOT YET the SBND default -- changes cosmic
# verdicts, owner sign-off pending.  Only meaningful WITH
# SBND_EVAL_DEMOTED_MAINS=1 above.  Env: SBND_TGM_EXEMPT_DEMOTED_MAIN=1|0.
case "${SBND_TGM_EXEMPT_DEMOTED_MAIN:-}" in
    1) CATH_TLA+=(--tla-code "tgm_exempt_demoted_main=true") ;;
    0) CATH_TLA+=(--tla-code "tgm_exempt_demoted_main=false") ;;
esac
# Act on that verdict (doc pr/20 Part I P4): drop a TGM/STM-tagged companion
# from the neutrino's other_clusters, keeping any shorter than the floor.
# Env: SBND_SKIP_COSMIC_COMPANIONS=1|0  SBND_COSMIC_COMPANION_MIN_LEN=<cm>.
case "${SBND_SKIP_COSMIC_COMPANIONS:-}" in
    1) CATH_TLA+=(--tla-code "skip_cosmic_companions=true") ;;
    0) CATH_TLA+=(--tla-code "skip_cosmic_companions=false") ;;
esac
[ -n "${SBND_COSMIC_COMPANION_MIN_LEN:-}" ] && \
    CATH_TLA+=(--tla-code "cosmic_companion_min_length=${SBND_COSMIC_COMPANION_MIN_LEN}")
# sp_photon_flag (doc pr/26 sec. 8.2): store singlephoton_tagger()'s verdict in
# TaggerInfo::photon_flag, the way prototype NeutrinoID.cxx:271 does.  The port
# ran that tagger and filled its ~90 shw_sp_* BDT features but discarded the
# return value, so the uBooNE tagger ntuple's photon_flag branch is a constant
# 0.  SBND DEFAULT ON (owner 2026-08-03): nothing in the chain reads the field,
# so it moves that one branch and nothing else -- 1215 of 1216 T_tagger
# branch-values are identical across the flip (doc pr/26 sec. 9.3).
# Env: SBND_SP_PHOTON_FLAG=1|0.  Unset = no TLA = the cfg default (now ON);
# pass 0 to reproduce the pre-fix gap.
case "${SBND_SP_PHOTON_FLAG:-}" in
    1) CATH_TLA+=(--tla-code "sp_photon_flag=true") ;;
    0) CATH_TLA+=(--tla-code "sp_photon_flag=false") ;;
esac
# DIAGNOSTIC ONLY.  SBND_NU_SKIP_COSMIC=0 clears both cosmic vetoes in
# TaggerCheckNeutrino (per-main and per-bundle), so neutrino PR runs on an
# in-window main that TGM/STM convicted.  This is how you get track_fit /
# shower_track / vertices layers for an event whose only in-window object is a
# cosmic (SBND evt 116962).  It is NOT an operating point -- production is
# BOTH ON (docs/pr/3 sec. 8, pr/16 sec. 7).  Unset = no TLA = cfg default.
case "${SBND_NU_SKIP_COSMIC:-}" in
    0) CATH_TLA+=(--tla-code "nu_skip_cosmic=false")
       CATH_TLA+=(--tla-code "nu_skip_cosmic_bundle=false") ;;
    1) CATH_TLA+=(--tla-code "nu_skip_cosmic=true")
       CATH_TLA+=(--tla-code "nu_skip_cosmic_bundle=true") ;;
esac
# doc pr/40: track (proton/pion/muon) mis-identified as electron.  Same
# tri-state contract (unset = cfg default, 1 = force on, 0 = force off).
#   F1 SBND_TRACK_PID_PERSIST_DQDX     persist type+mass whenever pdg_code!=0
#   F2 SBND_SHOWER_RECLASS_DQDX_GUARD  spare a decisively proton/muon-like
#                                      segment from wholesale reclassification
#   F3 SBND_SHOWER_TOPO_DQDX_GUARD     same guard inside the topology test
for _pr40 in \
    "SBND_TRACK_PID_PERSIST_DQDX:track_pid_persist_dqdx" \
    "SBND_SHOWER_RECLASS_DQDX_GUARD:shower_reclass_dqdx_guard" \
    "SBND_SHOWER_TOPO_DQDX_GUARD:shower_topo_dqdx_guard" ; do
    _env=${_pr40%%:*}; _key=${_pr40#*:}; _val=${!_env:-}
    [ "$_val" = 1 ] && CATH_TLA+=(--tla-code "$_key=true")
    [ "$_val" = 0 ] && CATH_TLA+=(--tla-code "$_key=false")
done
unset _pr40 _env _key _val
# doc pr/40 round 2: two follow-on defects from the pr/40 fix round (zero-KE
# persistence stub, reclass_pinfo negative-KE stub) plus a proton-daughter
# guard.  Same tri-state contract.  F4/F6 are SBND production default ON;
# F5 is STILL default OFF (fires but reverted downstream -- see
# porting_dictionary.md and doc pr/40 round 2; blocked pending round 3).
#   F4 SBND_TRACK_PID_PERSIST_4MOM       segment_cal_4mom unconditionally
#                                        instead of a rest-mass-only stub
#   F5 SBND_SHOWER_PROTON_DAUGHTER_PION  electron -> pion when it fathers a
#                                        PID'd, charge-confirmed proton
#   F6 SBND_RECLASS_NEVER_COMPUTED_KE_FLOOR  never-computed reclass_pinfo
#                                        reads KE==0, not KE==-mass
for _pr40r2 in \
    "SBND_TRACK_PID_PERSIST_4MOM:track_pid_persist_4mom" \
    "SBND_SHOWER_PROTON_DAUGHTER_PION:shower_proton_daughter_pion" \
    "SBND_RECLASS_NEVER_COMPUTED_KE_FLOOR:reclass_never_computed_ke_floor" ; do
    _env=${_pr40r2%%:*}; _key=${_pr40r2#*:}; _val=${!_env:-}
    [ "$_val" = 1 ] && CATH_TLA+=(--tla-code "$_key=true")
    [ "$_val" = 0 ] && CATH_TLA+=(--tla-code "$_key=false")
done
unset _pr40r2 _env _key _val
# doc pr/40 round 4: two follow-on defects from round 2/3's F5 fix (relabels
# pdg but not the shower flags; a muon fathering two protons is not
# physical).  Same tri-state contract.  Both default OFF pending gates.
#   F7 SBND_SHOWER_PROTON_DAUGHTER_PION_DISSOLVE  clear shower flags when F5
#                                                  relabels a segment to pion
#   F8 SBND_MUON_MULTI_PROTON_PION                muon -> pion when its far
#                                                  end is a multi-proton
#                                                  (>=2, charge-confirmed)
#                                                  hadronic vertex
for _pr40r4 in \
    "SBND_SHOWER_PROTON_DAUGHTER_PION_DISSOLVE:shower_proton_daughter_pion_dissolve" \
    "SBND_MUON_MULTI_PROTON_PION:muon_multi_proton_pion" ; do
    _env=${_pr40r4%%:*}; _key=${_pr40r4#*:}; _val=${!_env:-}
    [ "$_val" = 1 ] && CATH_TLA+=(--tla-code "$_key=true")
    [ "$_val" = 0 ] && CATH_TLA+=(--tla-code "$_key=false")
done
unset _pr40r4 _env _key _val
# doc pr/40 round 5: muon mis-identified as electron, three owner-reported Bee
# cases (84229, 54341, 55715), three independent mechanisms.  Same tri-state
# contract.  All three default OFF pending gates.
#   F9  SBND_TRACK_PID_PERSIST_DQDX_ELECTRON_GUARD  narrows F1 -- no longer
#                                                    rescues an undirected
#                                                    electron guess
#   F10 SBND_SHOWER_CONNECT_MAIN_VERTEX_STRAIGHT_GUARD  excludes a long,
#                                                    straight candidate from
#                                                    the main-vertex EM-shower
#                                                    selection
#   F11 SBND_SHOWER_TRAJ_STRAIGHT_GUARD              same straightness veto
#                                                    on segment_is_shower_
#                                                    trajectory that F3 gave
#                                                    segment_is_shower_topology
for _pr40r5 in \
    "SBND_TRACK_PID_PERSIST_DQDX_ELECTRON_GUARD:track_pid_persist_dqdx_electron_guard" \
    "SBND_SHOWER_CONNECT_MAIN_VERTEX_STRAIGHT_GUARD:shower_connect_main_vertex_straight_guard" \
    "SBND_SHOWER_TRAJ_STRAIGHT_GUARD:shower_traj_straight_guard" ; do
    _env=${_pr40r5%%:*}; _key=${_pr40r5#*:}; _val=${!_env:-}
    [ "$_val" = 1 ] && CATH_TLA+=(--tla-code "$_key=true")
    [ "$_val" = 0 ] && CATH_TLA+=(--tla-code "$_key=false")
done
unset _pr40r5 _env _key _val
# doc pr/40 round 6: the boundary-level fixes round 5's G2 measurement
# demanded.  Same tri-state contract.  All three default OFF pending gates.
#   F12 SBND_SHOWER_ABSORB_TRACK_GUARD           shower flood-fill no longer
#                                                 absorbs a confident straight
#                                                 non-electron track
#   F13 SBND_SHOWER_CONNECT_PROTECTED_PION_GUARD connecting_to_main_vertex no
#                                                 longer force-sets a proton-
#                                                 daughter pion to electron
#   F14 SBND_MICHEL_STEM_MUON_RESCUE             Michel stopping-muon rescue
#                                                 reaches a proton-called stem
#                                                 at a multi-prong vertex
for _pr40r6 in \
    "SBND_SHOWER_ABSORB_TRACK_GUARD:shower_absorb_track_guard" \
    "SBND_SHOWER_CONNECT_PROTECTED_PION_GUARD:shower_connect_protected_pion_guard" \
    "SBND_MICHEL_STEM_MUON_RESCUE:michel_stem_muon_rescue" ; do
    _env=${_pr40r6%%:*}; _key=${_pr40r6#*:}; _val=${!_env:-}
    [ "$_val" = 1 ] && CATH_TLA+=(--tla-code "$_key=true")
    [ "$_val" = 0 ] && CATH_TLA+=(--tla-code "$_key=false")
done
unset _pr40r6 _env _key _val
# doc pr/38 Round 3 + doc pr/44 (2026-08-07): PF-tree/image consistency on
# 18255-142421.  Same tri-state contract (unset = cfg default, 1 = force on,
# 0 = force off).
#   SBND_PF_ORPHAN_TRACK_PARENTAGE   barrier-orphaned PF track nodes attach by
#                                    graph topology instead of flat roots
#                                    (DISPLAY-ONLY: moves mc.json only)
#   SBND_SHOWER_LONG_MUON_KEEP_TYPE  multi-segment long-muon pseudo-shower
#                                    keeps its muon start segment (NOT display-
#                                    only: PID/kine/pi0 pairing move)
for _pr44 in \
    "SBND_PF_ORPHAN_TRACK_PARENTAGE:pf_orphan_track_parentage" \
    "SBND_SHOWER_LONG_MUON_KEEP_TYPE:shower_long_muon_keep_type" ; do
    _env=${_pr44%%:*}; _key=${_pr44#*:}; _val=${!_env:-}
    [ "$_val" = 1 ] && CATH_TLA+=(--tla-code "$_key=true")
    [ "$_val" = 0 ] && CATH_TLA+=(--tla-code "$_key=false")
done
unset _pr44 _env _key _val
# doc pr/74 round 2: track/shower separation fixes for the four owner cases.
# Same tri-state contract (unset = cfg default, 1 = force on, 0 = force off).
#   SBND_SHOWER_IN_CASCADE_GUARD    P1: refuse examine_direction's cascade
#                                   e- relabel for a long AND MIP-like segment
#                                   (18345-53361)
#   SBND_MICHEL_STEM_MICHEL_CHECK   P2: F14 Michel rescue requires the far
#                                   subtree to be Michel-sized (18255-90055)
#   SBND_SHOWER_STEM_BACKFILL       K4: absorb the walked-past track stem
#                                   between main vertex and each substantial
#                                   EM shower (18255-90055, 18255-469665)
#   SBND_SHOWER_CONN3_UNREACHABLE   K5 = pr/65 rung 2: conn-3 pseudo-gamma
#                                   promotion of unreachable main-cluster
#                                   segments (18306-142421 seg 7013)
#   SBND_SHOWER_TRAJ_MICHEL_STEM    K6 (round 4): demote a main-vertex
#                                   shower-trajectory stem to a stopping muon
#                                   when its far end is a terminal, large-angle
#                                   Michel (18255-506746 seg 21048)
for _pr74 in \
    "SBND_SHOWER_IN_CASCADE_GUARD:shower_in_cascade_guard" \
    "SBND_MICHEL_STEM_MICHEL_CHECK:michel_stem_michel_check" \
    "SBND_SHOWER_STEM_BACKFILL:shower_stem_backfill" \
    "SBND_SHOWER_CONN3_UNREACHABLE:shower_conn3_unreachable" \
    "SBND_SHOWER_TRAJ_MICHEL_STEM:shower_traj_michel_stem" ; do
    _env=${_pr74%%:*}; _key=${_pr74#*:}; _val=${!_env:-}
    [ "$_val" = 1 ] && CATH_TLA+=(--tla-code "$_key=true")
    [ "$_val" = 0 ] && CATH_TLA+=(--tla-code "$_key=false")
done
unset _pr74 _env _key _val
# doc pr/74 round 2 scalar tunables.  EMPTY = no TLA = cfg default (null =
# the C++ default: 40 cm / 1.3 / 40 cm).  Only read when the owning bool is on.
[ -n "${SBND_SHOWER_IN_MAX_LEN:-}" ] && CATH_TLA+=(--tla-code "shower_in_max_len=${SBND_SHOWER_IN_MAX_LEN}")
[ -n "${SBND_SHOWER_IN_MIP_HI:-}" ] && CATH_TLA+=(--tla-code "shower_in_mip_hi=${SBND_SHOWER_IN_MIP_HI}")
[ -n "${SBND_MICHEL_STEM_MAX_FAR_LEN:-}" ] && CATH_TLA+=(--tla-code "michel_stem_max_far_len=${SBND_MICHEL_STEM_MAX_FAR_LEN}")
[ -n "${SBND_STEM_BACKFILL_MAX_LEN:-}" ] && CATH_TLA+=(--tla-code "stem_backfill_max_len=${SBND_STEM_BACKFILL_MAX_LEN}")
[ -n "${SBND_STEM_BACKFILL_MIP_LO:-}" ] && CATH_TLA+=(--tla-code "stem_backfill_mip_lo=${SBND_STEM_BACKFILL_MIP_LO}")
[ -n "${SBND_STEM_BACKFILL_MIP_HI:-}" ] && CATH_TLA+=(--tla-code "stem_backfill_mip_hi=${SBND_STEM_BACKFILL_MIP_HI}")
[ -n "${SBND_STEM_BACKFILL_MIN_SHOWER_LEN:-}" ] && CATH_TLA+=(--tla-code "stem_backfill_min_shower_len=${SBND_STEM_BACKFILL_MIN_SHOWER_LEN}")
[ -n "${SBND_CONN3_UNREACHABLE_MIN_LEN:-}" ] && CATH_TLA+=(--tla-code "conn3_unreachable_min_len=${SBND_CONN3_UNREACHABLE_MIN_LEN}")
# doc pr/84 round 2.  Tri-state bools (unset = cfg default, 1 = on, 0 = off)
# + scalar radii in cm (EMPTY = no TLA = cfg default).  F1/F2 are
# display-only (move mc.json only); conn3_stitch_max moves fits + PF.
for _pr84 in \
    "SBND_PF_DIRECT_WHEN_TOUCHING:pf_direct_when_touching" \
    "SBND_PF_TOUCH_CROSS_MAIN:pf_touch_cross_main" \
    "SBND_PF_PSEUDO_GAP_FROM_MAIN:pf_pseudo_gap_from_main" \
    "SBND_SHOWER_DEDUP_START_SEG:shower_dedup_start_seg" \
    "SBND_SHOWER_ENDPOINT_SKIP_ORPHAN_VTX:shower_endpoint_skip_orphan_vtx" \
    "SBND_PF_UNIQUE_NODE_IDS:pf_unique_node_ids" ; do
    _env=${_pr84%%:*}; _key=${_pr84#*:}; _val=${!_env:-}
    [ "$_val" = 1 ] && CATH_TLA+=(--tla-code "$_key=true")
    [ "$_val" = 0 ] && CATH_TLA+=(--tla-code "$_key=false")
done
unset _pr84 _env _key _val
[ -n "${SBND_PF_TOUCH_MAX:-}" ] && CATH_TLA+=(--tla-code "pf_touch_max=${SBND_PF_TOUCH_MAX}")
[ -n "${SBND_PF_TOUCH_CROSS_MAX:-}" ] && CATH_TLA+=(--tla-code "pf_touch_cross_max=${SBND_PF_TOUCH_CROSS_MAX}")
[ -n "${SBND_CONN3_STITCH_MAX:-}" ] && CATH_TLA+=(--tla-code "conn3_stitch_max=${SBND_CONN3_STITCH_MAX}")
# doc pr/74 round 4 K6 scalar tunables.  EMPTY = no TLA = cfg default (null =
# the C++ default: 15 cm / 45 cm / 1.3x / 40 cm / 40 deg).  Only read when
# shower_traj_michel_stem is on.
[ -n "${SBND_MICHEL_STEM_TRAJ_MIN_LEN:-}" ] && CATH_TLA+=(--tla-code "michel_stem_traj_min_len=${SBND_MICHEL_STEM_TRAJ_MIN_LEN}")
[ -n "${SBND_MICHEL_STEM_TRAJ_MAX_LEN:-}" ] && CATH_TLA+=(--tla-code "michel_stem_traj_max_len=${SBND_MICHEL_STEM_TRAJ_MAX_LEN}")
[ -n "${SBND_MICHEL_STEM_TRAJ_MIP_LO:-}" ] && CATH_TLA+=(--tla-code "michel_stem_traj_mip_lo=${SBND_MICHEL_STEM_TRAJ_MIP_LO}")
[ -n "${SBND_MICHEL_STEM_TRAJ_MAX_FAR_LEN:-}" ] && CATH_TLA+=(--tla-code "michel_stem_traj_max_far_len=${SBND_MICHEL_STEM_TRAJ_MAX_FAR_LEN}")
[ -n "${SBND_MICHEL_STEM_TRAJ_MIN_KINK_DEG:-}" ] && CATH_TLA+=(--tla-code "michel_stem_traj_min_kink_deg=${SBND_MICHEL_STEM_TRAJ_MIN_KINK_DEG}")
# doc pr/43 round 2 tri-state env overrides (unset = cfg default, 1 = force
# on, 0 = force off):
#   SBND_SINGLE_MUON_PROTON_CHAIN_VETO  vertex muon selection walks the
#                                       degree-2 chain for a disqualifying
#                                       proton (a muon cannot end in one)
#   SBND_SINGLE_MUON_LONG_MUON_CLAIM    long-muon chain claims the vertex
#                                       muon slot; second pdg-13 arm -> pion
#   SBND_PID_FLAG_RECONCILE             late reconciliation pass: forced-e-
#                                       terminal rescue + stale shower-flag/
#                                       wrapper cleanup on confirmed tracks
for _pr43r2 in \
    "SBND_SINGLE_MUON_PROTON_CHAIN_VETO:single_muon_proton_chain_veto" \
    "SBND_SINGLE_MUON_LONG_MUON_CLAIM:single_muon_long_muon_claim" \
    "SBND_PID_FLAG_RECONCILE:pid_flag_reconcile" ; do
    _env=${_pr43r2%%:*}; _key=${_pr43r2#*:}; _val=${!_env:-}
    [ "$_val" = 1 ] && CATH_TLA+=(--tla-code "$_key=true")
    [ "$_val" = 0 ] && CATH_TLA+=(--tla-code "$_key=false")
done
unset _pr43r2 _env _key _val
# doc pr/45 tri-state env overrides (unset = cfg default, 1 = force on,
# 0 = force off):
#   SBND_OTHER_SEG_EMPTY_2D_GUARD    find_other_segments: the -1.0 empty-2D-
#                                    tree sentinel counts as "no information"
#                                    instead of "distance zero" (isochronous
#                                    tail beyond a cathode-crossing cluster's
#                                    segment end can now seed a component)
#   SBND_PSEUDO_SHOWER_TRACK_PAINT   Bee shower_track layer + PrDisplayDump:
#                                    muon-typed (+-13) pseudo-showers paint
#                                    as track, matching the PF tree verdict
for _pr45 in \
    "SBND_OTHER_SEG_EMPTY_2D_GUARD:other_seg_empty_2d_guard" \
    "SBND_PSEUDO_SHOWER_TRACK_PAINT:pseudo_shower_track_paint" ; do
    _env=${_pr45%%:*}; _key=${_pr45#*:}; _val=${!_env:-}
    [ "$_val" = 1 ] && CATH_TLA+=(--tla-code "$_key=true")
    [ "$_val" = 0 ] && CATH_TLA+=(--tla-code "$_key=false")
done
unset _pr45 _env _key _val
# doc pr/46: long-muon stub bridge -- a short (< 6 cm) vertex stub's noisy
# fitted direction blocks the long-muon chain walk at the 15 deg junction
# test; the bridge accepts a > 35 cm MIP continuation up to 45 deg unless
# the junction has another substantial (> 10 cm) track-like arm.
#   SBND_LONG_MUON_STUB_BRIDGE       tri-state: unset = cfg default,
#                                    1 = force on, 0 = force off
for _pr46 in \
    "SBND_LONG_MUON_STUB_BRIDGE:long_muon_stub_bridge" ; do
    _env=${_pr46%%:*}; _key=${_pr46#*:}; _val=${!_env:-}
    [ "$_val" = 1 ] && CATH_TLA+=(--tla-code "$_key=true")
    [ "$_val" = 0 ] && CATH_TLA+=(--tla-code "$_key=false")
done
unset _pr46 _env _key _val
# doc pr/48: back-to-back track fixes (18255-51513/56211/57903/59335/57485;
# nu vertex mid-segment on one unbroken track, dQ/dx rising at BOTH ends).
# two_end_break = the two-end residual-range break pass;
# kink_walk_dqdx_stop / kink_break_protect = the 59335 walk-overshoot +
# EV4-absorption fixes.  Same tri-state contract: unset = cfg default,
# 1 = force on, 0 = force off.
#   SBND_TWO_END_BREAK / SBND_KINK_WALK_DQDX_STOP / SBND_KINK_BREAK_PROTECT
for _pr48 in \
    "SBND_TWO_END_BREAK:two_end_break" \
    "SBND_KINK_WALK_DQDX_STOP:kink_walk_dqdx_stop" \
    "SBND_KINK_BREAK_PROTECT:kink_break_protect" ; do
    _env=${_pr48%%:*}; _key=${_pr48#*:}; _val=${!_env:-}
    [ "$_val" = 1 ] && CATH_TLA+=(--tla-code "$_key=true")
    [ "$_val" = 0 ] && CATH_TLA+=(--tla-code "$_key=false")
done
unset _pr48 _env _key _val
# doc pr/90 round 2: two refinements of the two_end_break pass (mcp1k
# 320865/172832/61681 -- unbroken kink, wrong nu vertex).
# teb_turn_min_arm_frac: route R2's turn argmax only considers indices whose
# PCA arms can each span this fraction of teb_turn_baseline (dimensionless).
# teb_second_max: the entry gate tolerates extra >stub prongs when exactly
# one segment exceeds this cap (cm).  Numeric TLAs, unset/empty omits
# (jsonnet default null => C++ default 0 = legacy => byte-identical).
#   SBND_TEB_TURN_MIN_ARM_FRAC
#   SBND_TEB_SECOND_MAX
if [ -n "${SBND_TEB_TURN_MIN_ARM_FRAC:-}" ]; then
    CATH_TLA+=(--tla-code "teb_turn_min_arm_frac=${SBND_TEB_TURN_MIN_ARM_FRAC}")
fi
if [ -n "${SBND_TEB_SECOND_MAX:-}" ]; then
    CATH_TLA+=(--tla-code "teb_second_max=${SBND_TEB_SECOND_MAX}")
fi
# doc pr/90 round 4 (sec 9.5 D1/D3/D4): chain-topology gate admission
# (boolean TLA true/false), route R3 local-turn threshold (deg) + activity
# floor (x mip median), and the R2 bragg-veto turn (deg).  Unset/empty
# omits (jsonnet default false/null => C++ default = legacy =>
# byte-identical).
#   SBND_TEB_CHAIN_TOPOLOGY (true/false)
#   SBND_TEB_R3_TURN
#   SBND_TEB_R3_HOT
#   SBND_TEB_BRAGG_VETO_TURN
if [ -n "${SBND_TEB_CHAIN_TOPOLOGY:-}" ]; then
    CATH_TLA+=(--tla-code "teb_chain_topology=${SBND_TEB_CHAIN_TOPOLOGY}")
fi
if [ -n "${SBND_TEB_R3_TURN:-}" ]; then
    CATH_TLA+=(--tla-code "teb_r3_turn=${SBND_TEB_R3_TURN}")
fi
if [ -n "${SBND_TEB_R3_HOT:-}" ]; then
    CATH_TLA+=(--tla-code "teb_r3_hot=${SBND_TEB_R3_HOT}")
fi
if [ -n "${SBND_TEB_BRAGG_VETO_TURN:-}" ]; then
    CATH_TLA+=(--tla-code "teb_bragg_veto_turn=${SBND_TEB_BRAGG_VETO_TURN}")
fi
# doc pr/49: own-blob-coverage down-weighting in the trajectory fit's 2D
# charge association (18255-57441 V-plane projection ghost).  NUMERIC knob,
# not tri-state: unset = cfg default; any value is passed through verbatim
# (-1 = force off, 0 = force on strict, N > 0 = on with N cells tolerance).
#   SBND_FIT_BLOB_COVERAGE
if [ -n "${SBND_FIT_BLOB_COVERAGE:-}" ]; then
    CATH_TLA+=(--tla-code "fit_blob_coverage=${SBND_FIT_BLOB_COVERAGE}")
fi
# doc pr/50: suspend the pr/49 deweighting during find_proto_vertex (the
# partition-forming stage) -- 172230-class near-vertex robustness.  Boolean
# TLA: unset = cfg default (false = pr/49 behavior); true/false passed
# verbatim.
#   SBND_FIT_BLOB_COVERAGE_DEFER
if [ -n "${SBND_FIT_BLOB_COVERAGE_DEFER:-}" ]; then
    CATH_TLA+=(--tla-code "fit_blob_coverage_defer=${SBND_FIT_BLOB_COVERAGE_DEFER}")
fi
# doc pr/50: main-vertex kink-consistency snap (172230-class near-vertex
# robustness).  Boolean TLA, same contract as the defer knob above.
#   SBND_VERTEX_KINK_SNAP
if [ -n "${SBND_VERTEX_KINK_SNAP:-}" ]; then
    CATH_TLA+=(--tla-code "vertex_kink_snap=${SBND_VERTEX_KINK_SNAP}")
fi
# doc pr/51: main-vertex graph audit (near-vertex graph-shape repair --
# duplicate-corridor merge / charge-less-bridge removal / micro-stub absorb
# + re-seat / one refit).  Boolean TLA, same contract as the snap knob.
#   SBND_MAIN_VERTEX_GRAPH_AUDIT
if [ -n "${SBND_MAIN_VERTEX_GRAPH_AUDIT:-}" ]; then
    CATH_TLA+=(--tla-code "main_vertex_graph_audit=${SBND_MAIN_VERTEX_GRAPH_AUDIT}")
fi
# doc pr/51 round 3: op3 satellite-anchor radius (cm), the mvga numeric
# knob that gets its own env (the rest ride the C++ defaults via
# main_vertex_graph_audit alone).  Numeric TLA -- pass a bare cm value, e.g.
# SBND_MVGA_SATELLITE=3.0.  Unset/empty omits the override (jsonnet default
# null => byte-identical).
#   SBND_MVGA_SATELLITE
if [ -n "${SBND_MVGA_SATELLITE:-}" ]; then
    CATH_TLA+=(--tla-code "mvga_satellite=${SBND_MVGA_SATELLITE}")
fi
# doc pr/85: op3 stub-length ceiling (cm), for the Q4 sweep.  Numeric TLA,
# unset/empty omits (C++ default 2.0).
#   SBND_MVGA_STUB
if [ -n "${SBND_MVGA_STUB:-}" ]; then
    CATH_TLA+=(--tla-code "mvga_stub=${SBND_MVGA_STUB}")
fi
# doc pr/85: op1/op3 overlap-fraction gate and op3 point-degeneracy ceiling.
# The pr/85 full-sample adjudication found every adverse >1 cm mover was a
# terminal absorb at overlap=0.75 nfit=4 (admitted by the degeneracy gate)
# while every clean absorb ran at overlap=1.00 -- 0.8/3 blocks exactly that
# class.  Numeric TLAs, unset/empty omits (C++ defaults 0.7 / 4).
#   SBND_MVGA_DUP_FRAC
#   SBND_MVGA_STUB_PTS
if [ -n "${SBND_MVGA_DUP_FRAC:-}" ]; then
    CATH_TLA+=(--tla-code "mvga_dup_frac=${SBND_MVGA_DUP_FRAC}")
fi
if [ -n "${SBND_MVGA_STUB_PTS:-}" ]; then
    CATH_TLA+=(--tla-code "mvga_stub_pts=${SBND_MVGA_STUB_PTS}")
fi
# doc pr/85: op3 re-seat collinearity threshold (deg).  0 DISABLES the
# re-seat sub-case (absorb-only op3) -- the pr/85 mover adjudication found
# both label-set re-seat firings moved the vertex OFF the owner's click
# (280017: 1.64 cm, 314838: 0.60 cm).  Numeric TLA, unset/empty omits.
#   SBND_MVGA_RESEAT_ANGLE
if [ -n "${SBND_MVGA_RESEAT_ANGLE:-}" ]; then
    CATH_TLA+=(--tla-code "mvga_reseat_angle=${SBND_MVGA_RESEAT_ANGLE}")
fi
# doc pr/85: op3 interposed-stub absorb at the main-vertex anchor (boolean;
# inert unless main_vertex_graph_audit) + its far-end collinearity angle
# (deg, C++ default 150).  Same contracts as the mvga knobs above.
#   SBND_MVGA_INTERPOSED
#   SBND_MVGA_INTERPOSED_ANGLE
if [ -n "${SBND_MVGA_INTERPOSED:-}" ]; then
    CATH_TLA+=(--tla-code "mvga_interposed=${SBND_MVGA_INTERPOSED}")
fi
if [ -n "${SBND_MVGA_INTERPOSED_ANGLE:-}" ]; then
    CATH_TLA+=(--tla-code "mvga_interposed_angle=${SBND_MVGA_INTERPOSED_ANGLE}")
fi
# doc pr/86: interposed-splice candidate ceiling, cm (C++ default 0 = use
# mvga_stub; widens only the splice, never the terminal absorb).  Numeric
# TLA, unset/empty omits (jsonnet default null => byte-identical).
#   SBND_MVGA_INTERPOSED_LEN
if [ -n "${SBND_MVGA_INTERPOSED_LEN:-}" ]; then
    CATH_TLA+=(--tla-code "mvga_interposed_len=${SBND_MVGA_INTERPOSED_LEN}")
fi
# doc pr/86 P4: satellite-anchor op3 overlap threshold (fraction; C++
# default 0 = use mvga_dup_frac everywhere).  Numeric TLA, unset/empty
# omits (jsonnet default null => byte-identical).
#   SBND_MVGA_SAT_DUP_FRAC
if [ -n "${SBND_MVGA_SAT_DUP_FRAC:-}" ]; then
    CATH_TLA+=(--tla-code "mvga_sat_dup_frac=${SBND_MVGA_SAT_DUP_FRAC}")
fi
# doc pr/86 P1b: interposed splice at degree-1 main anchors (boolean; C++
# default false).  Boolean TLA, unset/empty omits (jsonnet default false
# => key suppressed => byte-identical).
#   SBND_MVGA_INTERPOSED_DEG1
if [ -n "${SBND_MVGA_INTERPOSED_DEG1:-}" ]; then
    CATH_TLA+=(--tla-code "mvga_interposed_deg1=${SBND_MVGA_INTERPOSED_DEG1}")
fi
# doc pr/86 round 2 R1: op3 post-carry straighten reach past the junction
# (cm; C++ default 0 = concatenation verbatim).  Numeric TLA, unset/empty
# omits (jsonnet default null => byte-identical).
#   SBND_MVGA_SPLICE_STRAIGHTEN
if [ -n "${SBND_MVGA_SPLICE_STRAIGHTEN:-}" ]; then
    CATH_TLA+=(--tla-code "mvga_splice_straighten=${SBND_MVGA_SPLICE_STRAIGHTEN}")
fi
# doc pr/86 round 2 R2: op3.5 junction-collapse radius around the main
# vertex (cm; C++ default 0 = pass skipped).  Numeric TLA, unset/empty
# omits (jsonnet default null => byte-identical).
#   SBND_MVGA_APPROACH_COLLAPSE
if [ -n "${SBND_MVGA_APPROACH_COLLAPSE:-}" ]; then
    CATH_TLA+=(--tla-code "mvga_approach_collapse=${SBND_MVGA_APPROACH_COLLAPSE}")
fi
# doc pr/86 round 2: R1/R2 straight-chain charge-veto radius (cm; C++
# default 0 = the prototype 0.2 cm).  Numeric TLA, unset/empty omits
# (jsonnet default null => byte-identical).
#   SBND_MVGA_STRAIGHTEN_RADIUS
if [ -n "${SBND_MVGA_STRAIGHTEN_RADIUS:-}" ]; then
    CATH_TLA+=(--tla-code "mvga_straighten_radius=${SBND_MVGA_STRAIGHTEN_RADIUS}")
fi
# doc pr/83 r3: the duplicate-corridor round.  op1-only scope radius (cm;
# C++ default 0 = use mvga_radius, -1 = unscoped) and overlap threshold
# (fraction; 0 = use mvga_dup_frac); post-op3 dup pass incl. created
# segments (boolean, class A); interposed-carry prong ceiling (count; 0 =
# unlimited); abandoned-main-cluster dup audit inside swap_main_cluster
# (boolean, Mechanism C).  Numeric/boolean TLAs, unset/empty omits
# (jsonnet defaults null/false => keys suppressed => byte-identical).
#   SBND_MVGA_OP1_RADIUS
#   SBND_MVGA_OP1_DUP_FRAC
#   SBND_MVGA_OP1_POST
#   SBND_MVGA_CARRY_MAX
#   SBND_SWAP_ORPHAN_DUP_AUDIT
if [ -n "${SBND_MVGA_OP1_RADIUS:-}" ]; then
    CATH_TLA+=(--tla-code "mvga_op1_radius=${SBND_MVGA_OP1_RADIUS}")
fi
if [ -n "${SBND_MVGA_OP1_DUP_FRAC:-}" ]; then
    CATH_TLA+=(--tla-code "mvga_op1_dup_frac=${SBND_MVGA_OP1_DUP_FRAC}")
fi
if [ -n "${SBND_MVGA_OP1_POST:-}" ]; then
    CATH_TLA+=(--tla-code "mvga_op1_post=${SBND_MVGA_OP1_POST}")
fi
if [ -n "${SBND_MVGA_CARRY_MAX:-}" ]; then
    CATH_TLA+=(--tla-code "mvga_carry_max=${SBND_MVGA_CARRY_MAX}")
fi
if [ -n "${SBND_SWAP_ORPHAN_DUP_AUDIT:-}" ]; then
    CATH_TLA+=(--tla-code "swap_orphan_dup_audit=${SBND_SWAP_ORPHAN_DUP_AUDIT}")
fi
# doc pr/83 r4 -- projective duplicate collapse (see wct-pr-perevt.jsonnet)
#   SBND_MVGA_PROJ_DUP_FRAC
#   SBND_MVGA_PROJ_DQDX_RATIO
if [ -n "${SBND_MVGA_PROJ_DUP_FRAC:-}" ]; then
    CATH_TLA+=(--tla-code "mvga_proj_dup_frac=${SBND_MVGA_PROJ_DUP_FRAC}")
fi
if [ -n "${SBND_MVGA_PROJ_DQDX_RATIO:-}" ]; then
    CATH_TLA+=(--tla-code "mvga_proj_dqdx_ratio=${SBND_MVGA_PROJ_DQDX_RATIO}")
fi
if [ -n "${SBND_MVGA_PROJ_ANGLE:-}" ]; then
    CATH_TLA+=(--tla-code "mvga_proj_angle=${SBND_MVGA_PROJ_ANGLE}")
fi
# doc pr/85: carry the old vertex's arms through the snap residual below
# this arc (cm).  Numeric TLA -- pass a bare cm value, e.g.
# SBND_VKS_CARRY_PRONG=1.5.  Unset/empty omits the override (jsonnet
# default null => byte-identical).
#   SBND_VKS_CARRY_PRONG
if [ -n "${SBND_VKS_CARRY_PRONG:-}" ]; then
    CATH_TLA+=(--tla-code "vks_carry_prong=${SBND_VKS_CARRY_PRONG}")
fi
# doc pr/51 (18255-506746): DL rerank cross-cluster swap guard.  Boolean
# TLA, same contract.
#   SBND_DL_VTX_SWAP_GUARD
if [ -n "${SBND_DL_VTX_SWAP_GUARD:-}" ]; then
    CATH_TLA+=(--tla-code "dl_vtx_swap_guard=${SBND_DL_VTX_SWAP_GUARD}")
fi
# doc sbnd_xin/docs/pr/75: per-event vertex scoreboard (recording only, read
# by PrDisplayDump for the neutrino-vertex hand scan).  Boolean TLA, same
# contract.  Defaulted to true above when PR_EXTRA_STAGES names pr_display.
#   SBND_VERTEX_SCOREBOARD
if [ -n "${SBND_VERTEX_SCOREBOARD:-}" ]; then
    CATH_TLA+=(--tla-code "vertex_scoreboard=${SBND_VERTEX_SCOREBOARD}")
fi
# doc sbnd_xin/docs/pr/79 sec 10: live-feature harvest (SCN input cloud +
# discarded traditional-path features into the scoreboard dump).  Boolean
# TLA, same contract.  Requires the scoreboard; a truthy value defaults
# SBND_VERTEX_SCOREBOARD=true near the PR_EXTRA_STAGES block above.
#   SBND_DL_VTX_HARVEST
if [ -n "${SBND_DL_VTX_HARVEST:-}" ]; then
    CATH_TLA+=(--tla-code "dl_vtx_harvest=${SBND_DL_VTX_HARVEST}")
fi
# doc pr/51 round 3: apply the traditional main-vertex path's cluster-swap
# decision instead of discarding it.  Boolean TLA, same contract.
#   SBND_MAIN_VERTEX_SWAP_APPLY
if [ -n "${SBND_MAIN_VERTEX_SWAP_APPLY:-}" ]; then
    CATH_TLA+=(--tla-code "main_vertex_swap_apply=${SBND_MAIN_VERTEX_SWAP_APPLY}")
fi
# doc pr/51 round 4: diagnostic-only rough-path probe (near-vertex short-cut
# investigation, path-COST not graph-shape).  Boolean TLA, same contract.
# No graph/fit/segment content is ever changed; every line is TRACE.
#   SBND_ROUGH_PATH_PROBE
if [ -n "${SBND_ROUGH_PATH_PROBE:-}" ]; then
    CATH_TLA+=(--tla-code "rough_path_probe=${SBND_ROUGH_PATH_PROBE}")
fi
# doc pr/51 round 5: steiner gap penalty (the H1 short-cut fix).  Numeric
# TLAs, bare values (scale is dimensionless; the four sub-knobs are cm /
# fractions and ride the C++ defaults 0.25 / 0.5 / 0.3 / 0.2 unless set).
#   SBND_STEINER_GAP_PENALTY  (0 = off; validation scales 1-5)
if [ -n "${SBND_STEINER_GAP_PENALTY:-}" ]; then
    CATH_TLA+=(--tla-code "steiner_gap_penalty=${SBND_STEINER_GAP_PENALTY}")
fi
if [ -n "${SBND_SGP_DEAD_ALPHA:-}" ]; then
    CATH_TLA+=(--tla-code "sgp_dead_alpha=${SBND_SGP_DEAD_ALPHA}")
fi
if [ -n "${SBND_SGP_MIN_EDGE:-}" ]; then
    CATH_TLA+=(--tla-code "sgp_min_edge=${SBND_SGP_MIN_EDGE}")
fi
if [ -n "${SBND_SGP_SAMPLE_STEP:-}" ]; then
    CATH_TLA+=(--tla-code "sgp_sample_step=${SBND_SGP_SAMPLE_STEP}")
fi
if [ -n "${SBND_SGP_POINT_RADIUS:-}" ]; then
    CATH_TLA+=(--tla-code "sgp_point_radius=${SBND_SGP_POINT_RADIUS}")
fi
# doc pr/51 round 6: weak-charge deficit term on the gap flavor.
#   SBND_SGP_WEAK_SCALE  (0 = off; grid scales 1-5)
#   SBND_SGP_WEAK_QREF   (charge units; C++ default 2000)
if [ -n "${SBND_SGP_WEAK_SCALE:-}" ]; then
    CATH_TLA+=(--tla-code "sgp_weak_scale=${SBND_SGP_WEAK_SCALE}")
fi
if [ -n "${SBND_SGP_WEAK_QREF:-}" ]; then
    CATH_TLA+=(--tla-code "sgp_weak_qref=${SBND_SGP_WEAK_QREF}")
fi
# doc pr/73: per-edge DEBUG sentinel for the steiner_graph_gap scan.
#   SBND_SGP_EDGE_PROBE  (literal jsonnet "true"/"false"; default off)
# Log-only; emits one "sgp edge:" line per SCANNED edge.  Diagnostic runs only
# -- it does not change reconstruction, but it does add ~1k lines per cluster.
if [ -n "${SBND_SGP_EDGE_PROBE:-}" ]; then
    CATH_TLA+=(--tla-code "sgp_edge_probe=${SBND_SGP_EDGE_PROBE}")
fi
# doc pr/73 round 2, fix F3a: cap (cm) on how far the round-6 penalized route
# may stray from the unpenalized one in do_rough_path; over the cap the BASE
# route is kept.  C++ default -1 = off (unbounded, legacy).
#   SBND_SGP_MAX_SEP  (cm; -1 = off.  NB 0 is a real cap -- reject any
#                      excursion -- so "off" is a NEGATIVE value, not 0.)
if [ -n "${SBND_SGP_MAX_SEP:-}" ]; then
    CATH_TLA+=(--tla-code "sgp_max_sep=${SBND_SGP_MAX_SEP}")
fi
# doc pr/83: orient break_segment() splits to the wcpt path (find_vertices)
# instead of boost source/target; a reversed edge otherwise yields crossed
# children with vertex fits on the wrong track ends and stacked duplicate
# trajectories (mcp1k 283040/59899/72586).  C++ default false.
#   SBND_BREAK_SEG_ORIENT  (literal jsonnet "true"/"false")
if [ -n "${SBND_BREAK_SEG_ORIENT:-}" ]; then
    CATH_TLA+=(--tla-code "break_seg_orient=${SBND_BREAK_SEG_ORIENT}")
fi
# doc pr/51 round 7: robust vertex fit (mvfit_robust + satellites).
#   SBND_MVFIT_ROBUST      (true/false; escape for A/B: false)
#   SBND_MVFIT_MAIN_ONLY   (true/false; C++ default true)
#   SBND_MVFIT_MIN_LEN / RIN_MARGIN / ROUT_MIN / ROUT_MAX / PRIOR_RANGE (cm)
#   SBND_MVFIT_ROUT_FRAC / ANGLE (deg) / MIN_PTS / MIN_ANISO
for _pr51r7 in "SBND_MVFIT_ROBUST:mvfit_robust" \
               "SBND_MVFIT_MAIN_ONLY:mvfit_main_only" \
               "SBND_MVFIT_MIN_LEN:mvfit_min_len" \
               "SBND_MVFIT_RIN_MARGIN:mvfit_rin_margin" \
               "SBND_MVFIT_ROUT_FRAC:mvfit_rout_frac" \
               "SBND_MVFIT_ROUT_MIN:mvfit_rout_min" \
               "SBND_MVFIT_ROUT_MAX:mvfit_rout_max" \
               "SBND_MVFIT_ANGLE:mvfit_angle" \
               "SBND_MVFIT_MIN_PTS:mvfit_min_pts" \
               "SBND_MVFIT_MIN_ANISO:mvfit_min_aniso" \
               "SBND_MVFIT_PRIOR_RANGE:mvfit_prior_range"; do
    _env=${_pr51r7%%:*}; _key=${_pr51r7#*:}; _val=${!_env:-}
    if [ -n "$_val" ]; then
        CATH_TLA+=(--tla-code "$_key=$_val")
    fi
done
# doc pr/54: keep well-supported isolated residual segments in
# find_other_segments (18255-142421 separated EM shower with no fitted
# trajectory).  Boolean TLA, same contract; the two numeric floors ride the
# C++ defaults (25 points / 3 cm) unless their own envs are set (bare
# values: integer count, cm).
#   SBND_OTHER_SEG_KEEP_ISOLATED
if [ -n "${SBND_OTHER_SEG_KEEP_ISOLATED:-}" ]; then
    CATH_TLA+=(--tla-code "other_seg_keep_isolated=${SBND_OTHER_SEG_KEEP_ISOLATED}")
fi
if [ -n "${SBND_OTHER_SEG_KEEP_ISOLATED_MIN_POINTS:-}" ]; then
    CATH_TLA+=(--tla-code "other_seg_keep_isolated_min_points=${SBND_OTHER_SEG_KEEP_ISOLATED_MIN_POINTS}")
fi
if [ -n "${SBND_OTHER_SEG_KEEP_ISOLATED_MIN_LENGTH:-}" ]; then
    CATH_TLA+=(--tla-code "other_seg_keep_isolated_min_length=${SBND_OTHER_SEG_KEEP_ISOLATED_MIN_LENGTH}")
fi
# doc pr/67 round 3 (S2): size gate on the isochronous snap in
# find_other_segments -- the machinery that ATTACHES a short isochronously
# displaced branch to its parent.  Bare value in cm; unset means the C++
# default 10.0 (legacy), which is byte-identical to pre-pr/67.  The doc pr/67
# branches sit at dir_mag 4.3-4.7 cm, so a value near 4 admits them.
#   SBND_ISO_SNAP_MIN_DIR_MAG (cm)
if [ -n "${SBND_ISO_SNAP_MIN_DIR_MAG:-}" ]; then
    CATH_TLA+=(--tla-code "iso_snap_min_dir_mag=${SBND_ISO_SNAP_MIN_DIR_MAG}")
fi
# doc pr/59 round 2 (18255-142421 seg 20; 116944-71372 segs
# 19052/19053/136199): per-cluster re-association rescue for a segment
# created after its cluster's clustering_points pass, left with a null
# associate_points cloud (zero shower_track-global points, energy fallback to
# the sparse fit polyline).  Boolean TLA, same contract.
#   SBND_ASSOC_FULL_RECLUSTER
if [ -n "${SBND_ASSOC_FULL_RECLUSTER:-}" ]; then
    CATH_TLA+=(--tla-code "assoc_full_recluster=${SBND_ASSOC_FULL_RECLUSTER}")
fi
# doc pr/64 round 7 (18259-18625: 12-18 pt blob at PF segment 126042's own
# fit endpoint, present in img charge but absent from
# shower_track/associate_points): reassign, instead of drop, a point that
# loses (or never enters) clustering_points_segments' Stage-C ghost removal
# to a SAME-cluster segment that actually wins the global 2D projection
# contest.  Boolean TLA, same contract.
#   SBND_ASSOC_REASSIGN_ORPHANS
if [ -n "${SBND_ASSOC_REASSIGN_ORPHANS:-}" ]; then
    CATH_TLA+=(--tla-code "assoc_reassign_orphans=${SBND_ASSOC_REASSIGN_ORPHANS}")
fi
# doc pr/64 round 8 (same 18259-18625 blob: examine_structure_final_1/_1p/_3,
# called unconditionally inside determine_main_vertex, merge a short/
# duplicate/degenerate segment into a surviving neighbor without carrying its
# associate_points forward).  When the deleted segment had non-empty
# associate_points, clears the survivor's too, so pr/59's
# reassociate_cluster_orphans any_orphan trigger correctly re-fires.  Boolean
# TLA, same contract.
#   SBND_ASSOC_CLEAR_ON_MERGE
if [ -n "${SBND_ASSOC_CLEAR_ON_MERGE:-}" ]; then
    CATH_TLA+=(--tla-code "assoc_clear_on_merge=${SBND_ASSOC_CLEAR_ON_MERGE}")
fi
# doc pr/72 round 2: examine_structure_3 stub guard (18255-196649).  Bool
# TLA, same contract as SBND_ASSOC_CLEAR_ON_MERGE (pass-through, must be the
# literal jsonnet string "true"/"false").  Numeric/bool sub-params are bare
# value pass-throughs -- unset means the component keeps its own C++
# default (fitted from a 117-event census, see doc pr/72 round 2).
#   SBND_ES3_STUB_GUARD
if [ -n "${SBND_ES3_STUB_GUARD:-}" ]; then
    CATH_TLA+=(--tla-code "es3_stub_guard=${SBND_ES3_STUB_GUARD}")
fi
#   SBND_ES3SG_STUB_MAX (cm)
[ -n "${SBND_ES3SG_STUB_MAX:-}" ] && \
    CATH_TLA+=(--tla-code "es3sg_stub_max=${SBND_ES3SG_STUB_MAX}")
#   SBND_ES3SG_LEN_RATIO
[ -n "${SBND_ES3SG_LEN_RATIO:-}" ] && \
    CATH_TLA+=(--tla-code "es3sg_len_ratio=${SBND_ES3SG_LEN_RATIO}")
#   SBND_ES3SG_ANG3_MIN (degrees)
[ -n "${SBND_ES3SG_ANG3_MIN:-}" ] && \
    CATH_TLA+=(--tla-code "es3sg_ang3_min=${SBND_ES3SG_ANG3_MIN}")
#   SBND_ES3SG_ANG_RATIO
[ -n "${SBND_ES3SG_ANG_RATIO:-}" ] && \
    CATH_TLA+=(--tla-code "es3sg_ang_ratio=${SBND_ES3SG_ANG_RATIO}")
#   SBND_ES3SG_REQUIRE_TERMINAL (literal jsonnet "true"/"false")
[ -n "${SBND_ES3SG_REQUIRE_TERMINAL:-}" ] && \
    CATH_TLA+=(--tla-code "es3sg_require_terminal=${SBND_ES3SG_REQUIRE_TERMINAL}")
# doc pr/65 round 3 (18259-54095 PF-root orphan mu-/e- fragments): rung 1
# relaxes the shower absorbers' cluster()==main_cluster guards to a
# main_vertex-reachability test so kept-isolated pr/54 residuals can be
# absorbed; rung 3 replaces the orphan net's fabricated root-level PF node
# with an audit log line (display-only, moves ONLY mc.json).  Boolean TLAs,
# value passed verbatim (true/false), same contract as the pr/54 hook above.
#   SBND_SHOWER_ABSORB_UNREACHABLE_MAIN
#   SBND_PF_ORPHAN_AUDIT_ONLY
if [ -n "${SBND_SHOWER_ABSORB_UNREACHABLE_MAIN:-}" ]; then
    CATH_TLA+=(--tla-code "shower_absorb_unreachable_main=${SBND_SHOWER_ABSORB_UNREACHABLE_MAIN}")
fi
if [ -n "${SBND_PF_ORPHAN_AUDIT_ONLY:-}" ]; then
    CATH_TLA+=(--tla-code "pf_orphan_audit_only=${SBND_PF_ORPHAN_AUDIT_ONLY}")
fi
# doc pr/43: four owner-reported PID cases on run 18255 (142421, 54351,
# 56463, 57661) -- one segment absorbed and missing from both the flow tree
# and the energy sum, and three muon/pion mis-selections at the
# single-muon-per-cluster rule and the topology shower test.  Same
# tri-state contract.  All default OFF pending gates.
#   F1  SBND_MUON_CHAIN_PROTON_VETO         multi-hop generalization of the
#                                            muon-candidate loop's 1-hop
#                                            proton veto
#       SBND_SHOWER_TYPE_CACHE_REFRESH      keep Shower's cached
#                                            particle_type in lock-step with
#                                            update_particle_type's own
#                                            segment relabel
#       SBND_SHOWER_TRAJ_DQDX_GUARD         trust a confident non-electron
#                                            track-PID conclusion inside
#                                            segment_determine_shower_
#                                            direction_trajectory instead
#                                            of discarding it
#       SBND_SHOWER_TRAJ_CHAIN_PION         companion to the above: relabel
#                                            a main-vertex proton's short
#                                            muon-pdg continuation chain to
#                                            pion except the deepest,
#                                            confirmed-muon segment
#       SBND_KINE_SHOWER_VERTEX_BARRIER     kine-tree parity with pr/38's
#                                            pf_shower_vertex_barrier;
#                                            MOVES kine_reco_Enu when on
for _pr43 in \
    "SBND_MUON_CHAIN_PROTON_VETO:muon_chain_proton_veto" \
    "SBND_SHOWER_TYPE_CACHE_REFRESH:shower_type_cache_refresh" \
    "SBND_SHOWER_TRAJ_DQDX_GUARD:shower_traj_dqdx_guard" \
    "SBND_SHOWER_TRAJ_CHAIN_PION:shower_traj_chain_pion" \
    "SBND_KINE_SHOWER_VERTEX_BARRIER:kine_shower_vertex_barrier" ; do
    _env=${_pr43%%:*}; _key=${_pr43#*:}; _val=${!_env:-}
    [ "$_val" = 1 ] && CATH_TLA+=(--tla-code "$_key=true")
    [ "$_val" = 0 ] && CATH_TLA+=(--tla-code "$_key=false")
done
unset _pr43 _env _key _val
# DL (SCN) neutrino-vertex weights (doc pr/24 attribution arms).  UNSET = emit
# no TLA = the cfg default = the SBND operating point (DL vertex ON, doc pr/4).
# SBND_DL_WEIGHTS='' selects the geometric vertex -- the arm that isolates a
# DL-vertex effect from a PR-structure one.  Set-but-empty is honoured, hence
# the ${VAR+x} test rather than ${VAR:-}.
[ -n "${SBND_DL_WEIGHTS+x}" ] && CATH_TLA+=(--tla-str "dl_weights=${SBND_DL_WEIGHTS}")
# DL re-rank operating point (doc pr/79 deployment A/B).  UNSET = no TLA = the
# cfg defaults (min_accept 4.0, top_k 5, pinned in sbnd/clus.jsonnet since
# 2026-07-30).  Numeric values, passed as --tla-code.
[ -n "${SBND_DL_VTX_MIN_ACCEPT:-}" ] && CATH_TLA+=(--tla-code "dl_vtx_min_accept_score=${SBND_DL_VTX_MIN_ACCEPT}")
[ -n "${SBND_DL_VTX_TOP_K:-}" ] && CATH_TLA+=(--tla-code "dl_vtx_top_k=${SBND_DL_VTX_TOP_K}")
# DL main-cluster swap guard (doc pr/24).  EMPTY = no TLA = the cfg default
# null = C++ 0/0 = OFF = the legacy DL vertex.  _MIN_LEN is in CM (the jsonnet
# multiplies wc.cm); _MIN_FRAC is a bare fraction of the incumbent main
# cluster's total track length.
true
true

# The embedded interpreter needs libpython loaded RTLD_GLOBAL for the SCN
# (DL vertex) import to succeed -- same idiom as run_pr3_evt_dl.sh / M4.
PYLIB=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.11.so.1.0
[ -r "$PYLIB" ] || { echo "ERROR: libpython not found: $PYLIB" >&2; exit 1; }

process_event() {
    local EVT_ID=$1
    local QLDIR="$QLROOT/ql_evt${EVT_ID}"
    local PCT="$QLDIR/pctree-evt${EVT_ID}.tar.gz"
    local PRDIR="$OUTROOT/pr_evt${EVT_ID}"
    local LOG="$PRDIR/wct_pr_evt${EVT_ID}.log"

    [ -s "$PCT" ] || { echo "ERROR: [evt $EVT_ID] no pctree: $PCT" >&2; return 1; }

    rm -rf "$PRDIR"; mkdir -p "$PRDIR"

    # RSE from the Q/L job's own opflash metadata (same source run_nusel_evt.sh
    # uses; correct across every sample this driver targets -- verified for
    # data/MCP2025C, nueCC48, and both MC roots, doc pr/11 sec 1).
    local RUN_NO=0 SUBRUN_NO=0 _md
    _md=$(tar xzOf "$QLDIR/opflash_apa0.tar.gz" "opflash_tensorset_${EVT_ID}_metadata.json" 2>/dev/null) || _md=''
    if [ -n "$_md" ]; then
        local _rse
        _rse=$(printf '%s' "$_md" | python3 -c \
            'import json,sys; d=json.load(sys.stdin); print(int(d.get("run",0)), int(d.get("subrun",0)))' \
            2>/dev/null) && [ -n "$_rse" ] && read -r RUN_NO SUBRUN_NO <<< "$_rse"
    fi

    echo "[evt $EVT_ID] rse=($RUN_NO, $SUBRUN_NO, $EVT_ID) pipeline=($PIPELINE) reality=$REALITY dl=on"

    (
        cd "$PRDIR" || exit 1
        export LD_PRELOAD="$PYLIB"
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
        # doc pr/57: env-gated per-edge JSONL dump feeding
        # overclustering_display (hand-scan of S6 2D-connectivity
        # removals). PR_OC56_SCAN_DUMP=1 => WCT_OC56_SCAN_DUMP points at a
        # per-event file under this event's own PRDIR; unset (default) =>
        # unset, so the C++ side's Oc56DumpWriter stays disabled and every
        # other batch invocation is unaffected -- same pattern as
        # PR_EXTRA_STAGES / SBND_PROTECT_GRAPH above. Exported only inside
        # this subshell, so parallel siblings each get their own event id.
        if [ "${PR_OC56_SCAN_DUMP:-}" = "1" ]; then
            export WCT_OC56_SCAN_DUMP="$PRDIR/oc56scan-evt${EVT_ID}.jsonl"
        fi
        # PR_TIMEOUT (seconds, default 3600) bounds a single event.  A pattern-
        # recognition hang is real: SBND MCP2025C evt 352365 spun at 100% CPU with
        # byte-flat RSS for 8h17m in shower_clustering_with_nv_from_vertices and
        # stalled the whole 1000-event batch on its last slot (doc pr/11 sec 6).
        # `timeout` sits INSIDE timecmd.py so .time.meta is still written (rc=124).
        setarch x86_64 -R python3 "$AB/timecmd.py" "$PRDIR/.time.meta" \
        timeout --signal=TERM --kill-after=60 "${PR_TIMEOUT:-3600}" \
        wire-cell \
            -l stderr -l "${LOG}:${SBND_WCT_LOGLEVEL:-debug}" -L "${SBND_WCT_LOGLEVEL:-debug}" \
            --tla-str  "input=$PCT" \
            --tla-code "anode_indices=[0,1]" \
            --tla-str  "output_dir=$PRDIR" \
            --tla-code "run=${RUN_NO}" --tla-code "subrun=${SUBRUN_NO}" --tla-code "event=${EVT_ID}" \
            --tla-str  "reality=$REALITY" \
            `# doc 68: the LAr set, the beam window and every tgm_*/stm_* knob` \
            `# spelled out here were byte-for-byte the job's own defaults, so` \
            `# they are gone.  PIPELINE stays explicit -- this chain adds the` \
            `# neutrino taggers + BDT scorers on top of the default list.` \
            --tla-code "pipeline_names=[$(echo "$PIPELINE" | sed "s/[^,]\+/'&'/g")]" \
            "${TFJSON_TLA[@]}" \
            "${CATH_TLA[@]}" \
            --tla-str  "save_tensors=$PRDIR/pctree-pr-evt${EVT_ID}.tar.gz" \
            -c "$JSONNET"
        echo "rc=$?" > "$PRDIR/rc.txt"
    ) > "$PRDIR/stdout.log" 2>&1
    rm -f "$PRDIR/trash-pr.tar.gz"

    local rc; rc=$(sed -n 's/^rc=//p' "$PRDIR/rc.txt" 2>/dev/null); rc=${rc:-1}

    # DL-engagement proof (WARN visible at debug level -- doc pr/11 sec 1):
    # absent = DL ran without a hard failure this event (it may still have
    # legitimately deferred to the traditional vertex on a low rerank score).
    if grep -q "DL vertex failed" "$LOG" 2>/dev/null; then
        echo "[evt $EVT_ID] WARN: DL vertex failed this event (see $LOG)" >&2
    fi

    # Only extract when wire-cell actually succeeded.  On a crash the zip/prtree
    # are zero-byte and nusel_extract.py dies with a BadZipFile traceback that
    # buries the real cause at the tail of stdout.log -- exactly what happened to
    # the 73 doc-pr/11 failures, where SIGABRT/SIGTERM all read as "rc=250
    # BadZipFile".  rc here is wire-cell's own (timecmd.py encodes a fatal signal
    # N as 256-N: 250 = SIGABRT, 241 = SIGTERM, 124 = PR_TIMEOUT).
    if [ "$rc" != 0 ]; then
        echo "[evt $EVT_ID] wire-cell rc=$rc -- skipping nusel_extract (no usable outputs)" \
            >> "$PRDIR/stdout.log"
        echo "[evt $EVT_ID] rc=$rc  -> $PRDIR"
        return 1
    fi

    # Per-bundle label table -- same nusel_extract.py production call
    # (unmodified script; --prtree uses the save_tensors dump above, so labels
    # come from the authoritative flag_TGM/STM/FC, not the log fallback).
    python3 "$SX/nusel_extract.py" \
        --pctree "$PCT" --prbee "$PRDIR/mabc-pr.zip" --prlog "$LOG" \
        --prtree "$PRDIR/pctree-pr-evt${EVT_ID}.tar.gz" \
        --qlbee "$QLDIR/mabc-all-apa.zip" \
        --beam-window "0.2,2.2" \
        --run "$RUN_NO" --subrun "$SUBRUN_NO" \
        --out "$PRDIR/nusel-evt${EVT_ID}.tsv" 2>>"$PRDIR/stdout.log"

    echo "[evt $EVT_ID] rc=$rc  -> $PRDIR"
    [ "$rc" = 0 ]
}

batch_init
BATCH_MAX=${PR_JOBS:-6}
echo "ql_root=$QLROOT out_root=$OUTROOT reality=$REALITY events=${#EVENT_IDS[@]} jobs=$BATCH_MAX"
for evt in "${EVENT_IDS[@]}"; do
    _blog="$OUTROOT/.batch_pr_evt${evt}.log"
    batch_wait_slot
    ( process_event "$evt" ) > "$_blog" 2>&1 &
    BATCH_PIDS[$!]=$evt
    echo "  [start] evt=$evt  log: $_blog"
done
batch_drain
batch_summary

# Merge the per-event nusel tables (label/TGM/STM/FC/LM census).
_tsvs=()
for evt in "${EVENT_IDS[@]}"; do
    _t="$OUTROOT/pr_evt${evt}/nusel-evt${evt}.tsv"
    [ -s "$_t" ] && _tsvs+=("$_t")
done
if [ "${#_tsvs[@]}" -gt 0 ]; then
    python3 "$SX/nusel_extract.py" --merge "${_tsvs[@]}" \
        --out "$OUTROOT/nusel-table.tsv" \
        --events-out "$OUTROOT/nusel-events.tsv"
    echo "merged -> $OUTROOT/nusel-table.tsv + nusel-events.tsv"
fi

echo "loadavg: $(cat /proc/loadavg)"
