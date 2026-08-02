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
# Honors SBND_WORK_ROOT / SBND_INPUT_DIR / SBND_SAMPLE like run_ql_evt.sh
# (also SBND_WCT_LOGLEVEL=trace for the STM discriminant TRACE lines), so
# it runs directly on reprocessing trees, e.g. the MCP2025C reco1 sample:
#   SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
#   SBND_WORK_ROOT=$PWD/work-mcp10 ./run_nusel_evt.sh data all
#
# Prerequisite: ./run_img_evt.sh <mode> <idx> (per-event imaging npz).
# Output: work/nusel_evt<ID>/{wct_nusel_evt<ID>.log, mabc-pr.zip, nusel-evt<ID>.tsv}
#         'all' also merges: work/nusel-table.tsv (per bundle) and
#         work/nusel-events.tsv (per event).

set -e

# pwd -P: sbnd_xin is reachable through a symlink (toolkit/sbnd_xin), so
# canonicalize to the REAL location (wcp-porting-img/sbnd/sbnd_xin) before
# building paths from it.  (Historically this was REQUIRED because the PR
# jsonnet imported '../particle_dataset.jsonnet' relatively; since the 2026-07-27
# config sync those tables live in-tree as
# pgrapher/experiment/sbnd/particle_dataset.jsonnet and resolve through
# WIRECELL_PATH from anywhere -- see sbnd_xin/docs/64_cfg-sync.md.  Kept because
# $SBND_DIR is still used for data paths and the track-fitting JSON.)
SBND_DIR=$(cd "$(dirname "$0")" && pwd -P)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WCT_BASE}/wire-cell-data/sbnd/photodet:${WIRECELL_PATH}

. "$SBND_DIR/_runlib.sh"

JSONNET="$SBND_DIR/wct-pr-perevt.jsonnet"
# NOTE (doc 68): the SBND production operating point is NOT in this script.  It
# lives in the job's TLA defaults, cfg/pgrapher/experiment/sbnd/
# wct-pr-perevt.jsonnet -- the LAr DL/DT/lifetime/driftSpeed set, the beam
# window, every tgm_* and stm_* knob, mip_dqdx, unmerge_bundle_mode, the tagger
# pipeline, and the TrackFitting parameter file (WIRECELL_PATH-resolved since
# doc 64 sec 4a).  This runner passes only what is per-event plus explicit
# OVERRIDES.  Do not reintroduce a default value here.
#
# The TrackFitting parameter file is the one knob that is NOT inert: TaggerCheckSTM
# and the PR vertex fitter read DL/DT from it at RUNTIME (it never enters the
# compiled jsonnet), so it is the single live consumer of the diffusion constants
# in the data chain.  SBND_TRACKFIT_JSON overrides it so an A/B can run both
# diffusion arms from one binary in either order -- see docs/66 sec 2.
TFJSON="${SBND_TRACKFIT_JSON:-}"
# The tagger pipeline is the job's default:
#   switch_scope, unmerge_bundle, unmerge_assoc, steiner, fiducialutils,
#   tagger_check_tgm, tagger_check_stm, tagger_check_fc
# fiducialutils MUST precede the taggers (they silently no-op without it); TGM
# before STM (STM skips TGM-flagged mains); tagger_check_fc is LAST because it
# evaluates every in-scope main regardless of the TGM/STM verdicts, and running
# it after them keeps their inputs free of the PCA/hough/steiner-boundary caches
# cluster_fc_check populates (verified identical with and without it on the
# 10-event MCP2025C sample, docs/25_fc-flag.md).  Only a flag that CHANGES that
# list makes this runner name it -- see the pipeline_override block below.
PIPELINE_FULL="switch_scope,unmerge_bundle,unmerge_assoc,steiner,fiducialutils,tagger_check_tgm,tagger_check_stm,tagger_check_fc"

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
  -no-bwonly    run steiner + the TGM/STM/FC taggers on EVERY matched bundle
                (the pre-doc-56 behavior).  DEFAULT is beam-window-only: only
                the bundle(s) whose cluster_t0 is inside -bw get a steiner
                graph and a tagger verdict; out-of-time bundles are cosmics by
                construction.  Off => compiled config byte-identical to
                pre-doc-56.  Env: SBND_BW_ONLY=0.
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
                box in cm (config 5; the pre-adoption value was 3).  Shared by
                tagger_check_tgm and tagger_check_fc.
                Env: SBND_TGM_FVZ_MARGIN=<cm>.
  -fvzi <cm>    downstream-z inset used by check_tgm's CASE-A INTERIOR
                support tests (chord midpoints + waypoint re-check) when > 0
                (config 3; 0 = off, interior tests then share -fvz).  Makes
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
  -mip <e/cm>   MIP dQ/dx scale for TaggerCheckSTM (default 56000 = SBND,
                docs/48).  Pass 50000 (MicroBooNE, the C++ default) to isolate
                the *DeDx reference-table change from the MIP-scale change.
                Env: SBND_MIP_DQDX=<e/cm>.
  -stm-fit      persist the per-pass STM track fits (doc 40): cluster PCs
                stm_fit/stm_pass/stm_eval, a Bee 'stm_fit' layer in
                mabc-pr.zip, and tracking-stm.root (SbndMagnifyTrackingVisitor
                appended to the pipeline) for Magnify-tracking-SBND.
                DEFAULT OFF = byte-identical legacy outputs.
                Env: SBND_STM_FIT=1.
  -stm-guards   doc-63 round-1 STM acceptance guards in TaggerCheckSTM:
                (1) charge-desert one-objectness veto (the fit bridges >=3 cm
                of chargeless path), (2) spike-not-ramp nu-vertex veto (end
                peak >3.2x the 2-6 cm base with a 1.5-4 cm shoulder <1.8 MIP),
                (3) eval acceptance ratio2 cap (measured charge far below MIP
                normalization).  Thresholds measured on the doc-62 owner
                baseline.  DEFAULT ON = SBND production as of doc 63 (owner
                2026-07-26); -no-stm-guards / SBND_STM_GUARDS=0 restores the
                byte-identical legacy verdicts.
  -stm-proton-guard  doc-63 round-2 muon-consistency guard on TaggerCheckSTM's
                detect_proton: an end region matching the muon hypothesis in
                shape AND normalization (ks1 < 0.04, ratio3 < 1.1) is never
                called a proton, so a textbook stopping muon cannot be
                rejected by the high-dQ/dx proton branches (doc-62 miss
                evt62613).  Michel/delta-ray checks untouched.  DEFAULT ON
                (owner 2026-07-26); -no-stm-proton-guard /
                SBND_STM_PROTON_GUARD=0 restores legacy.
  -stm-cathode-guard  doc-63 round-3 cathode-truncation veto in TaggerCheckSTM:
                a fitted stop within 5 cm of the CPA with no Bragg rise (end
                peak < 2.5 MIP) is drift-boundary truncation, not a stop
                (doc-62 false STMs 72586/392200; near-cathode genuine stops
                keep their Bragg and survive).  DEFAULT ON (owner 2026-07-26);
                -no-stm-cathode-guard / SBND_STM_CATHODE_GUARD=0 restores
                legacy.
  -stm-anode-fix  doc-63 round-4a: fix the inverted face selection in the STM
                tagger's dist_to_anode helper -- legacy returned the distance
                to the CATHODE on SBND, so the prototype's anode-clipped-TGM
                catch fired at the wrong drift boundary.  DEFAULT ON (owner
                2026-07-26, after validation); -no-stm-anode-fix /
                SBND_STM_ANODE_FIX=0 restores legacy.
  -stm-track-guard  doc-63 round-4b/4c: second-track vetoes in TaggerCheckSTM.
                4b: a leftover past the kink > 25 cm with straightness > 0.9
                is a second track (V topology, doc-62 false STM 353223), not
                a Michel.  4c: a search_other_tracks segment > 15 cm at
                MIP-like dQ/dx (0.4-1.5) is a second track regardless of
                straightness (349241; the longest MIP-like segment on any
                correct STM is 12.4 cm).  DEFAULT ON (owner 2026-07-26,
                after validation); -no-stm-track-guard /
                SBND_STM_TRACK_GUARD=0 restores legacy.
  -stm-deficit-guard  doc-63 round-5a: an accepted stop whose last-5cm median
                dQ/dx is below 0.6 MIP with no Bragg rise in the last 15 cm
                is a reconstruction truncation, not a stop (doc-62 false STM
                321371; correct STMs bottom out at 0.72 MIP).  DEFAULT ON
                (owner 2026-07-26, after validation); -no-stm-deficit-guard /
                SBND_STM_DEFICIT_GUARD=0 restores legacy.
  -stm-vertex-guard  doc-63 round-5b: a sharp (>45 deg) turn within 12 cm of
                the stop into a >2.2 MIP prong is a nu vertex plus proton,
                not a Bragg rise (doc-62 false STM 402330: 53.5 deg into
                3.67 MIP; sharpest correct turn with a hot prong is 34.8
                deg).  DEFAULT ON (owner 2026-07-26, after validation);
                -no-stm-vertex-guard / SBND_STM_VERTEX_GUARD=0 restores
                legacy.
  -stm-d66cuts  doc-66 sec 12 diffusion-margin cut package in TaggerCheckSTM:
                Michel-veto res_length floor 6 -> 6.5 cm, detect_proton
                track_medium gate 1.0 -> 1.05, block-B ks2 entry 0.05 ->
                0.055, C1 peak clause 4.3 -> 4.1.  Restores the four vetoes
                the 4.0/8.8 diffusion revert lost by hairline margins
                (281632:8, 317543:15, 319809:20, 390864:16 -- all
                owner-adjudicated); full-1000-event sweep found zero
                collateral.  DEFAULT ON (owner 2026-07-27);
                -no-stm-d66cuts / SBND_STM_D66CUTS=0 restores the
                byte-identical legacy verdicts.
  -unmerge      restore the prototype "main cluster + associated clusters"
                data product before the taggers (doc 45): split each
                flash-merged bundle back into its pre-merge main (retained,
                keeps flag_main_cluster and matched_flash_gid) and one
                flag_associated_cluster cluster per co-merged member.  The
                'unmerge_bundle' visitor is inserted between switch_scope and
                steiner.  Without it TaggerCheckSTM fits a bundle of detached
                cosmics as ONE track (showcase-stmfit-mc-evt18 cluster 80) and
                check_other_clusters() has no companions left to count.  Uses
                exact per-blob provenance ONLY: a pctree saved without
                -save-rcid leaves every cluster unsplit (warning per cluster),
                it does NOT silently fall back to a connectivity split.  This
                runner passes -save-rcid to a Q/L step it LAUNCHES, but an
                EXISTING pctree is reused as-is -- point SBND_WORK_ROOT at a
                root whose ql_evt*/ came from a -save-rcid run.
                DEFAULT ON (doc 45): 14 of 329 bundles on the 30-event
                sample move, all toward the prototype's per-cluster view.
                Env: SBND_UNMERGE=0, or -no-unmerge, for the legacy
                merged-bundle pipeline.
  -unmerge-assoc doc 52: ALSO un-merge the per-APA isolated GROUPING (the small
                clusters clustering_isolated merged into each main without
                requiring connectivity), so STM/PR fit the main alone as the
                prototype does.  Implies -unmerge and inserts 'unmerge_assoc'
                after 'unmerge_bundle'.  BOTH un-merges are in the config's
                default pipeline, and the Q/L config saves the arrays this one
                needs, so this flag only re-asserts them; -no-unmerge-assoc
                drops the inner one and -no-unmerge drops both.
                Env: SBND_UNMERGE_ASSOC=0.
  -unmerge-comp like -unmerge but pick the main from the longest RELAXED-GRAPH
                CONNECTED COMPONENT instead of the flash-merge provenance.
                That is a clustering decision, not bookkeeping (the relaxed
                graph does not join the two halves of a cathode crosser), so
                plain -unmerge never falls back to it: a cluster with no
                provenance is left alone with a warning.  Use this only as a
                deliberate probe.  Env: SBND_UNMERGE_MODE=component.
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

NOTE (doc 68): the SBND production operating point lives in the job configs
(cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet and, for the Q/L step it
launches, wct-clus-matching-perevt.jsonnet), not in this script.  Every flag
above is an OVERRIDE; with no flags at all you get production.  "config <x>"
in a flag's text is the value that job's TLA default supplies.

Requires per-event imaging (./run_img_evt.sh) in \$SBND_WORK_ROOT/evt<ID>/.
The Q/L step reruns automatically when the pctree tarball is missing.
EOF
    sbnd_common_help
}

# Every knob variable below is EMPTY by default, meaning "pass no TLA" => the
# config default (= the production operating point) applies.  1 forces true,
# 0 forces false; numeric knobs pass their TLA only when non-empty.  See the
# doc-68 note at the top of this file.
MODE=mc
# -bw l,h overrides the config's beam window (0.2,2.2 us).
BEAM_WINDOW=""
# -no-bwonly / SBND_BW_ONLY=0: evaluate EVERY bundle instead of only the one
# whose cluster_t0 is inside the beam window.  Beam-window-only is ON in the
# config as of doc 56: with the taggers validated, out-of-time bundles are
# cosmics by construction and evaluating them cost ~10x the work for verdicts
# nothing consumes.
BWONLY="${SBND_BW_ONLY:-}"
# Post-PR pctree.  DEFAULT ON: its cluster_scalar carries flag_TGM/flag_STM/
# flag_FC, which is the authoritative tagger verdict source for nusel_extract.
# The log-regex fallback is not trustworthy on its own -- a log line can be torn
# by interleaved writes and then reads as "no verdict" (evt287517 cluster 8
# logged STM=1 but the line split, and the bundle came out not-tagged).
# Costs ~3.5 MB/event.  -no-save-pr-tree to skip it.
SAVEPRT=1
# -no-nucand / SBND_TGM_NUCAND=0: restore the check_neutrino_candidate veto in
# tagger_check_tgm, i.e. make in-beam-window bundles untaggable by construction.
# The veto is OFF in the config as of doc 26 -- leaving it on is not the
# behavior we want from the selection.
NUCAND="${SBND_TGM_NUCAND:-}"
# -no-chord / SBND_TGM_CHORD=0: drop the chord-charge guard in tagger_check_tgm
# (and, with it, tgm_component_extremes).  ON in the config.
CHORD="${SBND_TGM_CHORD:-}"
# Support-measurement mode for the guard: "path" (piecewise charge path, doc 31,
# the config value) or "chord" (straight-chord sampling, doc 29).
CHORD_MODE="${SBND_TGM_CHORD_MODE:-}"
# -no-rescue / SBND_TGM_RESCUE=0: drop the component rescue in tagger_check_tgm
# (path-connected short components keep their extremes).  ON in the config.
RESCUE="${SBND_TGM_RESCUE:-}"
# -no-rescue-chord / SBND_TGM_RESCUE_CHORD=0: drop the straight-chord check on
# rescued-end pairs (doc 33).  ON in the config.
RESCUE_CHORD="${SBND_TGM_RESCUE_CHORD:-}"
# TGM/FC fiducial-box insets, cm (config: z 5, z-interior 3, x 2.5, y 3 --
# doc 39; the pre-adoption values were 3 / 0 / 2 / 2.5).  Empty = inherit.
FVZ_MARGIN="${SBND_TGM_FVZ_MARGIN:-}"
FVZ_INTERIOR="${SBND_TGM_FVZ_INTERIOR:-}"
FVX_MARGIN="${SBND_TGM_FVX_MARGIN:-}"
FVY_MARGIN="${SBND_TGM_FVY_MARGIN:-}"
# MIP dQ/dx scale (e/cm) for TaggerCheckSTM.  Config: 56000 = SBND (docs/48,
# matches the *DeDx tables regenerated at 0.5 kV/cm); pass 50000 for the
# inherited MicroBooNE value (the C++ default), to isolate the table change
# from the MIP-scale change.
MIP_DQDX="${SBND_MIP_DQDX:-}"
# -no-main-pair / SBND_TGM_MAIN_PAIR=0: drop the requirement that TGM pairs
# touch the cluster's main charge component (doc 36).  ON in the config.
MAIN_PAIR="${SBND_TGM_MAIN_PAIR:-}"
# How -main-pair identifies the main: "real" = per-blob flash-merge provenance
# (doc 38, the config value; needs a save_rcid pctree, which is now the Q/L
# default), "path" = largest-component proxy (doc 36).
MAIN_PAIR_MODE="${SBND_TGM_MAIN_PAIR_MODE:-}"
# -no-lm / SBND_QL_LM=0: turn OFF the LM (light-mismatch) tagger in the Q/L step
# (QLMatching lm_tagger, doc 34), which is ON in the Q/L config.  Only affects a
# Q/L step this runner LAUNCHES (pctree missing); an existing pctree is reused
# as-is, so mix arms deliberately or start a fresh work root.
QL_LM="${SBND_QL_LM:-}"
# -stm-fit / SBND_STM_FIT=1: persist per-pass STM track fits + tracking-stm.root
# (doc 40).  OFF in the config -- a diagnostic output, per the doc-64 line.
STM_FIT="${SBND_STM_FIT:-}"
# -no-stm-fv / SBND_STM_FV=0: restore the pre-doc-49 STM containment volume for
# an A/B.  In the config, TaggerCheckSTM's containment gate uses the SAME
# fiducial + margins as tagger_check_tgm / tagger_check_fc, so "contained" means
# one thing across all three verdicts (doc 49).  The old fallback was
# FiducialUtils' un-inset union of per-face sensitive volumes, which exceeds the
# TGM/FC box at every wall -- 96 of 147 "contained" STM skips on the 30-event
# sample were exiters to FC, and none of them ever got a dQ/dx fit.  The
# prototype has no such split (one ToyFiducial shared by check_stm/check_tgm).
STM_FV="${SBND_STM_FV:-}"
# The doc-63 STM guard campaign + the doc-66 sec 12 diffusion-margin cut
# package: all ON in the config (owner 2026-07-26 / 2026-07-27, after
# validation).  Each -no-* flag below opts one out for a pre-campaign A/B.
#   guards   round-1 acceptance guards (charge desert, spike-not-ramp, ratio2 cap)
#   pguard   round-2 muon-consistency guard on detect_proton
#   cguard   round-3 cathode-truncation veto
#   afix     round-4a dist_to_anode face fix
#   tguard   round-4b/4c second-track vetoes
#   dguard   round-5a deficit-end stop-region veto
#   vguard   round-5b vertex-kink stop-region veto
#   d66cuts  Michel res_length 6->6.5cm, detect_proton track_medium 1.0->1.05,
#            block-B ks2 0.05->0.055, C1 peak 4.3->4.1
STM_GUARDS="${SBND_STM_GUARDS:-}"
STM_PGUARD="${SBND_STM_PROTON_GUARD:-}"
STM_CGUARD="${SBND_STM_CATHODE_GUARD:-}"
STM_AFIX="${SBND_STM_ANODE_FIX:-}"
STM_TGUARD="${SBND_STM_TRACK_GUARD:-}"
STM_DGUARD="${SBND_STM_DEFICIT_GUARD:-}"
STM_VGUARD="${SBND_STM_VERTEX_GUARD:-}"
STM_D66CUTS="${SBND_STM_D66CUTS:-}"
# The two un-merge visitors are both in the config's default pipeline.
#
# unmerge_bundle (doc 45) restores the prototype main+associated data product
# before the taggers by splitting each flash-merged bundle back into its
# pre-merge members.  Without it TaggerCheckSTM fits a flash-merged bundle of
# detached cosmics as ONE track -- evt284657 cluster 27 was fitted across the
# cathode (139 pts, 71.8 cm, x -18.3..+6.9) and rejected as "long leftover";
# split, the same cluster is a 29.5 cm one-sided stopping muon that PASSES
# eval_stm.  On the 30-event dq48v3 sample the split moved 14 of 329 bundles
# (+7 STM, +3 TGM, +2 FC, -2 FC on sub-steiner leftovers), every split from
# exact per-blob provenance (0 proxy fallbacks).  It sits between switch_scope
# and steiner (it MUST precede steiner_pc creation).
#
# unmerge_assoc (doc 52) then ALSO undoes the per-APA isolated GROUPING, which
# merged each main cluster with the small clusters near but not connected to it.
# It needs the Q/L save_assoc arrays (a Q/L config default since doc 68);
# without them the visitor is a no-op and warns.  Runs right after
# unmerge_bundle (flash outer, isolated inner), still before steiner.
#
# -no-unmerge / SBND_UNMERGE=0 drops BOTH (the legacy merged-bundle pipeline);
# -no-unmerge-assoc / SBND_UNMERGE_ASSOC=0 drops only the inner one.
UNMERGE="${SBND_UNMERGE:-}"
# How unmerge_bundle identifies the main: "real" = per-blob flash-merge
# provenance (exact, the config value), "component" = longest connected
# component (proxy).  -unmerge-comp selects the proxy.
UNMERGE_MODE="${SBND_UNMERGE_MODE:-}"
UNMERGE_ASSOC="${SBND_UNMERGE_ASSOC:-}"
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -N) SBND_SAMPLE="$2"; shift 2 ;;
        -N*) SBND_SAMPLE="${1#-N}"; shift ;;
        mc|data) MODE="$1"; shift ;;
        -bw) BEAM_WINDOW="$2"; shift 2 ;;
        -bwonly|--bwonly) BWONLY=1; shift ;;
        -no-bwonly|--no-bwonly) BWONLY=0; shift ;;
        -save-pr-tree|--save-pr-tree) SAVEPRT=1; shift ;;
        -no-save-pr-tree|--no-save-pr-tree) SAVEPRT=""; shift ;;
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
        -mip|--mip) MIP_DQDX="$2"; shift 2 ;;
        -main-pair|--main-pair) MAIN_PAIR=1; shift ;;
        -main-pair-real|--main-pair-real) MAIN_PAIR=1; MAIN_PAIR_MODE=real; shift ;;
        -no-main-pair|--no-main-pair) MAIN_PAIR=0; MAIN_PAIR_MODE=path; shift ;;
        -lm|--lm) QL_LM=1; shift ;;
        -no-lm|--no-lm) QL_LM=0; shift ;;
        -stm-fit|--stm-fit) STM_FIT=1; shift ;;
        -no-stm-fit|--no-stm-fit) STM_FIT=0; shift ;;
        -stm-guards|--stm-guards) STM_GUARDS=1; shift ;;
        -no-stm-guards|--no-stm-guards) STM_GUARDS=0; shift ;;
        -stm-proton-guard|--stm-proton-guard) STM_PGUARD=1; shift ;;
        -no-stm-proton-guard|--no-stm-proton-guard) STM_PGUARD=0; shift ;;
        -stm-cathode-guard|--stm-cathode-guard) STM_CGUARD=1; shift ;;
        -no-stm-cathode-guard|--no-stm-cathode-guard) STM_CGUARD=0; shift ;;
        -stm-anode-fix|--stm-anode-fix) STM_AFIX=1; shift ;;
        -no-stm-anode-fix|--no-stm-anode-fix) STM_AFIX=0; shift ;;
        -stm-track-guard|--stm-track-guard) STM_TGUARD=1; shift ;;
        -no-stm-track-guard|--no-stm-track-guard) STM_TGUARD=0; shift ;;
        -stm-deficit-guard|--stm-deficit-guard) STM_DGUARD=1; shift ;;
        -no-stm-deficit-guard|--no-stm-deficit-guard) STM_DGUARD=0; shift ;;
        -stm-vertex-guard|--stm-vertex-guard) STM_VGUARD=1; shift ;;
        -no-stm-vertex-guard|--no-stm-vertex-guard) STM_VGUARD=0; shift ;;
        -stm-d66cuts|--stm-d66cuts) STM_D66CUTS=1; shift ;;
        -no-stm-d66cuts|--no-stm-d66cuts) STM_D66CUTS=0; shift ;;
        -stm-fv|--stm-fv) STM_FV=1; shift ;;
        -no-stm-fv|--no-stm-fv) STM_FV=0; shift ;;
        -unmerge|--unmerge) UNMERGE=1; shift ;;
        -unmerge-comp|--unmerge-comp) UNMERGE=1; UNMERGE_MODE=component; shift ;;
        -unmerge-assoc|--unmerge-assoc) UNMERGE=1; UNMERGE_ASSOC=1; shift ;;
        -no-unmerge-assoc|--no-unmerge-assoc) UNMERGE_ASSOC=0; shift ;;
        -no-unmerge|--no-unmerge) UNMERGE=0; shift ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"

# --- Pipeline override ---------------------------------------------------
# The config's default pipeline is $PIPELINE_FULL.  We name pipeline_names ONLY
# when a flag actually CHANGES that list; otherwise the TLA is omitted and the
# config's own list stands (doc 68).
#
# Ordering constraints encoded in PIPELINE_FULL: unmerge_bundle must run AFTER
# switch_scope (which re-applies the T0 correction and carves the out-of-volume
# shards, carrying the per-blob provenance across that carve) and BEFORE steiner
# (separate() does not carry node-local PCs, so a steiner_pc built from the
# pre-split blob set would survive on the retained main).  unmerge_assoc is the
# INNER un-merge and sits right after the outer one, still before steiner.
PIPELINE="$PIPELINE_FULL"
PIPELINE_TLA=()
_pipe_changed=0
if [ "$UNMERGE" = 0 ]; then
    PIPELINE="${PIPELINE/unmerge_bundle,/}"; PIPELINE="${PIPELINE/unmerge_assoc,/}"
    _pipe_changed=1
elif [ "$UNMERGE_ASSOC" = 0 ]; then
    PIPELINE="${PIPELINE/unmerge_assoc,/}"
    _pipe_changed=1
fi
# -stm-fit appends the Magnify-tracking ROOT dump to the tagger pipeline.
if [ "$STM_FIT" = 1 ]; then
    PIPELINE="$PIPELINE,stm_magnify"
    _pipe_changed=1
fi
if [ "$_pipe_changed" = 1 ]; then
    PIPELINE_TLA=(--tla-code "pipeline_names=[$(echo "$PIPELINE" | sed "s/[^,]\+/'&'/g")]")
fi

# --- Override TLAs -------------------------------------------------------
# Built once: an EMPTY knob emits no TLA, so the job's own default (= the SBND
# production operating point, doc 68) stands.  Only an explicit value speaks.
KNOB_TLA=()
knob_bool() {   # knob_bool <tla-name> <value>
    case "$2" in
        1) KNOB_TLA+=(--tla-code "$1=true") ;;
        0) KNOB_TLA+=(--tla-code "$1=false") ;;
    esac
}
knob_num() {    # knob_num <tla-name> <value>
    [ -n "$2" ] && KNOB_TLA+=(--tla-code "$1=$2")
    return 0    # an empty knob is the normal case, not a failure (set -e)
}
knob_str() {    # knob_str <tla-name> <value>
    [ -n "$2" ] && KNOB_TLA+=(--tla-str "$1=$2")
    return 0
}
knob_bool beam_window_only     "$BWONLY"
knob_bool tgm_neutrino_candidate "$NUCAND"
# -chord drives the guard and the component-extremes pair together, as it always did.
knob_bool tgm_chord_charge       "$CHORD"
knob_bool tgm_component_extremes "$CHORD"
knob_str  tgm_chord_mode         "$CHORD_MODE"
knob_bool tgm_component_rescue   "$RESCUE"
knob_bool tgm_rescue_chord       "$RESCUE_CHORD"
knob_bool tgm_main_pair          "$MAIN_PAIR"
knob_str  tgm_main_pair_mode     "$MAIN_PAIR_MODE"
knob_num  tgm_fv_zmax_margin          "$FVZ_MARGIN"
knob_num  tgm_fv_zmax_margin_interior "$FVZ_INTERIOR"
knob_num  tgm_fv_x_margin             "$FVX_MARGIN"
knob_num  tgm_fv_y_margin             "$FVY_MARGIN"
knob_num  mip_dqdx                    "$MIP_DQDX"
knob_str  unmerge_bundle_mode         "$UNMERGE_MODE"
knob_bool save_stm_fit           "$STM_FIT"
knob_bool stm_consistent_fv      "$STM_FV"
knob_bool stm_accept_guards      "$STM_GUARDS"
knob_bool stm_proton_muon_guard  "$STM_PGUARD"
knob_bool stm_cathode_guard      "$STM_CGUARD"
knob_bool stm_anode_dist_fix     "$STM_AFIX"
knob_bool stm_second_track_guard "$STM_TGUARD"
knob_bool stm_deficit_guard      "$STM_DGUARD"
knob_bool stm_vertex_kink_guard  "$STM_VGUARD"
knob_bool stm_d66_cuts           "$STM_D66CUTS"
[ -n "$BEAM_WINDOW" ] && KNOB_TLA+=(--tla-code "beam_window_us=[$BEAM_WINDOW]")
# The TrackFitting parameter file: only an SBND_TRACKFIT_JSON override is named;
# otherwise the config resolves its own copy through WIRECELL_PATH (doc 64 4a).
TFJSON_TLA=()
[ -n "$TFJSON" ] && TFJSON_TLA=(--tla-str "trackfitting_config=$TFJSON")
# nusel_extract.py is a post-processor, not a wire-cell node, so it cannot read
# the job's beam window -- it needs the number spelled out.  Keep this in step
# with beam_window_us in cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet.
BEAM_WINDOW_EFF="${BEAM_WINDOW:-0.2,2.2}"

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
        # The Q/L step's own config supplies lm, save_rcid and save_assoc (doc
        # 68), so only the per-event OUTPUT paths are named here: the pctree
        # this job consumes and the calib JSON carrying the per-bundle lm*
        # metrics the tuning/hand-scan read (doc 34).
        local QL_FLAGS=(-save-pctree -calib)
        # Pass an explicit LM override straight through when the caller gave one.
        [ "$QL_LM" = 1 ] && QL_FLAGS+=(-lm)
        [ "$QL_LM" = 0 ] && QL_FLAGS+=(-no-lm)
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
    local _ov="config defaults"
    [ ${#KNOB_TLA[@]} -gt 0 ] && _ov="${KNOB_TLA[*]}"
    echo "[evt $EVT_ID] rse=($RUN_NO, $SUBRUN_NO, $EVT_ID) taggers ($PIPELINE${SAVEPRT:+, save-pr-tree}) overrides: $_ov"
    rm -f "$LOG"
    # Provenance: the fit JSON is the only live diffusion consumer, so name it
    # (and its DL/DT) in every log -- an A/B arm is worthless if you cannot tell
    # afterwards which file it ran with.  Empty TFJSON = the config's own
    # WIRECELL_PATH-resolved pgrapher/experiment/sbnd/sbnd_track_fitting.json.
    echo "trackfitting_config=${TFJSON:-<config default: pgrapher/experiment/sbnd/sbnd_track_fitting.json>} $(grep -h '"D[LT]"' "${TFJSON:-$WCT_BASE/toolkit/cfg/pgrapher/experiment/sbnd/sbnd_track_fitting.json}" 2>/dev/null | tr -d ' \n')"
    (
        cd "$NUDIR"
        wire-cell \
            -l stderr -l "${LOG}:${SBND_WCT_LOGLEVEL:-debug}" -L "${SBND_WCT_LOGLEVEL:-debug}" \
            --tla-str  "input=$PCT" \
            --tla-str  "output_dir=$NUDIR" \
            --tla-code "run=${RUN_NO}" --tla-code "subrun=${SUBRUN_NO}" --tla-code "event=${EVT_ID}" \
            --tla-str  "reality=$REALITY" \
            --tla-str  "save_tensors=$SAVEPRT_TLA" \
            "${TFJSON_TLA[@]}" \
            "${PIPELINE_TLA[@]}" \
            "${KNOB_TLA[@]}" \
            -c "$JSONNET"
    ) || return 1
    rm -f "$NUDIR/trash-pr.tar.gz"

    # 3. Per-bundle label table.
    python3 "$SBND_DIR/nusel_extract.py" \
        --pctree "$PCT" --prbee "$NUDIR/mabc-pr.zip" --prlog "$LOG" \
        --prtree "$NUDIR/pctree-pr-evt${EVT_ID}.tar.gz" \
        --qlbee "$QLDIR/mabc-all-apa.zip" \
        --beam-window "$BEAM_WINDOW_EFF" \
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
