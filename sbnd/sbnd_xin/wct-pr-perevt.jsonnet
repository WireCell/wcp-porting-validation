// Per-event SBND pattern-recognition (PR) job — standalone, self-contained.
//
// Loads the persisted post-QL point-cloud tree written by the Q/L job
// (run_ql_evt.sh -save-pctree -> work/ql_evt<ID>/pctree-evt<ID>.tar.gz) and
// runs the PR-tail visitors on it inside a MultiAlgBlobClustering node.  With
// the default empty pipeline this is the round-trip identity gate: the
// 'clustering' Bee layer of the output mabc-pr.zip must be content-identical
// to the Q/L job's mabc-all-apa.zip clustering layer.
// See ../docs/sbnd-pattern-recognition.md.  Driven by run_pr_evt.sh.
//
// Usage (called from run_pr_evt.sh):
//   wire-cell \
//     --tla-str  input=work/ql_evt2/pctree-evt2.tar.gz \
//     --tla-code anode_indices='[0,1]' \
//     --tla-str  output_dir=work/pr_evt2 \
//     --tla-code run=0 --tla-code subrun=0 --tla-code event=2 \
//     --tla-str  reality=sim \
//     --tla-code 'pipeline_names=[]' \
//     -c wct-pr-perevt.jsonnet

local g = import 'pgraph.jsonnet';
local wc = import 'wirecell.jsonnet';
local tools_maker = import 'pgrapher/common/tools.jsonnet';

function(
    input          = '.',
    anode_indices  = [0, 1],
    output_dir     = '.',
    run            = 0,
    subrun         = 0,
    event          = 0,
    reality        = 'sim',
    // Same LAr TLAs as the Q/L job so the anode/params objects are identical.
    DL             = 6.5781,
    DT             = 13.1349,
    lifetime       = 6.0,
    driftSpeed     = 1.563,
    // PR visitors to run, by name (see clus_pr's cm_by_name in
    // cfg/pgrapher/experiment/sbnd/clus.jsonnet). [] = pass-through (identity gate).
    // Full STM demo: ['switch_scope','steiner','fiducialutils','tagger_check_stm'].
    pipeline_names = [],
    // TrackFitting parameter JSON (absolute path; plain ifstream, not
    // WIRECELL_PATH-resolved).  Required when tagger_check_stm is in the
    // pipeline -- the C++ preset defaults are uBooNE-hard-coded.
    trackfitting_config = '',
    // MIP dQ/dx scale in e/cm handed to TaggerCheckSTM.  56000 = the SBND value
    // (docs/48), matching the *DeDx tables now regenerated at 0.5 kV/cm.  Pass
    // 50000 (the MicroBooNE value, and the C++ default) to isolate the
    // reference-table change from the MIP-scale change in an A/B.
    mip_dqdx       = 56000,
    // TaggerCheckSTM's containment gate (cluster_fc_check) uses the same fiducial
    // + margins as tagger_check_tgm / tagger_check_fc, so "contained" means one
    // thing across all three verdicts.  true = SBND production (docs/49); false
    // restores the pre-doc-49 FiducialUtils sensitive-volume fallback, which is
    // what the A/B compares against.  Runner flag: -no-stm-fv.
    stm_consistent_fv = true,
    // doc-63 round-1 STM acceptance guards (charge-desert one-objectness,
    // spike-not-ramp nu-vertex veto, eval ratio2 cap).  C++ default false;
    // key omitted when off => byte-identical compiled config.  Runner flag:
    // -stm-guards / SBND_STM_GUARDS=1.
    stm_accept_guards = false,
    // When set, re-save the (post-PR) tree to this TensorDM tarball.  Used by the
    // round-trip gate (re-save with pipeline_names=[] and compare member hashes
    // against the input tarball).
    save_tensors   = '',
    // SCN vertex weights (WIRECELL_PATH-resolved, e.g.
    // 'uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth').  '' = geometric vertex.
    // NOTE: only uBooNE-trained weights exist; SBND use is an untuned demo.
    dl_weights     = '',
    // Beam window [low, high) in us on cluster_t0 (= matched flash time) selecting
    // the bundle that gets neutrino PR.  [0,0] disables the gate (then
    // tagger_check_neutrino falls back to uBooNE single-main selection, which on
    // SBND picks an arbitrary main -- always set a window with tagger_check_neutrino).
    beam_window_us = [0, 0],
    // Run the PR tail (steiner + tagger_check_{tgm,stm,fc}) ONLY on the
    // beam-coincident bundle, i.e. the mains whose cluster_t0 is inside
    // beam_window_us (plus, for steiner, the companions sharing their
    // matched_flash_gid).  TRUE = SBND production default as of docs/56;
    // false restores the evaluate-every-bundle behavior with a compiled config
    // byte-identical to the pre-doc-56 one.  Inert when beam_window_us is empty.
    // Runner flag: -no-bwonly.
    beam_window_only = true,
    // Enable the ported check_neutrino_candidate veto in tagger_check_tgm so
    // in-beam-window bundles may be tagged TGM (C++ default false; key
    // omitted when off => byte-identical pre-port config).
    tgm_neutrino_candidate = false,
    // Require charge along the chord between the two extreme points before a
    // pair may tag TGM (C++ default false; key omitted when off =>
    // byte-identical pre-fix config).  Guards against the QL flash-time merge
    // grafting a detached fragment into the tagged cluster -- see
    // docs/29_tgm-chord-charge.md.
    tgm_chord_charge = false,
    // How the chord-charge guard measures support (C++ default "chord"; key
    // omitted then => byte-identical).  "chord" samples the STRAIGHT segment
    // between the extremes -- it falsely rejects curved tracks (evt285185
    // cluster 16: a continuous 480 cm top->anode crosser bows 10 cm off its
    // chord and lost a real TGM, doc 31).  "path" requires a piecewise charge
    // path through the cluster's own points with no jump > chord_max_gap.
    // The -chord runner flag passes "path" (SBND_TGM_CHORD_MODE overrides).
    tgm_chord_mode = 'chord',
    // Find the extreme points PER connected component and union them, instead
    // of 8 global extremes over the whole flash-merged cluster (C++ default
    // false; key omitted when off).  Must be used WITH tgm_chord_charge -- see
    // docs/29_tgm-chord-charge.md.  The -chord runner flag sets both.
    tgm_component_extremes = false,
    // Let a component shorter than component_min_length still donate its
    // extremes when it is path-connected (30 cm-step charge path) to a
    // component that passed the length cut, so a fragmented genuine track END
    // keeps its wall exit (evt286681 cluster 7) while detached specks stay
    // dropped (C++ default false; key omitted when off => byte-identical).
    // Only meaningful WITH tgm_component_extremes.  Runner flag: -rescue.
    tgm_component_rescue = false,
    // Rescued-end pairs must ALSO pass the straight-chord support test even
    // in path mode (C++ default false; key omitted when off => byte-identical
    // doc-32 rescue behavior).  Path mode alone lets a rescued speck pair
    // across TWO merged cosmics through an L-shaped charge detour (evt288727
    // cluster 6, doc 33).  Only meaningful WITH tgm_component_rescue.
    // Runner flag: -rescue-chord.
    tgm_rescue_chord = false,
    // A pair may tag TGM only when at least one end lies in the cluster's
    // MAIN charge component -- the largest 30 cm path component; a cathode
    // crosser is ONE such component (C++ default false; key omitted when off
    // => byte-identical).  A merged-in fragment that is itself through-going
    // otherwise tags the whole bundle on its own within-component pair,
    // which the chord guard deliberately allows and the nu-candidate veto
    // cannot protect (it walks the pair's own path): evt289343 cluster 9,
    // in-beam bundle tagged TGM by a 26 cm corner-clipping cosmic fragment
    // 450 cm from the main track (doc 36).  Runner flag: -main-pair.
    tgm_main_pair = false,
    // How tgm_main_pair identifies the main (C++ default "path"; key omitted
    // then => byte-identical doc-36 behavior).  "path" = largest 30 cm path
    // component (proxy; wrong if a merged-in cosmic outweighs the main).
    // "real" = per-blob real_cluster_main flash-merge provenance -- exact,
    // needs a pctree saved with run_ql_evt.sh -save-rcid; falls back to the
    // proxy on old tarballs (doc 38).  Runner flag: -main-pair-real.
    tgm_main_pair_mode = 'path',
    // Downstream-z (z ~ 500 cm face) inset of the TGM/FC fiducial box, in cm
    // (default 3 = byte-identical legacy margin).  Shared by tagger_check_tgm
    // and tagger_check_fc so containment keeps one meaning.  Runner flag:
    // -fvz <cm>.
    tgm_fv_zmax_margin = 3,
    // Downstream-z inset used by check_tgm's CASE-A INTERIOR-support tests
    // (chord midpoints + waypoint re-check) when > 0, in cm (default 0 =
    // OFF, key omitted => byte-identical; the interior tests then share
    // tgm_fv_zmax_margin).  Makes the doc-32 widening endpoint-only so a
    // wall-hugging corner clipper keeps its midpoint support (doc 35,
    // evt287517 cluster 16 / evt289805 cluster 9).  TGM only; FC untouched.
    // Runner flag: -fvzi <cm>.
    tgm_fv_zmax_margin_interior = 0,
    // Drift-x and vertical-y insets of the TGM/FC fiducial box, in cm, both
    // faces symmetric (defaults 2 / 2.5 = byte-identical legacy margins).
    // Shared by tagger_check_tgm and tagger_check_fc, same as
    // tgm_fv_zmax_margin.  Runner flags: -fvx <cm> / -fvy <cm>.
    tgm_fv_x_margin = 2,
    tgm_fv_y_margin = 2.5,
    // Persist the per-pass STM track fits (C++ default false; key omitted
    // when off => byte-identical): cluster PCs stm_fit/stm_pass/stm_eval, a
    // Bee 'stm_fit' layer in mabc-pr.zip, and (when 'stm_magnify' is added
    // to pipeline_names) tracking-stm.root for Magnify-tracking-SBND.  Also
    // gates loading the WireCellRoot plugin.  Runner flag: -stm-fit.
    save_stm_fit = false,
    // How 'unmerge_bundle' (when named in pipeline_names) identifies the main
    // sub-cluster of a flash-merged bundle (C++ default "real").  "real" =
    // per-blob real_cluster_main/real_cluster_id provenance -- exact, needs a
    // pctree saved with run_ql_evt.sh -save-rcid; a cluster without it is
    // left UNSPLIT (warning), never proxied.  "component" = longest connected
    // component -- a clustering decision (breaks cathode crossers), opt-in
    // only.  Runner flags: -unmerge / -unmerge-comp (doc 45).
    unmerge_bundle_mode = 'real',
)
    local base = import 'pgrapher/experiment/sbnd/simparams.jsonnet';
    local params = base {
        lar: super.lar {
            DL:          DL * wc.cm2 / wc.s,
            DT:          DT * wc.cm2 / wc.s,
            lifetime:    lifetime * wc.ms,
            drift_speed: driftSpeed * wc.mm / wc.us,
        },
    };

    local tools_all = tools_maker(params);
    local anodes  = [tools_all.anodes[i] for i in anode_indices];

    local source = g.pnode({
        type: 'TensorFileSource',
        name: 'pr_pctree',
        data: { inname: input, prefix: 'clustering_' },
    }, nin=0, nout=1);

    local clus_mod = import 'clus.jsonnet';
    local clus_maker = clus_mod(
        output_dir=output_dir,
        runNo=run,
        subRunNo=subrun,
        eventNo=event,
        reality=reality,
    );
    // dQ/dx (e/cm, SBND 0.5 kV/cm) + range (detector-agnostic) LinterpFunctions.
    local pds = (import '../particle_dataset.jsonnet')();
    local pr = clus_maker.pr(anodes, dump=true,
                             pipeline_names=pipeline_names,
                             tensor_outname=save_tensors,
                             trackfitting_config_file=trackfitting_config,
                             particle_dataset=pds.particle_dataset,
                             extra_uses=pds.all,
                             dl_weights=dl_weights,
                             beam_window=[t * wc.us for t in beam_window_us],
                             beam_window_only=beam_window_only,
                             tgm_neutrino_candidate=tgm_neutrino_candidate,
                             tgm_chord_charge=tgm_chord_charge,
                             tgm_chord_mode=tgm_chord_mode,
                             tgm_component_extremes=tgm_component_extremes,
                             tgm_component_rescue=tgm_component_rescue,
                             tgm_rescue_chord=tgm_rescue_chord,
                             tgm_main_pair=tgm_main_pair,
                             tgm_main_pair_mode=tgm_main_pair_mode,
                             tgm_fv_zmax_margin=tgm_fv_zmax_margin,
                             tgm_fv_zmax_margin_interior=tgm_fv_zmax_margin_interior,
                             tgm_fv_x_margin=tgm_fv_x_margin,
                             tgm_fv_y_margin=tgm_fv_y_margin,
                             save_stm_fit=save_stm_fit,
                             unmerge_bundle_mode=unmerge_bundle_mode,
                             mip_dqdx=mip_dqdx,
                             stm_consistent_fv=stm_consistent_fv,
                             stm_accept_guards=stm_accept_guards);

    local graph = g.intern(
        innodes=[source],
        outnodes=[pr],
        edges=[g.edge(source, pr, 0, 0)],
    );

    local app = {
        type: 'Pgrapher',
        data: { edges: g.edges(graph) },
    };

    local cmdline = {
        type: 'wire-cell',
        data: {
            plugins: ['WireCellGen', 'WireCellPgraph', 'WireCellAux', 'WireCellSio',
                      'WireCellSigProc', 'WireCellImg', 'WireCellClus']
                     // WireCellRoot hosts SbndMagnifyTrackingVisitor; loaded
                     // only with the knob so the compiled config stays
                     // byte-identical when off.
                     + (if save_stm_fit then ['WireCellRoot'] else []),
            apps: ['Pgrapher'],
        },
    };

    [cmdline] + g.uses(graph) + [app]
