// SBND imaging + clustering + charge/light (Q/L) matching, faithfully following
// Xin's standalone chain (sbnd_xin/wct-clus-matching-standalone.jsonnet +
// wct-img-all.jsonnet) but reading the artROOT DIRECTLY instead of dumped files:
//   - charge: wclsCookedFrameSource (recob::Wire)        [vs ClusterFileSource npz]
//   - light : wclsOpFlashSource per TPC (ITensorSet)     [vs TensorFileSource tar.gz]
//
// Imaging uses the TOOLKIT img.jsonnet 'multi-3view' with full_deghost=true: one
// per-anode pipe emits the live (active, multi_active 2-view-recovering) clusters
// on port 0 and the dead (masked, multi_masked_2view) clusters on port 1 -- the
// same live+dead views Xin produces with wct-img-all.jsonnet.  These feed the
// toolkit clus.jsonnet per_apa (PointTreeBuilding live/dead -> MABC).
//
// Matching uses the canonical FlashTensorToOpticalPCs + QLMatching (WireCellMatch),
// JOINT over both APAs (one node, premerged all_apa), exactly as Xin's default.
//
// All MABC nodes (per-APA + all-APA) write into ONE shared Bee zip: mabc.zip.
//
// Requires the toolkit cfg + photodet on WIRECELL_PATH -- source sbnd/setup-ap.sh.
// Nothing under sbnd_xin is imported or modified (we import the in-tree canonical
// pgrapher/experiment/sbnd/{img,clus,qlmatching,cathode_fiducial}.jsonnet directly).

local g = import "pgraph.jsonnet";
local wc = import "wirecell.jsonnet";

local tools_maker = import 'pgrapher/common/tools.jsonnet';

local reality = std.extVar('reality');

// The reco chain's reality config (use_sce, pos_offset_on) is grouped inside
// the toolkit clus maker keyed by `reality` -- see clus.jsonnet `reco` local.

// The follow-up tail -- STM/TGM/FC PR taggers -> wclsTensorSetLabeler -> dump --
// is assembled below in this entry config (see the "follow-up tail" section);
// clus.jsonnet now produces only the clustering+matching all-APA MABC.  Requires
// the "WireCellAIML" plugin + "wclsTensorSetLabeler:clus_all_apa" inputer in the
// fcl (both fcls have them); the labeler runs in BOTH realities (sim attaches nu
// truth + truth_per_track + truth Bee sets + nugraph-with-truth, data attaches
// RSE + input-only HDF5).  The tagger_stm/tgm/fc Bee sets are emitted in both.

// Canonical SBND simparams (toolkit).  drift_speed (1.563 mm/us) flows into
// QLMatching from here; no DL/DT/lifetime/driftSpeed extVars are needed.
local params = import 'pgrapher/experiment/sbnd/simparams.jsonnet';

local tools_all = tools_maker(params);
local tools = tools_all {anodes: [tools_all.anodes[n] for n in [0, 1]]};
local nanode = std.length(tools.anodes);

// ---- artROOT inputs (must match the names used in the fcl inputers) ----
local wcls_input = {
    sigs: g.pnode({
        type: 'wclsCookedFrameSource',
        name: 'sigs',
        data: {
            nticks: params.daq.nticks,
            frame_scale: 50,                              // scale up input recob::Wire
            summary_scale: 50,                            // scale up input summary
            frame_tags: ["orig"],
            recobwire_tags: std.extVar('recobwire_tags'),
            trace_tags: std.extVar('trace_tags'),
            summary_tags: std.extVar('summary_tags'),
            input_mask_tags: std.extVar('input_mask_tags'),
            output_mask_tags: std.extVar('output_mask_tags'),
        },
    }, nin=0, nout=1),
    // wclsOpFlashSource emits an ITensorSet [nflashes x (npmts+1)] (npmts=312 ->
    // 313 cols) -- exactly the opflash matrix FlashTensorToOpticalPCs reads on
    // port 1, so it feeds flash_attach directly (no opflash file dump/read).
    opflashes0: g.pnode({
        type: 'wclsOpFlashSource',
        name: 'tpc0',
        data: { art_tag: std.extVar('opflash0_input_label') },
    }, nin=0, nout=1),
    opflashes1: g.pnode({
        type: 'wclsOpFlashSource',
        name: 'tpc1',
        data: { art_tag: std.extVar('opflash1_input_label') },
    }, nin=0, nout=1),
    opflashes: [wcls_input.opflashes0, wcls_input.opflashes1],
};

// ---- frame fanout: one sigs frame -> per-anode frame with the per-anode
// 'gauss<id>'/'wiener<id>' trace tags the imaging pre_proc expects ----
local fanout_rules = [
    {
        frame: { '.*': 'orig%d' % tools.anodes[n].data.ident },
        trace: {
            gauss: 'gauss%d' % tools.anodes[n].data.ident,
            wiener: 'wiener%d' % tools.anodes[n].data.ident,
        },
    }
    for n in std.range(0, nanode - 1)
];
local frame_fan = g.pnode({
    type: 'FrameFanout',
    name: 'sig_fanout',
    data: { multiplicity: nanode, tag_rules: fanout_rules },
}, nin=1, nout=nanode);

// ---- imaging: toolkit multi-3view + full_deghost (live=port0, dead=port1) ----
local img = import 'pgrapher/experiment/sbnd/img.jsonnet';
local img_maker = img();
local img_pipes = [
    img_maker.per_anode(tools.anodes[n], 'multi-3view', add_dump=false, full_deghost=true)
    for n in std.range(0, nanode - 1)
];

// ---- clustering: toolkit per_apa, shared Bee ----
local clus = import 'pgrapher/experiment/sbnd/clus.jsonnet';
// rse_from_ident: each frame's tensor ident carries the real event id, so the
// Bee display is labelled with the true event number (matches Xin's chain).
local clus_maker = clus(rse_from_ident=true, reality=reality);
// Single shared Bee sink: every MABC node (per-APA + all-APA) writes into this
// one zip instead of one zip per node.
local bee_shared = {
    type: 'BeeSink',
    name: 'mabc_shared',
    data: { outname: 'mabc.zip', initial_index: 0 },
};
// save_assoc_id=true: the per-APA clustering_isolated pass writes the isolated-
// grouping provenance (assoc_cluster_id/assoc_cluster_main); this flag
// homogenizes it so it SURVIVES serialization out of the per-APA MABC (the
// serializer silently drops heterogeneous perblob keys).  Without it the
// provenance is born here but dropped before matching_joint, so the all-APA
// save_assoc_cluster_id has nothing to preserve and the pr_node's unmerge_assoc
// finds nothing to split ("unmerged 0 main clusters") -- leaving the tagger
// main un-refined and mis-verdicted (MC 32-10-10 cluster 22: STM=0 un-split vs
// STM=1 after the isolated split, matching sbnd_xin's 2-step).  Pairs with the
// all_apa save_real/assoc_cluster_id below (real_cluster_id is instead created
// by all_apa's own examine_bundles, so it needs no per-APA flag).
local clus_pipes = [
    clus_maker.per_apa(tools.anodes[n], dump=false, bee_sink=bee_shared, save_assoc_id=true)
    for n in std.range(0, nanode - 1)
];

// ---- Q/L matching: canonical FlashTensorToOpticalPCs + joint QLMatching ----
local semimodel_file = 'semi-analytical-sbnd.json';
local pmt_nl = true;

local qlm = (import 'pgrapher/experiment/sbnd/qlmatching.jsonnet')(params);
local cathode_fv = (import 'pgrapher/experiment/sbnd/cathode_fiducial.jsonnet')(
    cx=0.5*wc.cm, cy=0.5*wc.cm, cz=0.5*wc.cm);
// cluster tree (port 0) + opflash matrix (port 1) -> tree with flash PCs attached
local flash_attach = [qlm.flash_attach(n) for n in std.range(0, nanode - 1)];
// ONE joint QLMatching node fed by both APAs' flash_attach outputs; it matches
// each APA and merges the per-APA trees, feeding the all-APA MABC directly.
local matching_joint = qlm.matching_joint(
    tools.anodes, clus_maker.detector_volumes(tools.anodes),
    reality, semimodel_file, cathode_fiducial=cathode_fv.tn, pmt_nl=pmt_nl);

// all-APA clustering + matching ONLY (joint matcher already merged ->
// premerged=true; dump=false -> no terminal sink here, the tail below consumes
// the MABC tensor output).  Bee (clustering-global etc.) still goes to the
// shared mabc.zip.
// save_real_cluster_id / save_assoc_cluster_id: attach the flash-merge (and
// isolated-grouping) provenance to the grouping so the pr_node's unmerge_bundle
// / unmerge_assoc can actually split the flash-merged bundles.  WITHOUT these,
// unmerge logs "no flash-merge provenance ... not split" and every cluster stays
// merged across the cathode -> the taggers judge the wrong (merged) geometry
// (e.g. nue evt 46363 cluster 19 spans both TPCs -> FC=false, vs the compact
// post-split main -> FC=true in sbnd_xin's 2-step chain).
local clus_all_apa = clus_maker.all_apa(tools.anodes, dump=false,
                                        bee_sink=bee_shared, premerged=true,
                                        save_real_cluster_id=true,
                                        save_assoc_cluster_id=true);

// ==== follow-up tail (moved out of clus.jsonnet into this entry config) ====
//   clus_all_apa (MABC) -> pr_node (STM/TGM/FC taggers) -> labeler -> tail_dump

// Beam gate shared by the tagger PR pass AND the labeler's tagger-Bee cluster_id
// encoding (which uses it to tell "beam-window candidate" mains from out-of-
// window mains) -- ONE source so the two never drift.
local beam_window = [0.2 * wc.us, 2.2 * wc.us];

// PR tagger pass: SBND production operating point (clus_maker.pr defaults),
// running the nusel tagger visitors so each cluster carries flag_STM/TGM/FC for
// the labeler to read.  dump=false -> a pass-through tensor node.
local pds = (import 'pgrapher/experiment/sbnd/particle_dataset.jsonnet')();
local pr_node = clus_maker.pr(
    tools.anodes, dump=false,
    pipeline_names=['switch_scope', 'unmerge_bundle', 'unmerge_assoc', 'steiner',
                    'fiducialutils', 'tagger_check_tgm', 'tagger_check_stm', 'tagger_check_fc'],
    particle_dataset=pds.particle_dataset, extra_uses=pds.all, beam_window=beam_window);

// wclsTensorSetLabeler (larwirecell "WireCellAIML" plugin).  Node name kept as
// 'clus_all_apa' so the fcl inputer "wclsTensorSetLabeler:clus_all_apa" still
// resolves.  Deps come from clus_maker primitives so the config is identical to
// what clus.jsonnet used to build; reads raw (non-t0-corrected) blob coords, so
// drift_speed/time_offset/tick MUST match the BlobSampler.
local lbl_dv = clus_maker.detector_volumes(tools.anodes, face='');
local lbl_pcts = clus_maker.pc_transforms(lbl_dv);
local lbl_fv = clus_maker.fiducial_box();
local labeler = g.pnode({
    type: 'wclsTensorSetLabeler',
    name: 'clus_all_apa',
    data: {
        inpath: 'pointtrees/%d',
        grouping: 'live',
        reality: reality,
        anodes: [wc.tn(anode) for anode in tools.anodes],
        detector_volumes: wc.tn(lbl_dv),
        pc_transforms: wc.tn(lbl_pcts),
        drift_speed: clus_maker.drift_speed,
        time_offset: clus_maker.time_offset,
        tick: 0.5 * wc.us,
        // shift the priorSCE (true) depos true->reco before association.
        sce_field: wc.tn(clus_maker.sce_field_fwd),
        sce_correction: true,
        n_sample_truth_depo_sce: 1,
        pf_ke_min: 10 * wc.MeV,
        pf_fiducial: wc.tn(lbl_fv),
        pf_nu_only: true,
        truth_tracks_nu_only: true,
        bee_sink: wc.tn(bee_shared),
        // tagger_stm/tgm/fc Bee sets use the SAME corrected coords as
        // clustering_global (data x_t0cor/y_cor/z_cor, sim x_sce/y_sce/z_sce).
        tagger_coords: clus_maker.bee_coords,
        // beam gate for the 4-case tagger cluster_id (0 non-main, 1 out-of-window
        // main, 2 in-window untagged, 3 in-window tagged); matches pr's window.
        beam_window: beam_window,
    },
}, nin=1, nout=1,
   uses=tools.anodes + [lbl_dv, lbl_pcts, clus_maker.sce_field_fwd, lbl_fv, bee_shared]);

// Terminal sink for the labeled pctree tarball (dump_mode=false -> real tensors,
// the historical labeler deliverable).  Bee -> mabc.zip and nugraph HDF5 are
// written upstream by the MABC/labeler, so this tarball is just a byproduct.
local tail_dump = g.pnode({
    type: 'TensorFileSink',
    name: 'clus_all_apa',
    data: {
        outname: 'trash-all-apa.tar.gz',
        prefix: 'clustering_',
        dump_mode: false,
    },
}, nin=1, nout=0);

// ---- assemble the graph ----
//  sigs ─FrameFanout─┬─ img[0] ─(live,dead)─ clus[0] ─ flash_attach[0] ─┐
//                    └─ img[1] ─(live,dead)─ clus[1] ─ flash_attach[1] ─┤
//  opflash[n] ───────────────────────────────────────(port1) flash_attach[n]
//        flash_attach[n] ─(port n)─ matching_joint ─ clus_all_apa (all-APA MABC)
//        clus_all_apa ─ pr_node (taggers) ─ labeler ─ tail_dump
local graph = g.intern(
    innodes=[wcls_input.sigs],
    centernodes=[frame_fan] + img_pipes + clus_pipes + flash_attach
                + [matching_joint] + wcls_input.opflashes
                + [clus_all_apa, pr_node, labeler],
    outnodes=[tail_dump],
    edges=
        [g.edge(wcls_input.sigs, frame_fan, 0, 0)]
        + [g.edge(frame_fan, img_pipes[n], n, 0) for n in std.range(0, nanode - 1)]
        + [g.edge(img_pipes[n], clus_pipes[n], 0, 0) for n in std.range(0, nanode - 1)]   // live / active
        + [g.edge(img_pipes[n], clus_pipes[n], 1, 1) for n in std.range(0, nanode - 1)]   // dead / masked
        + [g.edge(clus_pipes[n], flash_attach[n], 0, 0) for n in std.range(0, nanode - 1)]
        + [g.edge(wcls_input.opflashes[n], flash_attach[n], 0, 1) for n in std.range(0, nanode - 1)]
        + [g.edge(flash_attach[n], matching_joint, 0, n) for n in std.range(0, nanode - 1)]
        + [g.edge(matching_joint, clus_all_apa, 0, 0)]
        + [g.edge(clus_all_apa, pr_node, 0, 0)]
        + [g.edge(pr_node, labeler, 0, 0)]
        + [g.edge(labeler, tail_dump, 0, 0)]
);

local app = {
  type: 'Pgrapher',
  data: { edges: g.edges(graph) },
};

local cmdline = {
    type: "wire-cell",
    data: {
        // wcls reads plugins from the FCL; this list is for standalone use.
        plugins: ["WireCellGen", "WireCellPgraph", "WireCellAux", "WireCellSio",
                  "WireCellSigProc", "WireCellImg", "WireCellRoot", "WireCellTbb",
                  "WireCellClus", "WireCellMatch"],
        apps: ["Pgrapher"]
    }
};

[cmdline] + g.uses(graph) + cathode_fv.configs + [app]
