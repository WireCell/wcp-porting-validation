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

// run_labeler: append wclsTensorSetLabeler after the all-APA MABC.  Runs in
// BOTH realities: sim attaches run/subrun/event + nu truth metadata, a
// truth_per_track tensor, per-blob "trackid", the truth Bee sets, and a
// nugraph HDF5 WITH truth; data attaches only run/subrun/event metadata and
// an input-only nugraph HDF5 (no truth, no Bee).  Requires the "WireCellAIML"
// plugin + "wclsTensorSetLabeler" inputer in the fcl (both fcls have them).
local run_labeler = true;

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
local clus_maker = clus(rse_from_ident=true, reality=reality, run_labeler=run_labeler);
// Single shared Bee sink: every MABC node (per-APA + all-APA) writes into this
// one zip instead of one zip per node.
local bee_shared = {
    type: 'BeeSink',
    name: 'mabc_shared',
    data: { outname: 'mabc.zip', initial_index: 0 },
};
local clus_pipes = [
    clus_maker.per_apa(tools.anodes[n], dump=false, bee_sink=bee_shared)
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

// all-APA clustering: joint matcher already merged -> premerged=true.
// dump=true appends the terminal TensorFileSink that consumes the all-APA MABC
// output (without it the MABC output port is dangling -> "graph not fully
// connected").  Bee still goes to the shared mabc.zip; the tensor .tar.gz is a
// harmless byproduct (same as Xin's standalone).
local clus_all_apa = clus_maker.all_apa(tools.anodes, dump=true,
                                        bee_sink=bee_shared, premerged=true);

// ---- assemble the graph ----
//  sigs ─FrameFanout─┬─ img[0] ─(live,dead)─ clus[0] ─ flash_attach[0] ─┐
//                    └─ img[1] ─(live,dead)─ clus[1] ─ flash_attach[1] ─┤
//  opflash[n] ───────────────────────────────────────(port1) flash_attach[n]
//        flash_attach[n] ─(port n)─ matching_joint ─ clus_all_apa (all-APA MABC)
local graph = g.intern(
    innodes=[wcls_input.sigs],
    centernodes=[frame_fan] + img_pipes + clus_pipes + flash_attach
                + [matching_joint] + wcls_input.opflashes,
    outnodes=[clus_all_apa],
    edges=
        [g.edge(wcls_input.sigs, frame_fan, 0, 0)]
        + [g.edge(frame_fan, img_pipes[n], n, 0) for n in std.range(0, nanode - 1)]
        + [g.edge(img_pipes[n], clus_pipes[n], 0, 0) for n in std.range(0, nanode - 1)]   // live / active
        + [g.edge(img_pipes[n], clus_pipes[n], 1, 1) for n in std.range(0, nanode - 1)]   // dead / masked
        + [g.edge(clus_pipes[n], flash_attach[n], 0, 0) for n in std.range(0, nanode - 1)]
        + [g.edge(wcls_input.opflashes[n], flash_attach[n], 0, 1) for n in std.range(0, nanode - 1)]
        + [g.edge(flash_attach[n], matching_joint, 0, n) for n in std.range(0, nanode - 1)]
        + [g.edge(matching_joint, clus_all_apa, 0, 0)]
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
