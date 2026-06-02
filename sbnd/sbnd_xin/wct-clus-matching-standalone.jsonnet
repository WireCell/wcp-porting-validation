// Standalone SBND charge-light (Q/L) matching chain. Self-contained under
// sbnd_xin: the clustering graph comes from the local clus.jsonnet (a thin
// re-export of the in-tree canonical pgrapher/experiment/sbnd/clus.jsonnet),
// NOT yuhw's ../clus.jsonnet. The in-tree all_apa is the pre-tagging chain (no
// neutrino-tagging downstream), so no tagger_info handoff is required.
//
// Run via sbnd_xin/run_clust_QL_evt.sh.

local g = import "pgraph.jsonnet";
local f = import "pgrapher/common/funcs.jsonnet";
local wc = import "wirecell.jsonnet";

local tools_maker = import 'pgrapher/common/tools.jsonnet';

local reality = std.extVar('reality');
local semimodel_file = std.extVar('semimodel_file');

local base = import 'pgrapher/experiment/sbnd/simparams.jsonnet';
local params = base {
  lar: super.lar {
    DL: std.extVar('DL') * wc.cm2 / wc.s,
    DT: std.extVar('DT') * wc.cm2 / wc.s,
    lifetime: std.extVar('lifetime') * wc.ms,
    // drift_speed is taken from the common SBND config (simparams.jsonnet:
    // 1.563 mm/us), not overridden here, so charge and light share one value.
  },
};

local tools_all = tools_maker(params);
local tools = tools_all { anodes: [tools_all.anodes[n] for n in [0, 1]] };

local input = std.extVar('input');

// --- Sources ---
local ClusterFileSource(fname) = g.pnode({
    type: "ClusterFileSource",
    name: fname,
    data: {
        inname: fname,
        anodes: [wc.tn(a) for a in tools.anodes],
    }
}, nin=0, nout=1, uses=tools.anodes);

local active_clusters = [
    ClusterFileSource("%s/icluster-apa%d-active.npz" % [input, tools.anodes[n].data.ident])
    for n in std.range(0, std.length(tools.anodes) - 1)
];

// --- Dead (masked) clusters: imaged in-toolkit from the SP frames ---
// The dead pctree (/dead, port 1 of PointTreeBuilding) must be populated with the
// toolkit's 2-view dead blobs (multi_masked_2view_slicing_tiling), NOT yuhw's
// larsoft 1-view masked.npz.  We run only the masked imaging fork here.  The live
// (active) clusters above are also toolkit-imaged (run_clust_QL_evt.sh runs
// wct-img-all.jsonnet on the same SP-frame bundle before this graph), so both
// live and dead views come from the toolkit's multi-3view imaging -- this recovers
// charge across W-plane dead channels that yuhw's larsoft active npz omitted.  A
// single FrameFileSource feeds a FrameFanout that renames the 'dnnsp' tag into the
// per-anode 'gauss<id>'/'wiener<id>' tags the imaging pre_proc expects (mirrors
// wct-img-all.jsonnet).  `frames` is the combined 10-event sp-frames archive.
local frames = std.extVar('frames');
local img = import 'pgrapher/experiment/sbnd/img.jsonnet';
local img_maker = img();
local masked_img = [
    img_maker.per_anode(tools.anodes[n], 'masked', add_dump=false)
    for n in std.range(0, std.length(tools.anodes) - 1)
];
local fanout_rules = [
    {
        frame: { '.*': 'orig%d' % tools.anodes[n].data.ident },
        trace: { dnnsp: ['gauss%d' % tools.anodes[n].data.ident, 'wiener%d' % tools.anodes[n].data.ident] },
    }
    for n in std.range(0, std.length(tools.anodes) - 1)
];
local frame_src = g.pnode({
    type: 'FrameFileSource',
    name: 'qlframes',
    data: { inname: frames, tags: ['dnnsp'] },
}, nin=0, nout=1);
local frame_fan = g.pnode({
    type: 'FrameFanout',
    name: 'qlframe_fanout',
    data: { multiplicity: std.length(tools.anodes), tag_rules: fanout_rules },
}, nin=1, nout=std.length(tools.anodes));

// --- Per-APA clustering ---
// Local re-export of the in-tree canonical pgrapher/experiment/sbnd/clus.jsonnet
// (keeps sbnd_xin standalone; no dependency on ../sbnd).
local clus = import 'clus.jsonnet';
local clus_maker = clus();
// Single shared Bee sink: the per-APA and all-APA MultiAlgBlobClustering nodes
// all write into this one zip (mabc.zip) instead of one zip per node, so the
// run produces a single self-contained Bee file with every view.
local bee_shared = {
    type: 'BeeSink',
    name: 'mabc_shared',
    data: { outname: 'mabc.zip', initial_index: 0 },
};
local clus_pipes = [
    clus_maker.per_apa(tools.anodes[n], dump=false, bee_sink=bee_shared)
    for n in std.range(0, std.length(tools.anodes) - 1)
];

// --- SBND light I/O + QL matching ---
// All matching graph nodes (TensorFileSource opflash reader -> FlashTensorToOpticalPCs
// canonical-flash-PC fan-in -> QLMatching) plus the SBND matching constants live in
// the canonical in-tree helper, re-exported locally as ./qlmatching.jsonnet (same
// shim pattern as ./clus.jsonnet).
local qlm = (import 'qlmatching.jsonnet')(params);
// SBND CPA structure-exclusion fiducial (cushion is the live tuning knob).
local cathode_fv = (import 'cathode_fiducial.jsonnet')(cx=0.5*wc.cm, cy=0.5*wc.cm, cz=0.5*wc.cm);
local opflash_sources = [qlm.opflash_source(n) for n in std.range(0, std.length(tools.anodes) - 1)];
local flash_attach    = [qlm.flash_attach(n)    for n in std.range(0, std.length(tools.anodes) - 1)];
local matching_pipes  = [
    qlm.matching(tools.anodes[n], clus_maker.detector_volumes([tools.anodes[n]]),
                 n, reality, semimodel_file, cathode_fiducial=cathode_fv.tn)
    for n in std.range(0, std.length(tools.anodes) - 1)
];

// --- All-APA clustering ---
// In-tree all_apa is the pre-tagging chain (no nu_tagging param; see header).
local clus_all_apa = clus_maker.all_apa(tools.anodes, dump=true, bee_sink=bee_shared);

// --- Flat graph ---
// Per anode n:
//   active ClusterFileSource ───────────────┐(port 0, /live)
//   frame -> FrameFanout -> masked imaging ──┤(port 1, /dead)  -> PointTreeBuilding
//                                            └─> clustering -> FlashTensorToOpticalPCs
//   opflash TensorFileSource ──────────────────────────────┘(port 1) -> QLMatching
//   ... -> all-APA MultiAlgBlobClustering.  All MABC nodes (per-APA + all-APA)
//   write into one shared Bee zip (mabc.zip) via the bee_shared sink.
local nanode = std.length(tools.anodes);
local graph = g.intern(
    innodes=active_clusters + opflash_sources + [frame_src],
    centernodes=masked_img + clus_pipes + flash_attach + matching_pipes + [frame_fan],
    outnodes=[clus_all_apa],
    edges=
        [g.edge(frame_src, frame_fan, 0, 0)]
        + [g.edge(frame_fan, masked_img[n], n, 0) for n in std.range(0, nanode - 1)]
        + [g.edge(active_clusters[n], clus_pipes[n], 0, 0) for n in std.range(0, nanode - 1)]
        + [g.edge(masked_img[n], clus_pipes[n], 0, 1) for n in std.range(0, nanode - 1)]
        + [g.edge(clus_pipes[n], flash_attach[n], 0, 0) for n in std.range(0, nanode - 1)]
        + [g.edge(opflash_sources[n], flash_attach[n], 0, 1) for n in std.range(0, nanode - 1)]
        + [g.edge(flash_attach[n], matching_pipes[n], 0, 0) for n in std.range(0, nanode - 1)]
        + [g.edge(matching_pipes[n], clus_all_apa, 0, n) for n in std.range(0, nanode - 1)]
);

local app = {
  type: 'Pgrapher',
  data: { edges: g.edges(graph) },
};

local cmdline = {
    type: "wire-cell",
    data: {
        plugins: ["WireCellGen", "WireCellPgraph", "WireCellAux", "WireCellSio",
                  "WireCellSigProc", "WireCellImg", "WireCellRoot", "WireCellTbb",
                  "WireCellClus", "WireCellMatch"],
        apps: ["Pgrapher"]
    }
};

[cmdline] + g.uses(graph) + cathode_fv.configs + [app]
