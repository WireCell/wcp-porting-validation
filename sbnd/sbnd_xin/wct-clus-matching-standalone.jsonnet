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
// Joint multi-APA matching toggle (run_clust_QL_evt.sh passes -C joint=...).
// true = one joint QLMatching node matches both APAs and merges, feeding MABC
// directly; false = historical per-APA path (one QLMatching per APA ->
// PointTreeMerging -> MABC).  Both are byte-identical today (the joint algorithm
// is deferred); the joint wiring is the live home for it.
local joint = std.extVar('joint');
// Per-PMT predicted-PE non-linearity correction toggle (run_clust_QL_evt.sh passes
// -C pmt_nl=...; default true there, PMT_NL=false reproduces the OFF baseline). Threaded
// into the local qlmatching.jsonnet wrapper's matching()/matching_joint(); canonical
// production stays OFF.
local pmt_nl = std.extVar('pmt_nl');

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
// rse_from_ident: this bundled chain is one wire-cell call over many events, and
// each event's tensor ident already carries the real event id, so the MABC nodes
// label the Bee display with the true event number (run/subrun = 0) instead of a
// 0..N auto-increment.  Canonical production leaves this off (byte-identical).
// reality gates the data-only pos_offset transverse calibration (data -> on,
// sim -> off); see cfg/.../sbnd/clus.jsonnet pos_offset comment.
local clus_maker = clus(rse_from_ident=true, reality=reality);
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
    // lm / main_flag pinned to the pre-adoption values: the canonical module
    // defaults them ON as of 2026-07-27 (doc 64), and this legacy all-events
    // variant is kept byte-identical to its historical output.  Drop the pins to
    // follow production.
    qlm.matching(tools.anodes[n], clus_maker.detector_volumes([tools.anodes[n]]),
                 n, reality, semimodel_file, cathode_fiducial=cathode_fv.tn, pmt_nl=pmt_nl,
                 lm=false, main_flag=false)
    for n in std.range(0, std.length(tools.anodes) - 1)
];
// Joint matcher: one node fed by both APAs' flash_attach outputs (used iff joint).
local matching_joint = qlm.matching_joint(tools.anodes, clus_maker.detector_volumes(tools.anodes),
                                          reality, semimodel_file, cathode_fiducial=cathode_fv.tn,
                                          pmt_nl=pmt_nl,
                                          lm=false, main_flag=false);  // see matching() above

// --- All-APA clustering ---
// In-tree all_apa is the pre-tagging chain (no nu_tagging param; see header).
// premerged=joint: when joint, the joint matcher already merged the per-APA trees,
// so all_apa skips its PointTreeMerging and feeds the single tree to MABC.
local clus_all_apa = clus_maker.all_apa(tools.anodes, dump=true, bee_sink=bee_shared, premerged=joint);

// --- Flat graph ---
// Per anode n:
//   active ClusterFileSource ───────────────┐(port 0, /live)
//   frame -> FrameFanout -> masked imaging ──┤(port 1, /dead)  -> PointTreeBuilding
//                                            └─> clustering -> FlashTensorToOpticalPCs
//   opflash TensorFileSource ──────────────────────────────┘(port 1) -> QLMatching
//   ... -> all-APA MultiAlgBlobClustering.  All MABC nodes (per-APA + all-APA)
//   write into one shared Bee zip (mabc.zip) via the bee_shared sink.
local nanode = std.length(tools.anodes);
// Matching stage: joint = one matching_joint node (flash_attach[n] -> port n) that
// feeds MABC directly; per-APA = one matcher per APA -> all_apa's PointTreeMerging.
local match_center = if joint then [matching_joint] else matching_pipes;
local match_edges =
    if joint then
        [g.edge(flash_attach[n], matching_joint, 0, n) for n in std.range(0, nanode - 1)]
        + [g.edge(matching_joint, clus_all_apa, 0, 0)]
    else
        [g.edge(flash_attach[n], matching_pipes[n], 0, 0) for n in std.range(0, nanode - 1)]
        + [g.edge(matching_pipes[n], clus_all_apa, 0, n) for n in std.range(0, nanode - 1)];
local graph = g.intern(
    innodes=active_clusters + opflash_sources + [frame_src],
    centernodes=masked_img + clus_pipes + flash_attach + match_center + [frame_fan],
    outnodes=[clus_all_apa],
    edges=
        [g.edge(frame_src, frame_fan, 0, 0)]
        + [g.edge(frame_fan, masked_img[n], n, 0) for n in std.range(0, nanode - 1)]
        + [g.edge(active_clusters[n], clus_pipes[n], 0, 0) for n in std.range(0, nanode - 1)]
        + [g.edge(masked_img[n], clus_pipes[n], 0, 1) for n in std.range(0, nanode - 1)]
        + [g.edge(clus_pipes[n], flash_attach[n], 0, 0) for n in std.range(0, nanode - 1)]
        + [g.edge(opflash_sources[n], flash_attach[n], 0, 1) for n in std.range(0, nanode - 1)]
        + match_edges
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
