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
local masked_clusters = [
    ClusterFileSource("%s/icluster-apa%d-masked.npz" % [input, tools.anodes[n].data.ident])
    for n in std.range(0, std.length(tools.anodes) - 1)
];

// --- Per-APA clustering ---
// Local re-export of the in-tree canonical pgrapher/experiment/sbnd/clus.jsonnet
// (keeps sbnd_xin standalone; no dependency on ../sbnd).
local clus = import 'clus.jsonnet';
local clus_maker = clus();
local clus_pipes = [
    clus_maker.per_apa(tools.anodes[n], dump=false)
    for n in std.range(0, std.length(tools.anodes) - 1)
];

// --- SBND light I/O + QL matching ---
// All matching graph nodes (TensorFileSource opflash reader -> FlashTensorToOpticalPCs
// canonical-flash-PC fan-in -> QLMatching) plus the SBND matching constants live in
// the canonical in-tree helper, re-exported locally as ./qlmatching.jsonnet (same
// shim pattern as ./clus.jsonnet).
local qlm = (import 'qlmatching.jsonnet')(params);
local opflash_sources = [qlm.opflash_source(n) for n in std.range(0, std.length(tools.anodes) - 1)];
local flash_attach    = [qlm.flash_attach(n)    for n in std.range(0, std.length(tools.anodes) - 1)];
local matching_pipes  = [
    qlm.matching(tools.anodes[n], clus_maker.detector_volumes([tools.anodes[n]]),
                 n, reality, semimodel_file)
    for n in std.range(0, std.length(tools.anodes) - 1)
];

// --- Per-APA subgraphs ---
// active+masked -> clustering ─┐
//                             ├─ FlashTensorToOpticalPCs (canonical light PCs) -> QLMatching
//          TensorFileSource ──┘
local per_apa = [g.intern(
    innodes=[active_clusters[n], masked_clusters[n], opflash_sources[n]],
    centernodes=[clus_pipes[n], flash_attach[n]],
    outnodes=[matching_pipes[n]],
    edges=[
        g.edge(active_clusters[n], clus_pipes[n], 0, 0),
        g.edge(masked_clusters[n], clus_pipes[n], 0, 1),
        g.edge(clus_pipes[n], flash_attach[n], 0, 0),      // port 0 = pctree
        g.edge(opflash_sources[n], flash_attach[n], 0, 1), // port 1 = opflash
        g.edge(flash_attach[n], matching_pipes[n], 0, 0),
    ]
) for n in std.range(0, std.length(tools.anodes) - 1)];

// --- All-APA clustering ---
// In-tree all_apa is the pre-tagging chain (no nu_tagging param; see header).
local clus_all_apa = clus_maker.all_apa(tools.anodes, dump=true);

local graph = g.intern(
    innodes=per_apa,
    outnodes=[clus_all_apa],
    edges=[g.edge(per_apa[i], clus_all_apa, 0, i)
           for i in std.range(0, std.length(tools.anodes) - 1)]
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

[cmdline] + g.uses(graph) + [app]
