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
    drift_speed: std.extVar('driftSpeed') * wc.mm / wc.us,
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

// SBND light I/O: read the per-APA opflash archive and attach it as a "flash"
// point cloud onto the live root node of the cluster pctree (the matcher then
// reads the light from there). TensorFileToPCTree is the generic sio primitive
// for "attach a tensor file as a named root-node PC". Per APA, clustering stage.
local flash_io = [
    g.pnode({
        type: 'TensorFileToPCTree',
        name: 'flash_io_apa%d' % n,
        data: {
            input: "opflash_apa%d.tar.gz" % n,
            prefix: "opflash_",
            pcname: "flash",
        }
    }, nin=1, nout=1)
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

// --- QL Matching (now WCT-native) ---
local matching_pipes = [
    g.pnode({
        type: 'QLMatching',
        name: 'matching%d' % n,
        local dv = clus_maker.detector_volumes([tools.anodes[n]]),
        data: {
            anode: wc.tn(tools.anodes[n]),
            detector_volumes: wc.tn(dv),
            bee_dir: "data-sep",
            beamonly: false,
            data: if reality == 'data' then true else false,
            QtoL: 1.0,
            ch_mask: [39, 64, 66, 71, 85, 86, 87, 115, 138, 141, 197, 217, 221,
                      222, 223, 226, 245, 249, 302],
            flash_minPE: 50,
            semimodel_file: semimodel_file,
        },
    }, nin=1, nout=1)
    for n in std.range(0, std.length(tools.anodes) - 1)
];

// --- Per-APA subgraphs ---
// active+masked -> clustering -> TensorFileToPCTree (attach light) -> QLMatching
local per_apa = [g.intern(
    innodes=[active_clusters[n], masked_clusters[n]],
    centernodes=[clus_pipes[n], flash_io[n]],
    outnodes=[matching_pipes[n]],
    edges=[
        g.edge(active_clusters[n], clus_pipes[n], 0, 0),
        g.edge(masked_clusters[n], clus_pipes[n], 0, 1),
        g.edge(clus_pipes[n], flash_io[n], 0, 0),
        g.edge(flash_io[n], matching_pipes[n], 0, 0),
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
        plugins: ["WireCellGen", "WireCellPgraph", "WireCellSio", "WireCellSigProc",
                  "WireCellImg", "WireCellRoot", "WireCellTbb", "WireCellClus",
                  "WireCellMatch"],
        apps: ["Pgrapher"]
    }
};

[cmdline] + g.uses(graph) + [app]
