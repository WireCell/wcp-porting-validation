// Standalone (no LArSoft/WCLS) wire-cell config that reproduces the 1-step
// wcls-img-clus-matching-xin.jsonnet clustering + Q/L matching, reading the
// DUMPED image clusters + opflash instead of art products.  Kept in sync with
// sbnd/wcls-img-clus-matching-xin.jsonnet: same toolkit clustering
// (pgrapher/experiment/sbnd/clus.jsonnet) and the SAME joint QLMatching
// (FlashTensorToOpticalPCs + qlmatching.jsonnet matching_joint, premerged
// all-APA MABC).  Only the sources differ (file sources here vs art sources
// in the wcls chain), and there is no wclsTensorSetLabeler (larwirecell-only,
// needs the art event).
//
// Inputs (produced by the step-1 dumps, see docs_for_xin/2-*.md):
//   <input>/icluster-apa{0,1}-active.npz  (wcls-img-dump[-data].fcl)
//   <input>/icluster-apa{0,1}-masked.npz
//   <input>/opflash_apa{0,1}.tar.gz       (wcls-flash-dump[-data].fcl)
//
// Run (SL7 + setup-ap.sh so the toolkit cfg + photodet semimodel resolve):
//   wire-cell -l stdout -L info \
//     -V reality=data \
//     -V input=<dir with icluster/opflash> \
//     -c wct-clus-matching-standalone.jsonnet
//
// reality picks the SAME grouped reco config as the 1-step (sim -> use_sce=true,
// pos_offset_on=false; data -> use_sce=false, pos_offset_on=true).

local g = import "pgraph.jsonnet";
local wc = import "wirecell.jsonnet";
local tools_maker = import 'pgrapher/common/tools.jsonnet';

local reality = std.extVar('reality');
local input = std.extVar('input');

// Canonical SBND simparams -- NO DL/DT/lifetime/driftSpeed overrides, so the
// charge/light model matches the 1-step chain exactly.
local params = import 'pgrapher/experiment/sbnd/simparams.jsonnet';

local tools_all = tools_maker(params);
local tools = tools_all { anodes: [tools_all.anodes[n] for n in [0, 1]] };
local nanode = std.length(tools.anodes);

// ---- clustering: toolkit clus maker, SAME as the 1-step ----
local clus = import 'pgrapher/experiment/sbnd/clus.jsonnet';
// run_labeler=false: the labeler is a larwirecell (wcls) component and cannot
// run standalone (no art event); the 1-step's reco Bee sets are unaffected.
local clus_maker = clus(rse_from_ident=true, reality=reality, run_labeler=false);

// Single shared Bee sink -> one mabc.zip (per-APA + all-APA), as the 1-step.
local bee_shared = {
    type: 'BeeSink',
    name: 'mabc_shared',
    data: { outname: 'mabc.zip', initial_index: 0 },
};

// ---- sources: dumped image clusters (active/masked) + dumped opflash ----
local ClusterFileSource(fname) = g.pnode({
    type: "ClusterFileSource",
    name: fname,
    data: { inname: fname, anodes: [wc.tn(a) for a in tools.anodes] },
}, nin=0, nout=1, uses=tools.anodes);

local active_clusters = [
    ClusterFileSource("%s/icluster-apa%d-active.npz" % [input, tools.anodes[n].data.ident])
    for n in std.range(0, nanode - 1)
];
local masked_clusters = [
    ClusterFileSource("%s/icluster-apa%d-masked.npz" % [input, tools.anodes[n].data.ident])
    for n in std.range(0, nanode - 1)
];
local opflash_sources = [
    g.pnode({
        type: "TensorFileSource",
        name: "opflash_src_apa%d" % n,
        data: { inname: "%s/opflash_apa%d.tar.gz" % [input, n], prefix: "opflash_" },
    }, nin=0, nout=1)
    for n in std.range(0, nanode - 1)
];

local clus_pipes = [
    clus_maker.per_apa(tools.anodes[n], dump=false, bee_sink=bee_shared)
    for n in std.range(0, nanode - 1)
];

// ---- Q/L matching: canonical FlashTensorToOpticalPCs + joint QLMatching,
// IDENTICAL to the 1-step chain. ----
local semimodel_file = 'semi-analytical-sbnd.json';
local pmt_nl = true;
local qlm = (import 'pgrapher/experiment/sbnd/qlmatching.jsonnet')(params);
local cathode_fv = (import 'pgrapher/experiment/sbnd/cathode_fiducial.jsonnet')(
    cx=0.5*wc.cm, cy=0.5*wc.cm, cz=0.5*wc.cm);
local flash_attach = [qlm.flash_attach(n) for n in std.range(0, nanode - 1)];
local matching_joint = qlm.matching_joint(
    tools.anodes, clus_maker.detector_volumes(tools.anodes),
    reality, semimodel_file, cathode_fiducial=cathode_fv.tn, pmt_nl=pmt_nl);

// all-APA clustering: joint matcher already merged -> premerged=true.
local clus_all_apa = clus_maker.all_apa(tools.anodes, dump=true,
                                        bee_sink=bee_shared, premerged=true);

// ---- assemble the graph (mirror of the 1-step, file sources in place of art):
//  icluster-active[n] ─(port0)─┐
//  icluster-masked[n] ─(port1)─┴ clus[n] ─(port0)─ flash_attach[n] ─┐
//  opflash[n] ────────────────────────────(port1) flash_attach[n]  │
//        flash_attach[n] ─(port n)─ matching_joint ─ clus_all_apa (MABC)
local graph = g.intern(
    innodes=active_clusters + masked_clusters + opflash_sources,
    centernodes=clus_pipes + flash_attach + [matching_joint],
    outnodes=[clus_all_apa],
    edges=
        [g.edge(active_clusters[n], clus_pipes[n], 0, 0) for n in std.range(0, nanode - 1)]
        + [g.edge(masked_clusters[n], clus_pipes[n], 0, 1) for n in std.range(0, nanode - 1)]
        + [g.edge(clus_pipes[n], flash_attach[n], 0, 0) for n in std.range(0, nanode - 1)]
        + [g.edge(opflash_sources[n], flash_attach[n], 0, 1) for n in std.range(0, nanode - 1)]
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
        // WireCellAux: FlashTensorToOpticalPCs; WireCellMatch: QLMatching.
        plugins: ["WireCellGen", "WireCellPgraph", "WireCellSio", "WireCellSigProc",
                  "WireCellImg", "WireCellRoot", "WireCellTbb", "WireCellClus",
                  "WireCellMatch", "WireCellAux"],
        apps: ["Pgrapher"]
    }
};

// cathode_fv.configs: the cpa-exclusion CompositeFiducial + BoxFiducials are
// referenced by tn (not via graph `uses`), so append them explicitly -- same
// as the 1-step entry jsonnet's final line.
[cmdline] + g.uses(graph) + cathode_fv.configs + [app]
