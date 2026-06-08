// wcls imaging + clustering + Q/L matching, using Xin's canonical WCT-native
// matching (FlashTensorToOpticalPCs + QLMatching from WireCellMatch) instead of
// the old larwirecell wclsQLMatching node.
//
// Derived from wcls-img-clus-matching.jsonnet; ONLY the matching stage changed.
// Unlike sbnd_xin/wct-clus-matching-standalone.jsonnet (which reads dumped npz
// clusters + opflash/frame archives), this stays a wcls job reading the artROOT
// directly: charge via wclsCookedFrameSource, light via wclsOpFlashSource.  The
// opflash source already emits the [nflashes x (nchan+1)] matrix tensorset that
// FlashTensorToOpticalPCs expects, so it feeds flash_attach port 1 with no file
// round-trip.
//
// Matching is positioned exactly as in Xin's chain: per-APA clustering ->
// flash_attach (+opflash) -> QLMatching -> clus_all_apa (all-APA merge).
//
// The matching helpers come from the canonical in-tree modules
// (pgrapher/experiment/sbnd/{qlmatching,cathode_fiducial}.jsonnet); nothing in
// sbnd_xin is imported or modified.

local g = import "pgraph.jsonnet";
local f = import "pgrapher/common/funcs.jsonnet";
local wc = import "wirecell.jsonnet";

local tools_maker = import 'pgrapher/common/tools.jsonnet';

local reality = std.extVar('reality');

// Use the canonical SBND simparams as-is.  The fcl does NOT provide
// DL/DT/lifetime/driftSpeed extVars; the old jsonnet referenced them in a lar{}
// override but jsonnet laziness meant they were never forced (nothing read
// params.lar.*).  The canonical qlmatching DOES read params.lar.drift_speed, so
// we must give it a concrete value -- the common SBND drift_speed (1.563 mm/us)
// from simparams, which is exactly what Xin's standalone chain uses too.
local base = import 'pgrapher/experiment/sbnd/simparams.jsonnet';
local params = base;


local tools_all = tools_maker(params);
local tools = tools_all {anodes: [tools_all.anodes[n] for n in [0,1]]};

// must match name used in fcl
local wcls_input = {
    sigs: g.pnode({
        type: 'wclsCookedFrameSource',
        name: 'sigs',
        data: {
            nticks: params.daq.nticks,
            frame_scale: 50,                             // scale up input recob::Wire by this factor
            summary_scale: 50,                             // scale up input summary by this factor
            frame_tags: ["orig"],                 // frame tags (only one frame in this module)
            recobwire_tags: std.extVar('recobwire_tags'), // ["sptpc2d:gauss", "sptpc2d:wiener"],
            trace_tags: std.extVar('trace_tags'), // ["gauss", "wiener"],
            summary_tags: std.extVar('summary_tags'), // ["", "sptpc2d:wienersummary"],
            input_mask_tags: std.extVar('input_mask_tags'), // ["sptpc2d:badmasks"],
            output_mask_tags: std.extVar('output_mask_tags'), // ["bad"],
        },
    }, nin=0, nout=1),
  // wclsOpFlashSource emits an ITensorSet: a [nflashes x (npmts+1)] matrix
  // (npmts=312 -> ncol=313), exactly the opflash matrix FlashTensorToOpticalPCs
  // consumes on its port 1.  So no opflash file dump/read is needed here.
  opflashes0: g.pnode({
    type: 'wclsOpFlashSource',
    name: 'tpc0',
    data: {
      art_tag: std.extVar('opflash0_input_label'),
    },
  }, nin=0, nout=1),
  opflashes1: g.pnode({
    type: 'wclsOpFlashSource',
    name: 'tpc1',
    data: {
      art_tag: std.extVar('opflash1_input_label'),
    },
  }, nin=0, nout=1),
  opflashes: [wcls_input.opflashes0, wcls_input.opflashes1],
};

local img = import 'pgrapher/experiment/sbnd/img.jsonnet';
local img_maker = img();
local img_config = std.extVar('img_config');
local img_pipes = [img_maker.per_anode(a, img_config, add_dump = false,
                                       channels_per_apa = 5638 /* needs to be consistent with geom file */) for a in tools.anodes];

local clus = import 'pgrapher/experiment/sbnd/clus.jsonnet';
local clus_maker = clus();
local clus_pipes = [clus_maker.per_apa(anode, dump=false) for anode in tools.anodes];

// img (live port 0, dead port 1) -> per-APA clustering -> post-MABC cluster tree.
local img_clus_pipe = [g.intern(
    innodes = [img_pipes[n]],
    centernodes = [],
    outnodes = [clus_pipes[n]],
    edges = [
        g.edge(img_pipes[n], clus_pipes[n], p, p)
        for p in std.range(0, 1)
    ]
)
for n in std.range(0, std.length(tools.anodes) - 1)];

// ---- Q/L matching (Xin's canonical WCT-native path) ----
// semi-analytical photo-detector model (resolved by bare name via WIRECELL_PATH
// /exp/.../wire-cell-data/sbnd/photodet; see setup-ap.sh).
local semimodel_file = 'semi-analytical-sbnd.json';
// Per-PMT predicted-PE non-linearity correction (SBND-on, as in Xin's chain).
local pmt_nl = true;

local qlm = (import 'pgrapher/experiment/sbnd/qlmatching.jsonnet')(params);
// SBND CPA structure-exclusion fiducial (cushion is the live tuning knob); its
// component configs must be injected at the top level of the job (below).
local cathode_fv = (import 'pgrapher/experiment/sbnd/cathode_fiducial.jsonnet')(
    cx=0.5*wc.cm, cy=0.5*wc.cm, cz=0.5*wc.cm);

// FlashTensorToOpticalPCs: cluster tree (port 0) + opflash matrix (port 1)
// -> cluster tree with canonical flash/light/flashlight PCs attached.
local flash_attach = [qlm.flash_attach(n) for n in std.range(0, std.length(tools.anodes) - 1)];

// QLMatching: reads the flash PCs, writes back the per-cluster matched-flash
// scalar.  Per-APA (one node per anode), feeding the all-APA merge.
local matching_nodes = [
    qlm.matching(tools.anodes[n], clus_maker.detector_volumes([tools.anodes[n]]),
                 n, reality, semimodel_file, cathode_fiducial=cathode_fv.tn, pmt_nl=pmt_nl)
    for n in std.range(0, std.length(tools.anodes) - 1)
];

// Per-APA matching subgraph: charge tree from img_clus_pipe (innode, fed by the
// frame fanout), opflash source (light), flash_attach, QLMatching (outnode).
local matching_pipes = [
    g.intern(
        innodes = [img_clus_pipe[n]],
        outnodes = [matching_nodes[n]],
        centernodes = [wcls_input.opflashes[n], flash_attach[n]],
        edges = [
            g.edge(img_clus_pipe[n], flash_attach[n], 0, 0),
            g.edge(wcls_input.opflashes[n], flash_attach[n], 0, 1),
            g.edge(flash_attach[n], matching_nodes[n], 0, 0),
        ])
    for n in std.range(0, std.length(tools.anodes) - 1)
];

local fanout_apa_rules =
[
    {
        frame: {
            '.*': 'orig%d' % n,
        },
        trace: {
            gauss: 'gauss%d' % n,
            wiener: 'wiener%d' % n,
        },
    }
    for n in std.range(0, std.length(tools.anodes) - 1)
];

local magoutput = 'mag.root';
local magnify = import 'pgrapher/experiment/sbnd/magnify-sinks.jsonnet';
local magnify_sinks = magnify(tools, magoutput);
local prosessing_pipes = [
    g.pipeline([
        // magnify_sinks.decon_pipe[n],
        matching_pipes[n]], "prosessing_pipes%d" % n)
    for n in std.range(0, std.length(tools.anodes) - 1)
];

local img_clus_per_apa = f.fanout("FrameFanout", prosessing_pipes, "img_clus_per_apa", fanout_apa_rules);

// All-APA clustering: merges the per-APA matched cluster trees (PointTreeMerging
// + all-APA MultiAlgBlobClustering).  Per-APA matching path => premerged=false.
local clus_all_apa = clus_maker.all_apa(tools.anodes);

local graph = g.intern(
    innodes=[wcls_input.sigs],
    centernodes = [img_clus_per_apa],
    outnodes=[clus_all_apa],
    edges=
    [g.edge(wcls_input.sigs, img_clus_per_apa, 0, 0)] +
    [g.edge(img_clus_per_apa, clus_all_apa, i, i) for i in std.range(0, std.length(tools.anodes) - 1)]
);

local app = {
  type: 'Pgrapher', //Pgrapher, TbbFlow
  data: {
    edges: g.edges(graph),
  },
};

local cmdline = {
    type: "wire-cell",
    data: {
        // (wcls reads plugins from the FCL; this list is for standalone use.)
        plugins: ["WireCellGen", "WireCellPgraph", "WireCellAux", "WireCellSio",
                  "WireCellSigProc", "WireCellRoot", "WireCellTbb", "WireCellImg",
                  "WireCellClus", "WireCellMatch"],
        apps: ["Pgrapher"] //TbbFlow
    }
};

[cmdline] + g.uses(graph) + cathode_fv.configs + [app]
