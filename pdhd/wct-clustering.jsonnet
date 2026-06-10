local g = import "pgraph.jsonnet";
local f = import "pgrapher/common/funcs.jsonnet";
local wc = import "wirecell.jsonnet";

local io = import 'pgrapher/common/fileio.jsonnet';
local tools_maker = import 'pgrapher/common/tools.jsonnet';

local reality = 'data';
local data_params = import 'pgrapher/experiment/pdhd/params.jsonnet';
local simu_params = import 'pgrapher/experiment/pdhd/simparams.jsonnet';
local params = if reality == 'data' then data_params else simu_params;
local tools_all = tools_maker(params);

// Top-level function: parameters overridable via --tla-str / --tla-code
function(
    input = ".",
    // Indices into tools_all.anodes to process; default = all
    anode_indices = std.range(0, std.length(tools_all.anodes) - 1),
    // Directory for output mabc-*.zip and bee data files ('' means current directory)
    output_dir = '',
    run = 1,
    subrun = 1,
    event = 1,
    // Event-T0 / readout-tick0 compensation for the live BlobSampler drift-x
    // conversion (WCT units, ns).  PDHD has no per-event T0, so default 0 (no
    // preset T0).  -250us (-250000) would place trigger-time activity at its
    // true x; see pgrapher/experiment/pdhd/clus.jsonnet.
    time_offset = 0,
)

local anodes = [tools_all.anodes[i] for i in anode_indices];

local cluster_source(fname) = g.pnode({
    type: "ClusterFileSource",
    name: fname,
    data: {
        inname: fname,
        anodes: [wc.tn(a) for a in anodes],
    }
}, nin=0, nout=1, uses=anodes);

local clus = import 'pgrapher/experiment/pdhd/clus.jsonnet';
local clus_maker = clus(output_dir=output_dir, runNo=run, subRunNo=subrun, eventNo=event,
                        time_offset=time_offset);

// Drift-side groups: x-aligned APAs viewed through a COMMON face, i.e. one
// drift volume: even idents (APA0+APA2) image through face 0 (drift -x), odd
// (APA1+APA3) through face 1 (drift +x).  The opposite faces are PDHD wall
// faces that do not image, so each drift side has exactly one populated
// group.  With a subset anode_indices only non-empty groups are built and
// the final merge multiplicity shrinks to match.
local group_defs = [
    { name: "group02", face: 0, anodes: [a for a in anodes if a.data.ident % 2 == 0] },
    { name: "group13", face: 1, anodes: [a for a in anodes if a.data.ident % 2 == 1] },
];
local groups = [gd for gd in group_defs if std.length(gd.anodes) > 0];
local ngroups = std.length(groups);

// One drift-side pipe: per-APA sources -> per_apa (stages 1+2) -> per_group (stage 3).
local group_pipe(gd) =
    local n = std.length(gd.anodes);
    local actives = [cluster_source("%s/clusters-apa-apa%d-ms-active.tar.gz"%[input, a.data.ident]) for a in gd.anodes];
    local maskeds = [cluster_source("%s/clusters-apa-apa%d-ms-masked.tar.gz"%[input, a.data.ident]) for a in gd.anodes];
    local apa_pipes = [clus_maker.per_apa(gd.anodes[i], dump=false) for i in std.range(0, n - 1)];
    local pg = clus_maker.per_group(gd.anodes, gd.name, gd.face, dump=false);
    g.intern(
        innodes = actives + maskeds,
        centernodes = apa_pipes,
        outnodes = [pg],
        edges =
            [g.edge(actives[i], apa_pipes[i], 0, 0) for i in std.range(0, n - 1)] +
            [g.edge(maskeds[i], apa_pipes[i], 0, 1) for i in std.range(0, n - 1)] +
            [g.edge(apa_pipes[i], pg, 0, i) for i in std.range(0, n - 1)]
    );

local group_pipes = [group_pipe(gd) for gd in groups];
local clus_all_tpc = clus_maker.all_tpc(anodes, ngroups=ngroups);

local graph = g.intern(
    innodes=group_pipes,
    outnodes=[clus_all_tpc],
    edges=[g.edge(group_pipes[i], clus_all_tpc, 0, i) for i in std.range(0, ngroups - 1)]
);

local app = {
  type: 'Pgrapher',
  data: {
    edges: g.edges(graph),
  },
};

local cmdline = {
    type: "wire-cell",
    data: {
        plugins: ["WireCellGen", "WireCellPgraph", "WireCellSio", "WireCellSigProc", "WireCellImg", "WireCellRoot", "WireCellTbb", "WireCellClus"],
        apps: ["Pgrapher"]
    }
};

[cmdline] + g.uses(graph) + [app]
