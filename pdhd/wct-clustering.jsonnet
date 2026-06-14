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
    // Per-event readout-vs-trigger offset in MICROSECONDS (the opflash metadata
    // offset_us, ~250).  Applied DOWNSTREAM (Q/L matching geometry + T0Correction
    // x_t0cor) rather than baked into x_raw, so the raw imaging x stays offset-free
    // while flash-matched charge lands on the trigger time base.  Default 0 =>
    // bit-identical.  run_clus_evt.sh reads offset_us from the opflash archive.
    trigger_offset_us = 0,
    // Interpose charge-light (Q/L) matching between the per-drift-group
    // clustering (stage 3) and the final all-TPC clustering (stage 4).  Default
    // false => the historical no-matching chain, bit-identical.  Reads the
    // per-event opflash_pdhd-wct.tar.gz; a drift side with no light simply gets
    // no matched flashes.  See pgrapher/experiment/pdhd/qlmatching.jsonnet.
    do_qlmatch = false,
    // Hand-scan calibration dump: when true (and do_qlmatch true) each drift-side
    // QLMatching node writes work/<output_dir>/calib-evt<event>-<group>.json for the
    // pdhd/ql_scan viewer.  Default false => no dump, matching output bit-identical.
    // run_clus_evt.sh -calib sets this (and forces do_qlmatch).
    calib = false,
    // Dump the optical "op" bee instance (measured flash PE + Q/L predicted PE
    // per matched cluster) from the all-TPC MABC, for the Bee event display.
    // Needs do_qlmatch (the QLMatching "opflash" root PC); no-op otherwise.
    // Default false => bit-identical.  run_clus_evt.sh -op sets this (+do_qlmatch).
    save_opflash = false,
    // Post-resample readout-window length (ticks) for the Q/L window-truncation
    // flag.  run_clus_evt.sh reads the real value from the SP frame (~5999) and
    // passes it here; 6000 is a sane PDHD fallback.  See qlmatching.jsonnet.
    readout_window_ticks = 6000,
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

local trigger_offset = trigger_offset_us * wc.us;
local clus = import 'pgrapher/experiment/pdhd/clus.jsonnet';
local clus_maker = clus(output_dir=output_dir, runNo=run, subRunNo=subrun, eventNo=event,
                        time_offset=time_offset, trigger_offset=trigger_offset);

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
local clus_all_tpc = clus_maker.all_tpc(anodes, ngroups=ngroups, save_opflash=save_opflash);

// Q/L matching for one drift-side group: opflash source -> flash_attach (2->1
// fan-in: port0 = group cluster tree, port1 = opflash matrix) -> QLMatching.
// The subgraph's free input is flash_attach port0 (port1 is wired internally),
// its output is the matched cluster tree.  A representative anode of the group
// (gd.anodes[0]) carries the shared drift-side geometry / per-TPC OpDet mask.
local qlm = import 'pgrapher/experiment/pdhd/qlmatching.jsonnet';
local qlm_maker = qlm(params, trigger_offset, readout_window_ticks);
// Per-drift-side hand-scan calibration dump path (empty unless `calib`).  Lands in
// the event workspace beside mabc-*.zip, one file per group (the ql_scan viewer
// merges group02 + group13 of an event into one two-side view).
local calib_dump(gd) =
    if calib then '%s/calib-evt%s-%s.json' % [output_dir, std.toString(event), gd.name]
    else '';
local ql_chain(gd) =
    local src = qlm_maker.opflash_source(gd.name, "%s/opflash_pdhd-wct.tar.gz" % input);
    local att = qlm_maker.flash_attach(gd.name);
    local dv  = clus_maker.detector_volumes(gd.anodes, gd.face);
    local mat = qlm_maker.matching(gd.anodes, dv, gd.name, gd.face, calib_dump(gd));
    g.intern(
        innodes=[att],
        centernodes=[src],
        outnodes=[mat],
        edges=[
            g.edge(src, att, 0, 1),
            g.edge(att, mat, 0, 0),
        ]
    );

local graph = if do_qlmatch then
    local ql_pipes = [ql_chain(groups[i]) for i in std.range(0, ngroups - 1)];
    g.intern(
        innodes=group_pipes,
        centernodes=ql_pipes,
        outnodes=[clus_all_tpc],
        edges=
            [g.edge(group_pipes[i], ql_pipes[i], 0, 0) for i in std.range(0, ngroups - 1)] +
            [g.edge(ql_pipes[i], clus_all_tpc, 0, i) for i in std.range(0, ngroups - 1)]
    )
else g.intern(
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
        plugins: ["WireCellGen", "WireCellPgraph", "WireCellSio", "WireCellSigProc", "WireCellImg", "WireCellRoot", "WireCellTbb", "WireCellClus", "WireCellMatch", "WireCellAux"],
        apps: ["Pgrapher"]
    }
};

[cmdline] + g.uses(graph) + [app]
