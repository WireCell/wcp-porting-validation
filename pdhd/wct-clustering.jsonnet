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
    // Persist the post-Q/L point-cloud tree for the PR job (run_pr_evt.sh).
    // '' (default) leaves the all-TPC TensorFileSink in its inert dump_mode
    // (trash-all-apa.tar.gz) => compiled config byte-identical.  A path turns it
    // into a real TensorDM writer.  run_clus_evt.sh -save-pctree passes
    // work/<RUN6>_<EVT>/pctree-evt<EVT>.tar.gz.  See
    // pgrapher/experiment/pdhd/clus.jsonnet clus_all_tpc tensor_outname.
    save_tensors = '',
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
// With Q/L matching on, ONE joint QLMatching node emits a single pre-merged tree, so the
// all-TPC stage skips its input PointTreeMerging (premerged).  Without matching the two
// per-side clustering outputs still fan into the ngroups-way merge.
local clus_all_tpc = if do_qlmatch
    then clus_maker.all_tpc(anodes, premerged=true, save_opflash=save_opflash, tensor_outname=save_tensors)
    else clus_maker.all_tpc(anodes, ngroups=ngroups, save_opflash=save_opflash, tensor_outname=save_tensors);

// JOINT Q/L matching: both drift sides enter ONE QLMatching node (like SBND) so the
// cross-cathode (xTPC) consistency pass can pair a cathode-crosser's two halves.  Per
// side: opflash source -> flash_attach (2->1 fan-in: port0 = group cluster tree, port1 =
// opflash matrix); flash_attach[i] feeds the joint node's input port i.  The joint node
// emits one merged tree (-> premerged clus_all_tpc).  grouping_anodes / tpc_faces are
// per-side, so each run keeps its own drift-volume geometry and imaging face.
local qlm = import 'pgrapher/experiment/pdhd/qlmatching.jsonnet';
local qlm_maker = qlm(params, trigger_offset, readout_window_ticks);
// One combined hand-scan calibration dump (empty unless `calib`); both sides land in the
// single file (the ql_scan viewer reads one file with apa=0/1 bundles tagged by lit side).
local calib_dump_joint =
    if calib then '%s/calib-evt%s.json' % [output_dir, std.toString(event)]
    else '';

local graph = if do_qlmatch then
    local srcs = [qlm_maker.opflash_source(groups[i].name, "%s/opflash_pdhd-wct.tar.gz" % input)
                  for i in std.range(0, ngroups - 1)];
    local atts = [qlm_maker.flash_attach(groups[i].name) for i in std.range(0, ngroups - 1)];
    // Joint node sees ALL APAs through ONE DetectorVolumes (both faces registered); the
    // per-side anode lists + faces drive the per-run box union and wire face.
    local dv_all = clus_maker.detector_volumes(anodes);
    local sides  = [gd.anodes for gd in groups];
    local faces  = [gd.face for gd in groups];
    local mat = qlm_maker.matching_joint(sides, dv_all, faces, calib_dump_joint);
    // Flat fan-in: per side i, group cluster tree -> flash_attach[i] port 0, opflash ->
    // port 1; flash_attach[i] -> joint node input port i; joint output -> clus_all_tpc.
    g.intern(
        innodes=group_pipes,
        centernodes=srcs + atts + [mat],
        outnodes=[clus_all_tpc],
        edges=
            [g.edge(srcs[i], atts[i], 0, 1) for i in std.range(0, ngroups - 1)] +
            [g.edge(group_pipes[i], atts[i], 0, 0) for i in std.range(0, ngroups - 1)] +
            [g.edge(atts[i], mat, 0, i) for i in std.range(0, ngroups - 1)] +
            [g.edge(mat, clus_all_tpc, 0, 0)]
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
