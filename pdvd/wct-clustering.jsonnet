local g = import "pgraph.jsonnet";
local f = import "pgrapher/common/funcs.jsonnet";
local wc = import "wirecell.jsonnet";

local tools_maker = import 'pgrapher/common/tools.jsonnet';

local reality = 'data';
local data_params = import 'pgrapher/experiment/protodunevd/params.jsonnet';
local simu_params = import 'pgrapher/experiment/protodunevd/simparams.jsonnet';
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
    // Stepped-sampler fallback: blobs the stepped grid leaves point-less get
    // one point at the blob center.  Default off -> bit-identical output.
    stepped_center_fallback = false,
    // Event-T0 / readout-tick0 compensation for the live BlobSampler drift-x
    // conversion.  PDVD has no per-event T0, so default 0 (no preset T0).
    // -250us would place trigger-time activity at its true x; see
    // pgrapher/experiment/protodunevd/clus.jsonnet.
    time_offset = 0,
    // Accept any-volume containment in the switch_scope T0Correction filter
    // (out-of-time clusters keep participating in all-APA merge passes).
    relax_containment_filter = true,
    // Interpose charge-light (Q/L) matching between the per-drift-group
    // clustering (stage 3) and the final all-TPC clustering (stage 4).  Default
    // false => the historical no-matching chain, bit-identical.  PDVD runs ONE
    // all-PD flash list (opflash_input) against both drift volumes via the
    // shared-flash joint QLMatching.  See pgrapher/experiment/protodunevd/
    // qlmatching.jsonnet and pdvd/docs/pdvd-qlmatching.md.
    do_qlmatch = false,
    // Full path to the per-event opflash tensor archive (the light chain's
    // work/<RUN>_light<EVENTNO>/opflash_pdvd-wct.tar.gz — NOT in the charge
    // input dir, unlike PDHD).  Required when do_qlmatch.
    opflash_input = '',
    // Hand-scan calibration dump: when true (and do_qlmatch) QLMatching writes
    // work/<output_dir>/calib-evt<event>.json for the pdvd/ql_scan viewer.
    calib = false,
    // Dump the optical "op" bee instance (measured flash PE + Q/L predicted PE
    // per matched cluster) from the all-TPC MABC.  Needs do_qlmatch.
    save_opflash = false,
    // Per-event light-vs-charge time-base offset in MICROSECONDS (calibrated
    // constant + opflash metadata offset_us; see run_clus_evt.sh).  Applied
    // DOWNSTREAM (matching geometry + T0Correction x_t0cor).  Default 0.
    trigger_offset_us = 0,
    // Post-resample readout-window length (ticks) for the Q/L window-truncation
    // flag.  run_clus_evt.sh reads the real value from the SP frame (10000).
    readout_window_ticks = 10000,
    // Q/L visibility backend: 'library' (v5 ANN grid, default) or 'semi'
    // (fitted semi-analytical fallback; membrane XAs auto-masked).
    light_model = 'library',
    // Q/L knob overrides for the trigger-offset diagnostic runs (see
    // pdvd/ql_light_calib/): containment is meaningless until the time base is
    // calibrated, and a higher PE floor bounds the un-pruned bundle count.
    ql_require_containment = true,
    ql_flash_minpe = 25,
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
local clus = import 'pgrapher/experiment/protodunevd/clus.jsonnet';
local clus_maker = clus(output_dir=output_dir, runNo=run, subRunNo=subrun, eventNo=event, stepped_center_fallback=stepped_center_fallback,
                        time_offset=time_offset, relax_containment_filter=relax_containment_filter,
                        trigger_offset=trigger_offset);

// Drift-side groups: anodes 0-3 (bottom drift) and anodes 4-7 (top drift).
// With a subset anode_indices only non-empty groups are built and the final
// merge multiplicity shrinks to match.
local group_defs = [
    { name: "group0123", anodes: [a for a in anodes if a.data.ident < 4] },
    { name: "group4567", anodes: [a for a in anodes if a.data.ident >= 4] },
];
local groups = [gd for gd in group_defs if std.length(gd.anodes) > 0];
local ngroups = std.length(groups);

// One drift-side pipe: per-anode sources -> per_apa (stages 1+2) -> per_group (stage 3).
local group_pipe(gd) =
    local n = std.length(gd.anodes);
    local actives = [cluster_source("%s/clusters-apa-anode%d-ms-active.tar.gz"%[input, a.data.ident]) for a in gd.anodes];
    local maskeds = [cluster_source("%s/clusters-apa-anode%d-ms-masked.tar.gz"%[input, a.data.ident]) for a in gd.anodes];
    local apa_pipes = [clus_maker.per_apa(gd.anodes[i], dump=false) for i in std.range(0, n - 1)];
    local pg = clus_maker.per_group(gd.anodes, gd.name, dump=false);
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
// With Q/L matching on, ONE joint QLMatching node emits a single pre-merged tree, so
// the all-TPC stage skips its input PointTreeMerging (premerged).  Without matching
// the two per-side clustering outputs still fan into the ngroups-way merge.
local clus_all_tpc = if do_qlmatch
    then clus_maker.all_tpc(anodes, premerged=true, save_opflash=save_opflash)
    else clus_maker.all_tpc(anodes, ngroups=ngroups);

// JOINT Q/L matching (shared-flash): both drift sides enter ONE QLMatching node and
// each reads the SAME all-PD opflash archive.  Per side: opflash source ->
// flash_attach (2->1 fan-in: port0 = group cluster tree, port1 = opflash matrix);
// flash_attach[i] feeds the joint node's input port i.  The joint node runs ONE
// LASSO over the shared flashes (clusters from both sides explain each flash) and
// emits one merged tree (-> premerged clus_all_tpc).  PDVD imaging face is 0 for
// both drift sides (the two faces of a CRP share x-bounds).
local qlm = import 'pgrapher/experiment/protodunevd/qlmatching.jsonnet';
local qlm_maker = qlm(params, trigger_offset, readout_window_ticks, light_model,
                      ql_require_containment, ql_flash_minpe);
local calib_dump_joint =
    if calib then '%s/calib-evt%s.json' % [output_dir, std.toString(event)]
    else '';

local graph = if do_qlmatch then
    local srcs = [qlm_maker.opflash_source(groups[i].name, opflash_input)
                  for i in std.range(0, ngroups - 1)];
    local atts = [qlm_maker.flash_attach(groups[i].name) for i in std.range(0, ngroups - 1)];
    local dv_all = clus_maker.detector_volumes(anodes);
    local sides  = [gd.anodes for gd in groups];
    local faces  = [0 for gd in groups];
    local mat = qlm_maker.matching_joint(sides, dv_all, faces, calib_dump_joint);
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
