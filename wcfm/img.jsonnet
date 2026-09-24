// wcfm/img.jsonnet -- imaging module for the DUNE FD-HD 1x2x6 "workspace" (wcfm/docs/02).
//
// Forked BY DUPLICATION from toolkit cfg/pgrapher/experiment/pdhd/img.jsonnet (untouched;
// CLAUDE.md M10).  Differences from the PDHD file, and nothing else:
//   * params: wcfm_params.jsonnet (dune10kt simparams: 12 APAs at x=0, both faces live);
//   * the "full" solving pipeline can carry the W4 truth catchers (config.depos != ''):
//     a ClusterFanout + BlobDepoFill + ClusterFileSink pair at TWO points of the chain,
//       tier "tru0": right after BlobClustering  = every tiled blob, before any deghosting
//       tier "tru":  right after GlobalGeomClustering = the survivors the ms-active file holds
//     both fed by one DepoFileSource (the sim's drifted depos) through a DepoSetFanout.
//     BlobDepoFill keeps blob idents and vertex order, so the "tru" file is desc-aligned
//     with the "ms-active" file (doc 02 sec 4).  With depos == '' the chain is the PDHD one.
// Everything else (pre-processing, slicing, tiling faces [0,1], the 3-pass active fork,
// the masked fork, sink format) is the PDHD production configuration verbatim.
function(cfg={})
local config = {
    use_dnn_img: false,
    // Per-plane slicing activity threshold in units of channel RMS.
    // Default 3.6 sigma.  To slice on any positive charge use 1e-6: a
    // literal 0 would trip MaskSlice's `if(threshold==0)` fallback to a
    // default threshold (a HIGH MicroBooNE bar), so 1e-6 is the charge>0
    // surrogate (used by the standalone pdhd chain, see pdhd/wct-img-all.jsonnet).
    nthreshold: [3.6, 3.6, 3.6],
    // W4 truth: path of the drifted-depo file (DepoFileSink output); '' = no catchers.
    depos: '',
    // BlobDepoFill: drift speed and depo-time -> frame-time offset (wcfm_params).
    depofill_speed: 1.6 * 1e3 / 1e3,   // overridden by wct-img-all from wcfm_params
    depofill_time_offset: 0,
    depofill_nsigma: 3.0,
    depofill_pindex: 2,     // BlobDepoFill primary plane (diagnostics)
} + cfg;

local wc = import "wirecell.jsonnet";
local g = import "pgraph.jsonnet";
local f = import 'pgrapher/common/funcs.jsonnet';
local P = import "wcfm_params.jsonnet";
local params = P.params;
local tools_maker = import 'pgrapher/common/tools.jsonnet';
local tools = tools_maker(params);
local anodes = tools.anodes;

local tags = {
    signal: if config.use_dnn_img then "dnnsp%d" else "gauss%d",
    error_tag: if config.use_dnn_img then "dnnsp_error%d" else "gauss_error%d",
    slice_ref: if config.use_dnn_img then "dnnsp%d" else "wiener%d",
    masking: function(ident)
        if config.use_dnn_img
        then ["dnnsp%d" % ident]
        else ["gauss%d" % ident, "wiener%d" % ident],
};

local img = {
    // IFrame -> IFrame
    pre_proc :: function(anode, aname = "") {

    local waveform_map = {
        type: 'WaveformMap',
        name: 'wfm',
        data: {
            filename: "microboone-charge-error.json.bz2",
        }, uses: [],},

    local charge_err = g.pnode({
        type: 'ChargeErrorFrameEstimator',
        name: "cefe-"+aname,
        data: {
            intag: tags.signal % anode.data.ident,
            outtag: tags.error_tag % anode.data.ident,
            anode: wc.tn(anode),
            rebin: 4,  // this number should be consistent with the waveform_map choice
            fudge_factors: [2.31, 2.31, 1.1],  // fudge factors for each plane [0,1,2]
            time_limits: [12, 800],  // the unit of this is in ticks
            errors: wc.tn(waveform_map),
        },
    }, nin=1, nout=1, uses=[waveform_map, anode]),

    local cmm_mod = g.pnode({
        type: 'CMMModifier',
        name: "cmm-mod-"+aname,
        data: {
            cm_tag: "bad",
            trace_tag: tags.signal % anode.data.ident,
            anode: wc.tn(anode),
            ncount_org: 1,   // organize the dead channel ranges according to these boundaries
            org_llimit: [0], // must be ordered ...
            org_hlimit: [8500], // must be ordered ...
        },
    }, nin=1, nout=1, uses=[anode]),

    local frame_masking = g.pnode({
            type: 'FrameMasking',
            name: "frame-masking-"+aname,
            data: {
                cm_tag: "bad",
                trace_tags: tags.masking(anode.data.ident),
                anode: wc.tn(anode),
            },
        }, nin=1, nout=1, uses=[anode]),

        ret: g.pipeline([cmm_mod, frame_masking, charge_err], "uboone-preproc"),
    }.ret,

    // A functio that sets up slicing for an APA.
    slicing :: function(anode, aname, span=4, active_planes=[0,1,2], masked_planes=[], dummy_planes=[]) {
        ret: g.pnode({
            type: "MaskSlices",
            name: "slicing-"+aname,
            data: {
                tick_span: span,
                wiener_tag: tags.slice_ref % anode.data.ident,
                summary_tag: tags.slice_ref % anode.data.ident,
                charge_tag: tags.signal % anode.data.ident,
                error_tag: tags.error_tag % anode.data.ident,
                anode: wc.tn(anode),
                // Both 0 = MaskSlice auto-derives the window from the input
                // frame.  A hard max_tbin beyond the readout fabricates
                // phantom dead slices past the frame end, and one below a
                // longer readout truncates real activity (see
                // pdvd/docs/sp-img-readout-window-truncation.md).
                min_tbin: 0,
                max_tbin: 0,
                active_planes: active_planes,
                masked_planes: masked_planes,
                dummy_planes: dummy_planes,
                nthreshold: config.nthreshold,
            },
        }, nin=1, nout=1, uses=[anode]),
    }.ret,

    // A function sets up tiling for an APA incuding a per-face split.
    tiling :: function(anode, aname) {

        local slice_fanout = g.pnode({
            type: "SliceFanout",
            name: "slicefanout-" + aname,
            data: { multiplicity: 2 },
        }, nin=1, nout=2),

        local tilings = [g.pnode({
            type: "GridTiling",
            name: "tiling-%s-face%d"%[aname, face],
            data: {
                anode: wc.tn(anode),
                face: face,
                nudge: 1e-2,
            }
        }, nin=1, nout=1, uses=[anode]) for face in [0,1]],

        local blobsync = g.pnode({
            type: "BlobSetSync",
            name: "blobsetsync-" + aname,
            data: { multiplicity: 2 }
        }, nin=2, nout=1),

        // two faces
        ret: g.intern(
            innodes=[slice_fanout],
            outnodes=[blobsync],
            centernodes=tilings,
            edges=
                [g.edge(slice_fanout, tilings[n], n, 0) for n in [0,1]] +
                [g.edge(tilings[n], blobsync, 0, n) for n in [0,1]],
            name='tiling-' + aname),
    }.ret,

    //
    multi_active_slicing_tiling :: function(anode, name, span=4) {
        local active_planes = [[0,1,2],[0,1],[1,2],[0,2],],
        local masked_planes = [[],[2],[0],[1]],
        local iota = std.range(0,std.length(active_planes)-1),
        local slicings = [$.slicing(anode, name+"_%d"%n, span, active_planes[n], masked_planes[n])
            for n in iota],
        local tilings = [$.tiling(anode, name+"_%d"%n)
            for n in iota],
        local multipass = [g.pipeline([slicings[n],tilings[n]]) for n in iota],
        ret: f.fanpipe("FrameFanout", multipass, "BlobSetMerge", "multi_active_slicing_tiling_%s"%name),
    }.ret,

    //
    multi_masked_2view_slicing_tiling :: function(anode, name, span=500) {
        local dummy_planes = [[2],[0],[1]],
        local masked_planes = [[0,1],[1,2],[0,2]],
        local iota = std.range(0,std.length(dummy_planes)-1),
        local slicings = [$.slicing(anode, name+"_%d"%n, span,
            active_planes=[],masked_planes=masked_planes[n], dummy_planes=dummy_planes[n])
            for n in iota],
        local tilings = [$.tiling(anode, name+"_%d"%n)
            for n in iota],
        local multipass = [g.pipeline([slicings[n],tilings[n]]) for n in iota],
        ret: f.fanpipe("FrameFanout", multipass, "BlobSetMerge", "multi_masked_slicing_tiling_%s"%name),
    }.ret,

    local clustering_policy = "uboone", // uboone, simple

    // Just clustering
    clustering :: function(anode, aname, spans=1.0) {
        ret : g.pnode({
            type: "BlobClustering",
            name: "blobclustering-" + aname,
            data:  { spans : spans, policy: clustering_policy }
        }, nin=1, nout=1),
    }.ret,

    // W4 truth catcher: ICluster in (fan port 0) -> ICluster out (fan port 0); the fan's
    // port 1 feeds BlobDepoFill port 0, whose port 1 takes the drifted IDepoSet.
    catcher :: function(aname, tier, output_dir='') {
        local outname = if output_dir == '' then "clusters-%s-%s.tar.gz"%[tier, aname]
                        else output_dir+"/clusters-%s-%s.tar.gz"%[tier, aname],
        fan: g.pnode({
            type: 'ClusterFanout',
            name: 'catch-%s-%s'%[tier, aname],
            data: { multiplicity: 2 },
        }, nin=1, nout=2),
        fill: g.pnode({
            type: 'BlobDepoFill',
            name: 'depofill-%s-%s'%[tier, aname],
            data: {
                speed: config.depofill_speed,
                time_offset: config.depofill_time_offset,
                nsigma: config.depofill_nsigma,
                pindex: config.depofill_pindex,
            },
        }, nin=2, nout=1),
        sink: g.pnode({
            type: "ClusterFileSink",
            name: "trusink-%s-%s"%[tier, aname],
            data: { outname: outname, format: "numpy" },
        }, nin=1, nout=0),
    },

    // in: IBlobSet out: ICluster
    solving :: function(anode, aname, solving_type = "simple", output_dir='') {

        local bc = g.pnode({
            type: "BlobClustering",
            name: "blobclustering-" + aname,
            data:  { policy: "uboone" }
        }, nin=1, nout=1),

        local gc = g.pnode({
            type: "GlobalGeomClustering",
            name: "global-clustering-" + aname,
            data:  { policy: "uboone" }
        }, nin=1, nout=1),

        solving :: function(suffix = "1st") {
            local bg = g.pnode({
                type: "BlobGrouping",
                name: "blobgrouping-" + aname + suffix,
                data:  {
                }
            }, nin=1, nout=1),
            local cs1 = g.pnode({
                type: "ChargeSolving",
                name: "cs1-" + aname + suffix,
                data:  {
                    weighting_strategies: ["uniform"], //"uniform", "simple", "uboone"
                    solve_config: "uboone",
                    whiten: true,
                }
            }, nin=1, nout=1),
            local cs2 = g.pnode({
                type: "ChargeSolving",
                name: "cs2-" + aname + suffix,
                data:  {
                    weighting_strategies: ["uboone"], //"uniform", "simple", "uboone"
                    solve_config: "uboone",
                    whiten: true,
                }
            }, nin=1, nout=1),
            local local_clustering = g.pnode({
                type: "LocalGeomClustering",
                name: "local-clustering-" + aname + suffix,
                data:  {
                    dryrun: false,
                }
            }, nin=1, nout=1),
            ret: g.pipeline([bg, cs1, local_clustering, cs2],"cs-pipe"+aname+suffix),
        }.ret,

        global_deghosting :: function(suffix = "1st") {
            ret: g.pnode({
                type: "ProjectionDeghosting",
                name: "ProjectionDeghosting-" + aname + suffix,
                data:  {
                    dryrun: false,
                }
            }, nin=1, nout=1),
        }.ret,

        local_deghosting :: function(config_round = 1, suffix = "1st", good_blob_charge_th=300) {
            ret: g.pnode({
                type: "InSliceDeghosting",
                name: "inslice_deghosting-" + aname + suffix,
                data:  {
                    dryrun: false,
                    config_round: config_round,
                    good_blob_charge_th: good_blob_charge_th,
                }
            }, nin=1, nout=1),
        }.ret,

        local gd1 = self.global_deghosting("1st"),
        local cs1 = self.solving("1st"),
        local ld1 = self.local_deghosting(1,"1st"),

        local gd2 = self.global_deghosting("2nd"),
        local cs2 = self.solving("2nd"),
        local ld2 = self.local_deghosting(2,"2nd"),

        local cs3 = self.solving("3rd"),
        local ld3 = self.local_deghosting(3,"3rd"),

        // The PDHD chain: bc, gd1, cs1, ld1, gd2, cs2, ld2, cs3, ld3, gc.  With truth
        // catchers: bc, [tru0], gd1, ..., gc, [tru].
        local full_with_truth = {
            local c0 = img.catcher(aname, "tru0", output_dir),
            local c1 = img.catcher(aname, "tru", output_dir),
            local depo_src = g.pnode({
                type: 'DepoFileSource',
                name: 'deposrc-' + aname,
                data: { inname: config.depos, scale: 1.0 },
            }, nin=0, nout=1),
            local depo_fan = g.pnode({
                type: 'DepoSetFanout',
                name: 'depofan-' + aname,
                data: { multiplicity: 2 },
            }, nin=1, nout=2),
            local mid = g.pipeline([gd1, cs1, ld1, gd2, cs2, ld2, cs3, ld3, gc], "uboone-solving-mid-"+aname),
            ret: g.intern(
                innodes=[bc],
                outnodes=[c1.fan],
                centernodes=[c0.fan, c0.fill, c0.sink, mid, c1.fill, c1.sink, depo_src, depo_fan],
                edges=[
                    g.edge(bc, c0.fan, 0, 0),
                    g.edge(c0.fan, mid, 0, 0),
                    g.edge(c0.fan, c0.fill, 1, 0),
                    g.edge(c0.fill, c0.sink, 0, 0),
                    g.edge(mid, c1.fan, 0, 0),
                    g.edge(c1.fan, c1.fill, 1, 0),
                    g.edge(c1.fill, c1.sink, 0, 0),
                    g.edge(depo_src, depo_fan, 0, 0),
                    g.edge(depo_fan, c0.fill, 0, 1),
                    g.edge(depo_fan, c1.fill, 1, 1),
                ],
                iports=bc.iports,
                oports=[c1.fan.oports[0]],
                name="uboone-solving-truth-"+aname),
        }.ret,

        ret:
        if solving_type == "full"
        then (if config.depos == '' then g.pipeline([bc, gd1, cs1, ld1, gd2, cs2, ld2, cs3, ld3, gc],"uboone-solving")
              else full_with_truth)
        else g.pipeline([bc, cs1, ld1, gc],"simple-solving"),
    }.ret,

    dump :: function(anode, aname, drift_speed, output_dir='') {
        local outname = if output_dir == '' then "clusters-apa-"+aname+".tar.gz"
                        else output_dir+"/clusters-apa-"+aname+".tar.gz",
        local cs = g.pnode({
            type: "ClusterFileSink",
            name: "clustersink-"+aname,
            data: {
                outname: outname,
                format: "numpy", // json, numpy, dummy; numpy avoids the jsoncpp DOM on load (~90% of clustering live heap, see clus/docs/imgclus-optimization-log.md entry 20)
            }
        }, nin=1, nout=0),
        ret: cs
    }.ret,
};

{
    // Output names use "anode<N>" (the pdvd convention) rather than tools' anode.name "apa<N>".
    local aname_of(anode) = "anode%d" % anode.data.ident,

    local imgpipe(anode, multi_slicing, output_dir='') =
    local aname = aname_of(anode);
    if multi_slicing == "active"
    then g.pipeline([
            img.multi_active_slicing_tiling(anode, aname+"-ms-active", 4),
            img.solving(anode, aname+"-ms-active", "full", output_dir),
            img.dump(anode, aname+"-ms-active", params.lar.drift_speed, output_dir)])
    else if multi_slicing == "masked"
    then g.pipeline([
            img.multi_masked_2view_slicing_tiling(anode, aname+"-ms-masked", 1500),
            img.clustering(anode, aname+"-ms-masked"),
            img.dump(anode, aname+"-ms-masked", params.lar.drift_speed, output_dir)])
    else {
        local active_fork = g.pipeline([
            img.multi_active_slicing_tiling(anode, aname+"-ms-active", 4),
            img.solving(anode, aname+"-ms-active", "full", output_dir),
            img.dump(anode, aname+"-ms-active", params.lar.drift_speed, output_dir),
        ]),
        local masked_fork = g.pipeline([
            img.multi_masked_2view_slicing_tiling(anode, aname+"-ms-masked", 1500), // was 500; masked fork carries geometry only (no charge solving), coarse span cuts masked blob count/memory ~3x
            img.clustering(anode, aname+"-ms-masked"),
            img.dump(anode, aname+"-ms-masked", params.lar.drift_speed, output_dir),
        ]),
        ret: g.fan.fanout("FrameFanout",[active_fork,masked_fork], "fan_active_masked-%s"%aname),
    }.ret,

    per_anode(anode, pipe_type = "multi", output_dir='') :: g.pipeline([
        img.pre_proc(anode, aname_of(anode)),
        imgpipe(anode, pipe_type, output_dir),
    ], "per_anode"),
}
