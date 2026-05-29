local wc = import "wirecell.jsonnet";
local g = import "pgraph.jsonnet";
local f = import 'pgrapher/common/funcs.jsonnet';
local clus = import "pgrapher/common/clus.jsonnet";

// SBND ParticleDataSet (5 dE/dx + 5 range LinterpFunctions + ParticleDataSet).
// Tables are NIST/PDG (detector-agnostic) copied from qlport.  See plan in
// sbnd/docs/qlport-to-sbnd-downstream-plan.md.
local pds = (import "particle_dataset.jsonnet")();


local time_offset = -200 * wc.us;
local drift_speed = 1.56 * wc.mm / wc.us;
local bee_dir = "data";
local bee_zip = "mabc.zip";

// dQ/dx calibration knobs passed through to TaggerCheckNeutrino /
// UbooneMagnifyTrackingVisitor.  Same defaults as qlport for now; revisit
// when SBND calibration is available.
local dQdx_scale  = 0.1;
local dQdx_offset = -1000;

// Box recombination model.  Constants are from the uboone qlport entry, but
// SBND nominal drift field is ~0.5 kV/cm vs uboone's 0.273.
local sbnd_box_recomb = {
    type: "BoxRecombination",
    name: "box_recomb",
    data: {
        A:      1.0,
        B:      0.255,
        Efield: 0.5,     // SBND nominal (kV/cm)
        rho:    1.38,
        Wi:     23.6e-6,
    },
};

// Fiducial volume.  v1 uses two rectangular PolyFiducial slabs (XY ∧ ZX)
// matching the existing dvm box: X ∈ [-202.5, +201.45] cm, Y ∈ ±200.312 cm,
// Z ∈ [4.05, 505.35] cm.  Refine with real cryostat polygons later.
local sbnd_fid_xy = {
    type: "PolyFiducial",
    name: "fiducial_sbnd_xy",
    data: {
        axis: 2,         // Z-slabs, polygons live in the X-Y plane
        slabs: [{
            min:  4.05  * wc.cm,
            max:  505.35* wc.cm,
            corners: [
                [-202.5  * wc.cm, -200.312 * wc.cm],
                [ 201.45 * wc.cm, -200.312 * wc.cm],
                [ 201.45 * wc.cm,  200.312 * wc.cm],
                [-202.5  * wc.cm,  200.312 * wc.cm],
            ],
        }],
    },
};
local sbnd_fid_zx = {
    type: "PolyFiducial",
    name: "fiducial_sbnd_zx",
    data: {
        axis: 1,         // Y-slabs, polygons live in the Z-X plane
        slabs: [{
            min: -200.312 * wc.cm,
            max:  200.312 * wc.cm,
            corners: [
                [  4.05  * wc.cm, -202.5  * wc.cm],
                [505.35  * wc.cm, -202.5  * wc.cm],
                [505.35  * wc.cm,  201.45 * wc.cm],
                [  4.05  * wc.cm,  201.45 * wc.cm],
            ],
        }],
    },
};
local sbnd_fid = {
    type: "CompositeFiducial",
    name: "sbnd_fid",
    data: {
        logic: "and",
        fiducials: [ wc.tn(sbnd_fid_xy), wc.tn(sbnd_fid_zx) ],
    },
};

// uboone-trained smoke-test weight directories (resolved via WIRECELL_PATH
// against wire-cell-data/).  Scores are meaningless for SBND but verify
// that the C++ scoring/DL paths run end-to-end.  Set to "" to disable.
local smoketest_numu_weights_dir = "uboone/weights";
local smoketest_nue_weights_dir  = "uboone/weights";
local smoketest_dl_weights = "uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth";

local initial_index = "0";
local initial_runNo = "1";
local initial_subRunNo = "1";
local initial_eventNo = "1";
local index = std.parseInt(initial_index);
local LrunNo = std.parseInt(initial_runNo);
local LsubRunNo = std.parseInt(initial_subRunNo);
local LeventNo  = std.parseInt(initial_eventNo);


local common_coords = ["x", "y", "z"];
local common_corr_coords = ["x_t0cor", "y", "z"];


local dvm = {
    overall: {
        FV_xmin: -202.5 * wc.cm,
        FV_xmax: 201.45 * wc.cm,
        FV_ymin: -200.312 * wc.cm,
        FV_ymax:  200.312 * wc.cm,
        FV_zmin: 4.05 * wc.cm,
        FV_zmax: 505.35 * wc.cm,
        FV_xmin_margin: 2 * wc.cm,
        FV_xmax_margin: 2 * wc.cm,
        FV_ymin_margin: 2.5 * wc.cm,
        FV_ymax_margin: 2.5 * wc.cm,
        FV_zmin_margin: 3 * wc.cm,
        FV_zmax_margin: 3 * wc.cm,
        vertical_dir: [0,1,0],
        beam_dir: [0,0,1]
    },
    a0f0pA: {
        drift_speed: drift_speed,
        tick: 0.5 * wc.us,  // 0.5 mm per tick
        tick_drift: self.drift_speed * self.tick,
        time_offset: time_offset,
        nticks_live_slice: 4,
        FV_xmin: -202.5 * wc.cm,
        FV_xmax: -0.45 * wc.cm,
        FV_xmin_margin: 2 * wc.cm,
        FV_xmax_margin: 2 * wc.cm,
    },
    a1f0pA: $.a0f0pA + {
        FV_xmin: 0.45 * wc.mm,
        FV_xmax: 201.45 * wc.mm,
    },
};

local anodes_name(anodes, face="") =
    std.join("-", [std.toString(a.data.ident) for a in anodes]) + if face == "" then "" else "-" + std.toString(face);
          

local detector_volumes(anodes, face="") = {
    "type": "DetectorVolumes",
    "name": "dv-apa" + anodes_name(anodes, face),
    "data": {
        "anodes": [wc.tn(anode) for anode in anodes],
        metadata:
            {overall: dvm["overall"]} +
            {a0f0pA: dvm["a0f0pA"]} +
            {a1f0pA: dvm["a1f0pA"]}
    },
    uses: anodes
};

      
local pctransforms(dv) = {
    type: "PCTransformSet",
    name: dv.name,
    data: { detector_volumes: wc.tn(dv) },
    uses: [dv]
};



local bs_live_face(apa, face) = {
    type: "BlobSampler",
    name: "live-%s-%d"%[apa, face],
    data: {
        drift_speed: drift_speed,
        time_offset: time_offset,
        strategy: ["stepped"],
        extra: ['.*wire_index', '.*charge_val', '.*charge_unc', 'wpid']
    }
};
local bs_dead_face(apa, face) = {
    type: "BlobSampler",
    name: "dead-%s-%d"%[apa, face],
    data: {
        strategy: ["center"],
        extra: [".*"] // want all the extra
    }
};
// The factory used to give blob samplers to ClusteringRetile ("rt").
local bs_rt_face = bs_live_face;

// Special sampler for ImproveCluster_2: enable dead-cell mixing so the
// retiler can interpolate across dead regions.  Matches qlport's
// `bs_live_no_dead_mix`.
local bs_live_no_dead_mix_face(apa, face) = {
    type: "BlobSampler",
    name: "live_no_dead_mix-%s-%d"%[apa, face],
    data: {
        drift_speed: drift_speed,
        time_offset: time_offset,
        strategy: {
            name: "charge_stepped",
            disable_mix_dead_cell: false,   // the key difference
        },
        extra: ['.*wire_index', '.*charge.*', 'wpid'],
    },
};


local clus_per_face (
    anode,
    face,
    dump = true,
    ) =
{

    local dv = detector_volumes([anode], face),
    local pcts = pctransforms(dv),


    local cluster_scope_filter_live = g.pnode({
        type: "ClusterScopeFilter",
        name: "csf-live-%s-%d"%[anode.name, face],
        data: {
            face_index: face,
        }
    }, nin=1, nout=1, uses=[]),

    local cluster_scope_filter_dead = g.pnode({
        type: "ClusterScopeFilter",
        name: "csf-dead-%s-%d"%[anode.name, face],
        data: {
            face_index: face,
        }
    }, nin=1, nout=1, uses=[]),

    local bsl = bs_live_face(anode.name, face),
    local bsd = bs_dead_face(anode.name, face),

    local ptb = g.pnode({
        type: "PointTreeBuilding",
        name: "%s-%d"%[anode.name, face],
        data:  {
            samplers: {
                "3d": wc.tn(bsl),
                "dead": wc.tn(bsd),
            },
            multiplicity: 2,
            tags: ["live", "dead"],
            anode: wc.tn(anode),
            face: face,
            detector_volumes: wc.tn(dv),
        }
    }, nin=2, nout=1, uses=[bsl, bsd, dv]),

    // local cluster2pct = g.intern(
    //     innodes = [cluster_scope_filter_live, cluster_scope_filter_dead],
    //     centernodes = [],
    //     outnodes = [ptb],
    //     edges = [
    //         g.edge(cluster_scope_filter_live, ptb, 0, 0),
    //         g.edge(cluster_scope_filter_dead, ptb, 0, 1)
    //     ]
    // ),
    local cluster2pct = ptb,

    local face_name = "%s-%d"%[anode.name, face],

    local cm = clus.clustering_methods(prefix=face_name,
                                       detector_volumes=dv,
                                       pc_transforms=pcts,
                                       coords=common_coords),
    local cm_pipeline = [
        cm.pointed(),
        // cm.ctpointcloud(),
        cm.live_dead(dead_live_overlap_offset=2),
        cm.extend(flag=4, length_cut=60*wc.cm, num_try=0, length_2_cut=15*wc.cm, num_dead_try=1),
        cm.regular(name="-one", length_cut=60*wc.cm, flag_enable_extend=false),
        cm.regular(name="_two", length_cut=30*wc.cm, flag_enable_extend=true),
        cm.parallel_prolong(length_cut=35*wc.cm),
        cm.close(length_cut=1.2*wc.cm),
        cm.extend_loop(num_try=3),
        cm.separate(use_ctpc=true),
        cm.connect1(),
        // cm.deghost(),
        // cm.examine_x_boundary(),
        // cm.protect_overclustering(),
        // cm.neutrino(),
        // cm.isolated(),
        // cm.retile(cut_time_low=3*wc.us, cut_time_high=5*wc.us, anodes=[anode], samplers=[clus.sampler(bsl, apa=anode.data.ident, face=face)]),
    ],

    local mabc = g.pnode({
        local name = "%s-%d"%[anode.name, face],
        type: "MultiAlgBlobClustering",
        name: name,
        data:  {
            inpath: "pointtrees/%d",
            outpath: "pointtrees/%d",
            // grouping2file_prefix: "grouping%s-%d"%[anode.name, face],
            perf: true,
            bee_dir: bee_dir, // "data/0/0", // not used
            bee_zip: "mabc-%s-face%d.zip"%[anode.name, face],
            bee_detector: "sbnd",
            initial_index: index,   // New RSE configuration
            use_config_rse: true,  // Enable use of configured RSE
            runNo: LrunNo,
            subRunNo: LsubRunNo,
            eventNo: LeventNo,
            save_deadarea: true, 
            anodes: [wc.tn(anode)],
            face: face,
            detector_volumes: wc.tn(dv),
            bee_points_sets: [  // New configuration for multiple bee points sets
                {
                    name: "clustering",         // Name of the bee points set
                    detector: "sbnd",         // Detector name
                    algorithm: "clustering",    // Algorithm identifier
                    pcname: "3d",           // Which scope to use
                    coords: ["x", "y", "z"],    // Coordinates to use
                    individual: true            // Output individual APA/Face
                }
            ],
            pipeline: wc.tns(cm_pipeline),
        }
    }, nin=1, nout=1, uses=[dv, anode, pcts]+cm_pipeline),

    local sink = g.pnode({
        type: "TensorFileSink",
        name: "clus_per_face-%s-%d"%[anode.name, face],
        data: {
            outname: "trash-%s-face%d.tar.gz"%[anode.name, face],
            prefix: "clustering_", // json, numpy, dummy
            dump_mode: true,
        }
    }, nin=1, nout=0),

    local end = if dump
    then g.pipeline([mabc, sink])
    else g.pipeline([mabc]),

    ret :: g.pipeline([cluster2pct, end], "clus_per_face-%s-%d"%[anode.name, face])
}.ret;

local clus_all_apa (
    anodes,
    dump = true,
    ) = {
    local nanodes = std.length(anodes),
    local pcmerging = g.pnode({
        type: "PointTreeMerging",
        name: "clus_all_apa",
        data:  {
            multiplicity: nanodes,
            inpath: "pointtrees/%d",
            outpath: "pointtrees/%d",
        }
    }, nin=nanodes, nout=1),

    local dv = detector_volumes(anodes),
    local pcts = pctransforms(dv),


    local cm_old = clus.clustering_methods(prefix="all",
                                           detector_volumes=dv,
                                           pc_transforms=pcts,
                                           coords=common_coords),


    local cm = clus.clustering_methods(prefix="all",
                                       detector_volumes=dv,
                                       pc_transforms=pcts,
                                       fiducial=sbnd_fid,        // fiducialutils needs this
                                       coords=common_corr_coords),

    // ImproveCluster_2 retiler with the no-dead-mix sampler per anode
    // (SBND has one face per anode in this pipeline).
    local imp2_samplers = [
        clus.sampler(bs_live_no_dead_mix_face(a.name, 0),
                     apa=a.data.ident, face=0)
        for a in anodes
    ],
    local improve_cluster_2_sbnd = cm.improve_cluster_2(
        anodes=anodes,
        samplers=imp2_samplers,
        verbose=true,
    ),

    // uboone-trained BDT/DL scorers — smoke test only.  Scores are
    // meaningless for SBND until retraining; the C++ paths should at
    // least run end-to-end.
    local numu_bdt_scorer = cm.numu_bdt_scorer(
        numu1_weights_xml=     smoketest_numu_weights_dir + "/numu_tagger1.weights.xml",
        numu2_weights_xml=     smoketest_numu_weights_dir + "/numu_tagger2.weights.xml",
        numu3_weights_xml=     smoketest_numu_weights_dir + "/numu_tagger3.weights.xml",
        cosmict10_weights_xml= smoketest_numu_weights_dir + "/cos_tagger_10.weights.xml",
        numu_xgboost_xml=      smoketest_numu_weights_dir + "/numu_scalars_scores_0923.xml",
    ),
    local nue_bdt_scorer = cm.nue_bdt_scorer(
        mipid_weights_xml=       smoketest_nue_weights_dir + "/mipid_BDT.weights.xml",
        gap_weights_xml=         smoketest_nue_weights_dir + "/gap_BDT.weights.xml",
        hol_lol_weights_xml=     smoketest_nue_weights_dir + "/hol_lol_BDT.weights.xml",
        cme_anc_weights_xml=     smoketest_nue_weights_dir + "/cme_anc_BDT.weights.xml",
        mgo_mgt_weights_xml=     smoketest_nue_weights_dir + "/mgo_mgt_BDT.weights.xml",
        br1_weights_xml=         smoketest_nue_weights_dir + "/br1_BDT.weights.xml",
        br3_weights_xml=         smoketest_nue_weights_dir + "/br3_BDT.weights.xml",
        br3_3_weights_xml=       smoketest_nue_weights_dir + "/br3_3_BDT.weights.xml",
        br3_5_weights_xml=       smoketest_nue_weights_dir + "/br3_5_BDT.weights.xml",
        br3_6_weights_xml=       smoketest_nue_weights_dir + "/br3_6_BDT.weights.xml",
        stemdir_br2_weights_xml= smoketest_nue_weights_dir + "/stem_dir_br2_BDT.weights.xml",
        trimuon_weights_xml=     smoketest_nue_weights_dir + "/stl_lem_brm_BDT.weights.xml",
        br4_tro_weights_xml=     smoketest_nue_weights_dir + "/br4_tro_BDT.weights.xml",
        mipquality_weights_xml=  smoketest_nue_weights_dir + "/mipquality_BDT.weights.xml",
        pio_1_weights_xml=       smoketest_nue_weights_dir + "/pio_1_BDT.weights.xml",
        pio_2_weights_xml=       smoketest_nue_weights_dir + "/pio_2_BDT.weights.xml",
        stw_spt_weights_xml=     smoketest_nue_weights_dir + "/stw_spt_BDT.weights.xml",
        vis_1_weights_xml=       smoketest_nue_weights_dir + "/vis_1_BDT.weights.xml",
        vis_2_weights_xml=       smoketest_nue_weights_dir + "/vis_2_BDT.weights.xml",
        stw_2_weights_xml=       smoketest_nue_weights_dir + "/stw_2_BDT.weights.xml",
        stw_3_weights_xml=       smoketest_nue_weights_dir + "/stw_3_BDT.weights.xml",
        stw_4_weights_xml=       smoketest_nue_weights_dir + "/stw_4_BDT.weights.xml",
        sig_1_weights_xml=       smoketest_nue_weights_dir + "/sig_1_BDT.weights.xml",
        sig_2_weights_xml=       smoketest_nue_weights_dir + "/sig_2_BDT.weights.xml",
        lol_1_weights_xml=       smoketest_nue_weights_dir + "/lol_1_BDT.weights.xml",
        lol_2_weights_xml=       smoketest_nue_weights_dir + "/lol_2_BDT.weights.xml",
        tro_1_weights_xml=       smoketest_nue_weights_dir + "/tro_1_BDT.weights.xml",
        tro_2_weights_xml=       smoketest_nue_weights_dir + "/tro_2_BDT.weights.xml",
        tro_4_weights_xml=       smoketest_nue_weights_dir + "/tro_4_BDT.weights.xml",
        tro_5_weights_xml=       smoketest_nue_weights_dir + "/tro_5_BDT.weights.xml",
        nue_xgboost_xml=         smoketest_nue_weights_dir + "/XGB_nue_seed2_0923.xml",
    ),

    local cm_pipeline = [
        // cm_old.examine_x_boundary(),
        cm_old.switch_scope(),

        cm.extend(flag=4, length_cut=60*wc.cm, num_try=0, length_2_cut=15*wc.cm, num_dead_try= 1),
        cm.regular(name="1", length_cut=60*wc.cm, flag_enable_extend=false),
        cm.regular(name="2", length_cut=30*wc.cm, flag_enable_extend=true),
        cm.parallel_prolong(length_cut=35*wc.cm),
        cm.close(length_cut=1.2*wc.cm),
        cm.extend_loop(num_try=3),
        // cm.separate(use_ctpc=true),
        cm.neutrino(),
        cm.isolated(),
        // cm.examine_bundles(),

        // --- qlport-style downstream chain (added 2026-05-27) ---
        // tagger_flag_transfer re-added 2026-05-28: WireCellMatch now tags
        // the main_clus, so the upstream `tagger_info` PC is available.
        cm.tagger_flag_transfer("tagger"),
        cm.clustering_recovering_bundle("recover_bundle", graph_name="relaxed_pid"),
        cm.steiner(retiler=improve_cluster_2_sbnd, perf=true),
        cm.fiducialutils(),
        cm.tagger_check_neutrino(
            trackfitting_config_file="sbnd_track_fitting.json",
            recombination_model=wc.tn(sbnd_box_recomb),
            particle_dataset=wc.tn(pds.particle_dataset),
            perf=true,
            dl_weights=smoketest_dl_weights,
            dQdx_scale=dQdx_scale,
            dQdx_offset=dQdx_offset,
            clus_geom_helper="",                 // no SBND SCE in v1
        ),
        numu_bdt_scorer,
        nue_bdt_scorer,
        // --- end qlport-style chain ---

        #cm.retile(cut_time_low=3*wc.us,
        #          cut_time_high=5*wc.us,
        #          anodes=anodes,
        #          samplers=[
        #              clus.sampler(bs_rt_face(0,0), apa=0, face=0),
        #              clus.sampler(bs_rt_face(0,1), apa=0, face=1),
        #              clus.sampler(bs_rt_face(1,0), apa=1, face=0),
        #              clus.sampler(bs_rt_face(1,1), apa=1, face=1),
        #              clus.sampler(bs_rt_face(2,0), apa=2, face=0),
        #              clus.sampler(bs_rt_face(2,1), apa=2, face=1),
        #              clus.sampler(bs_rt_face(3,0), apa=3, face=0),
        #              clus.sampler(bs_rt_face(3,1), apa=3, face=1),
        #          ]),
    ],

    local mabc = g.pnode({
        type: "MultiAlgBlobClustering",
        name: "clus_all_apa",
        data:  {
            inpath: "pointtrees/%d",
            outpath: "pointtrees/%d",
            // grouping2file_prefix: "grouping%s-%d"%[anode.name, face],
            perf: true,
            bee_dir: bee_dir, // "data/0/0", // not used
            bee_zip: "mabc-all-apa.zip",
            bee_detector: "sbnd",
            initial_index: index,   // New RSE configuration
            use_config_rse: true,  // Enable use of configured RSE
            runNo: LrunNo,
            subRunNo: LsubRunNo,
            eventNo: LeventNo,
            save_deadarea: true, 
            anodes: [wc.tn(a) for a in anodes],
            detector_volumes: wc.tn(dv),
            bee_points_sets: [  // New configuration for multiple bee points sets
            {
               name: "img",                // Name of the bee points set
               detector: "sbnd",         // Detector name
               algorithm: "img",           // Algorithm identifier
               pcname: "3d",           // Which scope to use
               coords: ["x", "y", "z"],    // Coordinates to use
               individual: false           // Whether to output as a whole or individual APA/Face
            },
            {
                    name: "clustering",         // Name of the bee points set
                    detector: "sbnd",         // Detector name
                    algorithm: "clustering",    // Algorithm identifier
                    pcname: "3d",           // Which scope to use
                    coords: ["x_t0cor", "y", "z"],    // Coordinates to use
                    individual: false            // Output individual APA/Face
                },

                // --- qlport-style bee point sets (added 2026-05-27) ---
                // Steiner-graph / track-fit / shower-track / vertex point
                // sets produced by CreateSteinerGraph and TaggerCheckNeutrino.
                {
                    name: "regular",
                    visitor: "CreateSteinerGraph:all",
                    detector: "sbnd",
                    algorithm: "regular",
                    pcname: "3d",
                    coords: ["x_t0cor", "y", "z"],
                    individual: false,
                    filter: 1,                   // apply scope filter
                },
                {
                    name: "steiner",
                    visitor: "CreateSteinerGraph:all",
                    detector: "sbnd",
                    algorithm: "steiner",
                    pcname: "steiner_pc",
                    coords: ["x_t0cor", "y", "z"],
                    individual: false,
                },
                {
                    name: "track_fit",
                    visitor: "TaggerCheckNeutrino:all",
                    grouping: "live",
                    detector: "sbnd",
                    algorithm: "track_fit",
                    pcname: "3d",
                    coords: ["x", "y", "z"],
                    individual: false,
                    dQdx_scale: dQdx_scale,
                    dQdx_offset: dQdx_offset,
                },
                {
                    name: "shower_track",
                    visitor: "TaggerCheckNeutrino:all",
                    grouping: "live",
                    detector: "sbnd",
                    algorithm: "shower_track",
                    pcname: "3d",
                    coords: ["x", "y", "z"],
                    individual: false,
                    use_associate_points: true,
                },
                {
                    name: "vertices",
                    visitor: "TaggerCheckNeutrino:all",
                    grouping: "live",
                    detector: "sbnd",
                    algorithm: "vertices",
                    pcname: "3d",
                    coords: ["x", "y", "z"],
                    individual: false,
                    use_graph_vertices: true,
                },
                // --- end qlport-style bee point sets ---
            ],
            // Particle-flow Bee output: one JSON per event in
            // data/{index}/{index}-mc.json, emitted after TaggerCheckNeutrino.
            bee_pf: [
                {
                    name: "mc",
                    visitor: "TaggerCheckNeutrino:all",
                    grouping: "live",
                },
            ],
            pipeline: wc.tns(cm_pipeline),
        },
    }, nin=1, nout=1, uses=anodes+[dv, pcts, sbnd_box_recomb, sbnd_fid, sbnd_fid_xy, sbnd_fid_zx]+pds.all+cm_pipeline),

    local sink = g.pnode({
        type: "TensorFileSink",
        name: "clus_all_apa",
        data: {
            outname: "trash-all-apa.tar.gz",
            prefix: "clustering_", // json, numpy, dummy
            dump_mode: true,
        }
    }, nin=1, nout=0),
    local end = if dump
    then g.pipeline([mabc, sink])
    else g.pipeline([mabc]),
    ret :: g.intern(
        innodes = [pcmerging],
        centernodes = [],
        outnodes = [end],
        edges = [
            g.edge(pcmerging, end, 0, 0),
        ]
    ),
}.ret;


function () {
    per_face(anode, face=0, dump=true) :: clus_per_face(anode, face=face, dump=dump),
    per_apa(anode, dump=true) :: clus_per_face(anode, face=0, dump=dump), # face=anode is specific to sbnd
    all_apa(anodes, dump=true) :: clus_all_apa(anodes, dump=dump),
    detector_volumes(anodes, face=0) :: detector_volumes(anodes=anodes, face=face),
}
