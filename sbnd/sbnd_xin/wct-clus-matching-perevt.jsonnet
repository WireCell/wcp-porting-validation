// Per-event SBND charge-light (Q/L) matching — standalone, self-contained.
//
// Reads the toolkit's OWN per-event imaging output (run_img_evt.sh →
// work/evt<ID>/icluster-apa<N>-{active,masked}.npz) plus that event's opflash,
// runs per-APA clustering + charge-light matching, and writes one
// work/evt<ID>/mabc-all-apa.zip (img + clustering + 2-view dead + op/Q-L layers).
//
// Unlike wct-clus-matching-standalone.jsonnet (all-10, yuhw's larsoft active
// clusters, masked imaged in-graph), this reads active AND masked from the
// per-event npz — no in-graph imaging — so it is self-contained per event and
// parallelizable.  Charge clusters are the toolkit's own (run_img_evt.sh), not
// yuhw's larsoft dumps.  Driven by run_ql_evt.sh.
//
// Usage (called from run_ql_evt.sh):
//   wire-cell \
//     --tla-str  input=work/evt2 \
//     --tla-code anode_indices='[0,1]' \
//     --tla-str  output_dir=work/evt2 \
//     --tla-code run=0 --tla-code subrun=0 --tla-code event=2 \
//     --tla-str  reality=sim \
//     --tla-str  semimodel_file=semi-analytical-sbnd.json \
//     --tla-code DL=6.2 --tla-code DT=9.8 \
//     --tla-code lifetime=6 --tla-code driftSpeed=1.563 \
//     -c wct-clus-matching-perevt.jsonnet

local g = import 'pgraph.jsonnet';
local wc = import 'wirecell.jsonnet';
local tools_maker = import 'pgrapher/common/tools.jsonnet';

function(
    input          = '.',
    anode_indices  = [0, 1],
    output_dir     = '.',
    run            = 0,
    subrun         = 0,
    event          = 0,
    reality        = 'sim',
    semimodel_file = 'semi-analytical-sbnd.json',
    DL             = 6.2,
    DT             = 9.8,
    lifetime       = 6.0,
    driftSpeed     = 1.563,
    // Joint multi-APA matching toggle. false (default) = historical per-APA path
    // (one QLMatching per APA -> PointTreeMerging -> all-APA MABC). true = one
    // joint QLMatching node matches both APAs and merges, feeding MABC directly.
    joint          = false,
    // Hand-scan calibration dump path. '' (default) = off, production-identical.
    // When set, QLMatching writes one per-event JSON (both TPCs) for the Q/L
    // hand-scan viewer (sbnd_xin/ql_scan). run_ql_evt.sh -calib points it at
    // work/evt<ID>/calib-evt<ID>.json.
    calib_dump     = '',
    // Cathode-crossing TPC0/TPC1 offset diagnostic. '' (default) = off,
    // production-identical. When set (any non-empty string), QLMatching logs the
    // three-vector (dir0, dir1, conn) decomposition per cross-TPC cathode-crossing
    // pair (grep "QLCATHODE" in the run log). run_ql_evt.sh -cathode-diag enables it.
    cathode_diag   = '',
    // Per-PMT predicted-light non-linearity correction. true (default, SBND going
    // forward) maps each PMT's accumulated predicted PE into the saturated (observed)
    // space (qlmatching.jsonnet -> pmt_nonlinearity_params.jsonnet). false = identity.
    // run_ql_evt.sh threads it via PMT_NL / --tla-code pmt_nl.
    pmt_nl         = true,
    // Per-event dynamic dead-PMT auto-mask. false (default) = off, production-identical.
    // true masks, per event, a PMT that never fires while its live neighbours do (a
    // run-dead channel absent from the static ch_mask). run_ql_evt.sh -auto-mask enables it.
    auto_mask      = false,
    // Beam-window flash preference OVERRIDE overlay. Since the validation-round-2
    // adoption the preference is ON in the production config
    // (cfg/.../sbnd/qlmatching.jsonnet: weight 0.5, rescue 0.2, gate ks 0.3 /
    // pred 2%; docs/22_ql-beam-flash-preference.md), so false (default) here
    // simply inherits production. true re-asserts the same keys via the shim
    // overlay -- useful only with beam_pref_weight/beam_pref_rescue below to
    // scan a different operating point (run_ql_evt.sh -beam-pref +
    // BEAMPREF_WEIGHT/BEAMPREF_RESCUE).
    beam_pref      = false,
    // Beam-preference operating point (inert while beam_pref=false; keys are
    // suppressed then, so the compiled config stays byte-identical). weight =
    // LASSO L1 multiplier for beam-window bundles (validated 0.5; 0.2
    // over-collects), rescue = empty-flash rescue steal guard (a non-beam flash
    // must beat a beam-window match by 1/rescue to re-steal). run_ql_evt.sh
    // threads BEAMPREF_WEIGHT / BEAMPREF_RESCUE env into these for scans.
    beam_pref_weight = 0.5,
    beam_pref_rescue = 0.2,
    // Persistent post-QL intermediate output. '' (default) = off, production-identical
    // (the terminal TensorFileSink stays a dump_mode no-op). When set to a path, the
    // all-APA MABC output point-cloud tree (live+dead, cluster_t0/flash annotations,
    // opflash PC) is written there as a TensorDM tar.gz -- the input of the
    // downstream pattern-recognition job (sbnd/docs/sbnd-pattern-recognition.md).
    // run_ql_evt.sh -save-pctree points it at work/ql_evt<ID>/pctree-evt<ID>.tar.gz.
    save_tensors   = '',
)
    // Build params inside the function so all physics values are TLAs.  These
    // are the documented Q/L drift/diffusion values (matching run_clust_QL_evt.sh),
    // not wct-clustering's 4.0/8.8/35.
    local base = import 'pgrapher/experiment/sbnd/simparams.jsonnet';
    local params = base {
        lar: super.lar {
            DL:          DL * wc.cm2 / wc.s,
            DT:          DT * wc.cm2 / wc.s,
            lifetime:    lifetime * wc.ms,
            drift_speed: driftSpeed * wc.mm / wc.us,
        },
    };

    local tools_all = tools_maker(params);
    local anodes  = [tools_all.anodes[i] for i in anode_indices];
    local nanodes = std.length(anodes);

    // --- Charge sources: the toolkit's own per-event imaging output ---
    local cluster_source(fname) = g.pnode({
        type: 'ClusterFileSource',
        name: fname,
        data: { inname: fname, anodes: [wc.tn(a) for a in anodes] },
    }, nin=0, nout=1, uses=anodes);

    local active_clusters = [cluster_source('%s/icluster-apa%d-active.npz' % [input, a.data.ident]) for a in anodes];
    local masked_clusters = [cluster_source('%s/icluster-apa%d-masked.npz' % [input, a.data.ident]) for a in anodes];

    local clus_mod = import 'clus.jsonnet';
    local clus_maker = clus_mod(
        output_dir=output_dir,
        runNo=run,
        subRunNo=subrun,
        eventNo=event,
        reality=reality);
    local clus_pipes = [clus_maker.per_apa(anodes[n], dump=false)
                        for n in std.range(0, nanodes - 1)];

    // --- Q/L matching nodes ---
    // opflash TensorFileSource is built inline so it reads from the per-event
    // `input` dir (qlm.opflash_source hardcodes a bare cwd-relative filename).
    local qlm = (import 'qlmatching.jsonnet')(params);
    // SBND CPA structure-exclusion fiducial (cushion is the live tuning knob).
    local cathode_fv = (import 'cathode_fiducial.jsonnet')(cx=0.5*wc.cm, cy=0.5*wc.cm, cz=0.5*wc.cm);
    local opflash_sources = [g.pnode({
        type: 'TensorFileSource',
        name: 'opflash_src_apa%d' % anodes[n].data.ident,
        data: {
            inname: '%s/opflash_apa%d.tar.gz' % [input, anodes[n].data.ident],
            prefix: 'opflash_',
        },
    }, nin=0, nout=1) for n in std.range(0, nanodes - 1)];
    local flash_attach   = [qlm.flash_attach(n) for n in std.range(0, nanodes - 1)];
    local matching_pipes = [qlm.matching(anodes[n], clus_maker.detector_volumes([anodes[n]]),
                                         n, reality, semimodel_file,
                                         cathode_fiducial=cathode_fv.tn,
                                         calib_dump=calib_dump, cathode_diag=cathode_diag,
                                         pmt_nl=pmt_nl, auto_mask=auto_mask, beam_pref=beam_pref,
                                         beam_pref_weight=beam_pref_weight, beam_pref_rescue=beam_pref_rescue)
                            for n in std.range(0, nanodes - 1)];

    // --- Graph: per-APA matching (default) or joint multi-APA matching ---
    local graph =
        if joint then
            // active ─┐
            // masked ─┼─► clus.per_apa ─► FlashTensorToOpticalPCs ─┐
            // opflash ┘                                            ├─► QLMatching(joint) ─► MABC
            //   (other APA's flash_attach) ───────────────────────┘     (merges per-APA trees)
            local jointql = qlm.matching_joint(anodes, clus_maker.detector_volumes(anodes),
                                               reality, semimodel_file,
                                               cathode_fiducial=cathode_fv.tn,
                                               calib_dump=calib_dump, cathode_diag=cathode_diag,
                                               pmt_nl=pmt_nl, auto_mask=auto_mask, beam_pref=beam_pref,
                                               beam_pref_weight=beam_pref_weight, beam_pref_rescue=beam_pref_rescue);
            // MABC takes the single pre-merged tree directly (no PointTreeMerging).
            local clus_all = clus_maker.all_apa(anodes, dump=true, premerged=true, tensor_outname=save_tensors);
            local per_apa_pre = [g.intern(
                innodes=[active_clusters[n], masked_clusters[n], opflash_sources[n]],
                centernodes=[clus_pipes[n]],
                outnodes=[flash_attach[n]],
                edges=[
                    g.edge(active_clusters[n], clus_pipes[n], 0, 0),   // port 0 = /live
                    g.edge(masked_clusters[n], clus_pipes[n], 0, 1),   // port 1 = /dead (2-view)
                    g.edge(clus_pipes[n], flash_attach[n], 0, 0),      // port 0 = pctree
                    g.edge(opflash_sources[n], flash_attach[n], 0, 1), // port 1 = opflash
                ]
            ) for n in std.range(0, nanodes - 1)];
            g.intern(
                innodes=per_apa_pre,
                centernodes=[jointql],
                outnodes=[clus_all],
                edges=[g.edge(per_apa_pre[i], jointql, 0, i) for i in std.range(0, nanodes - 1)]
                      + [g.edge(jointql, clus_all, 0, 0)]
            )
        else
            // active ─┐
            // masked ─┼─► clus.per_apa ─► FlashTensorToOpticalPCs ─► QLMatching ─► (PointTreeMerging -> MABC)
            // opflash ┘
            local per_apa = [g.intern(
                innodes=[active_clusters[n], masked_clusters[n], opflash_sources[n]],
                centernodes=[clus_pipes[n], flash_attach[n]],
                outnodes=[matching_pipes[n]],
                edges=[
                    g.edge(active_clusters[n], clus_pipes[n], 0, 0),   // port 0 = /live
                    g.edge(masked_clusters[n], clus_pipes[n], 0, 1),   // port 1 = /dead (2-view)
                    g.edge(clus_pipes[n], flash_attach[n], 0, 0),      // port 0 = pctree
                    g.edge(opflash_sources[n], flash_attach[n], 0, 1), // port 1 = opflash
                    g.edge(flash_attach[n], matching_pipes[n], 0, 0),
                ]
            ) for n in std.range(0, nanodes - 1)];
            local clus_all = clus_maker.all_apa(anodes, dump=true, tensor_outname=save_tensors);
            g.intern(
                innodes=per_apa,
                outnodes=[clus_all],
                edges=[g.edge(per_apa[i], clus_all, 0, i)
                       for i in std.range(0, nanodes - 1)]
            );

    local app = {
        type: 'Pgrapher',
        data: { edges: g.edges(graph) },
    };

    local cmdline = {
        type: 'wire-cell',
        data: {
            plugins: ['WireCellGen', 'WireCellPgraph', 'WireCellAux', 'WireCellSio',
                      'WireCellSigProc', 'WireCellImg', 'WireCellRoot', 'WireCellTbb',
                      'WireCellClus', 'WireCellMatch'],
            apps: ['Pgrapher'],
        },
    };

    [cmdline] + g.uses(graph) + cathode_fv.configs + [app]
