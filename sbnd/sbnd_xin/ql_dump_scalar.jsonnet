// THROWAWAY: per-event SBND Q/L matching that dumps each per-APA QLMatching
// output pctree (incl. the per-cluster "cluster_scalar" PC: flash,
// matched_flash_gid, cluster_t0, flag_main_cluster, flag_associated_cluster,
// ident) to a tar, so we can read the complete per-cluster flash+flag table.
// Mirrors wct-clus-matching-perevt.jsonnet but replaces the all-APA MABC with a
// TensorFileSink on each matching_pipe output. Safe to delete.

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
    DL             = 4.0,
    DT             = 8.8,
    lifetime       = 6.0,
    driftSpeed     = 1.563,
)
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
        eventNo=event);
    local clus_pipes = [clus_maker.per_apa(anodes[n], dump=false)
                        for n in std.range(0, nanodes - 1)];

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
                                         cathode_fiducial=cathode_fv.tn)
                            for n in std.range(0, nanodes - 1)];

    // Per-APA dump sink: writes the QLMatching output pctree (with cluster_scalar).
    local dump_sinks = [g.pnode({
        type: 'TensorFileSink',
        name: 'scalardump_apa%d' % anodes[n].data.ident,
        data: {
            outname: '%s/scalardump-apa%d.tar.gz' % [output_dir, anodes[n].data.ident],
            prefix: 'clustering_',
            dump_mode: false,
        },
    }, nin=1, nout=0) for n in std.range(0, nanodes - 1)];

    local per_apa = [g.intern(
        innodes=[active_clusters[n], masked_clusters[n], opflash_sources[n]],
        centernodes=[clus_pipes[n], flash_attach[n], matching_pipes[n]],
        outnodes=[dump_sinks[n]],
        edges=[
            g.edge(active_clusters[n], clus_pipes[n], 0, 0),
            g.edge(masked_clusters[n], clus_pipes[n], 0, 1),
            g.edge(clus_pipes[n], flash_attach[n], 0, 0),
            g.edge(opflash_sources[n], flash_attach[n], 0, 1),
            g.edge(flash_attach[n], matching_pipes[n], 0, 0),
            g.edge(matching_pipes[n], dump_sinks[n], 0, 0),
        ]
    ) for n in std.range(0, nanodes - 1)];

    local graph = g.intern(innodes=per_apa, edges=[]);

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
