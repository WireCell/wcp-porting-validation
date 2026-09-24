// wct-sim-iso-track-nf-sp.jsonnet -- DUNE FD-HD 1x2x6 "workspace" standalone simulation
// (wcfm/docs/01 sec 6 W2, wcfm/docs/02 sec 2).
//
// TrackDepos (the iso-track gun tracklists of gen_iso_tracks.py) -> Drifter -> DepoBagger
// -> DepoSetFanout -> per anode [DepoTransform -> Reframer -> AddNoise -> Digitizer ->
// OmnibusNoiseFilter -> OmnibusSigProc -> FrameFileSink (gauss+wiener, masks)] and one
// DepoFileSink port carrying the DRIFTED depos (+ their undrifted priors) for the
// BlobDepoFill truth stage (W4).
//
// Forked BY DUPLICATION from pdhd_sim/wct-sim-xtrack-sp.jsonnet (untouched) with the
// dune10kt-1x2x6 sim/nf/sp modules (pdhd's sp.jsonnet needs three field files + L1SP).
// None of the in-tree dune10kt wct-sim-* entries is usable: wct-sim-ideal-sn-nf-sp has no
// NF and its tag-less FrameFanin merges the gauss and wiener traces (doc 02 sec 2).
//
// NF: the workspace nf.jsonnet is the MicroBooNE mbOneChannelNoise on chndb-base, whose
// channel_info still carries MicroBooNE numbers.  The `chndb_fix` entries below (last
// mention wins) set the nominal ADC baselines this detector's Digitizer actually adds
// (params.adc.baselines through fullscale/resolution), drop the MicroBooNE U/V frequency
// masks and hand-made response, and disable the RMS cuts (a controlled simulation has no
// bad channels).  The NF is needed because OmnibusSigProc has no pedestal subtraction.
//
// Usage (run_sim_evt.sh):
//   wire-cell --tla-code "tracks=$(python3 -c '...')" --tla-code anode_indices=[0,2] \
//             --tla-str output_prefix=work/000001_1/sim-frames \
//             --tla-str depo_outname=work/000001_1/sim-depos.tar.bz2 \
//             --tla-code seed=1001 -c wct-sim-iso-track-nf-sp.jsonnet
// Output: <output_prefix>-anode<N>.tar.bz2 (float gauss<N>/wiener<N> frames, chanmask
//         "bad" if NF flagged anything), <depo_outname> (depo_data_<id>/depo_info_<id>).

local g = import 'pgraph.jsonnet';
local wc = import 'wirecell.jsonnet';
local tools_maker = import 'pgrapher/common/tools.jsonnet';
local P = import 'wcfm_params.jsonnet';
local sim_maker = import 'pgrapher/experiment/dune10kt-1x2x6/sim.jsonnet';
local chndb_base = import 'pgrapher/experiment/dune10kt-1x2x6/chndb-base.jsonnet';
local nf_maker = import 'pgrapher/experiment/dune10kt-1x2x6/nf.jsonnet';
local sp_maker = import 'pgrapher/experiment/dune10kt-1x2x6/sp.jsonnet';

function(
    tracks = [],                // [{tail:[x,y,z], head:[x,y,z], charge:-500}] in cm; charge<=0 = electrons PER step
    anode_indices = [0],        // indices into tools.anodes (== anode idents for this wires file)
    output_prefix = 'sim-frames',
    depo_outname = 'sim-depos.tar.bz2',
    seed = 0,                   // Random seeds[0] (noise + fluctuations)
    noise = true,               // false: sim.signal_pipelines (no AddNoise)
    step_mm = 0.1,              // TrackDepos step; charge is per step
    save_raw = false,           // also save the NF output (raw<N>) frames
)

local params = P.params;
local tools = tools_maker(params);
local sim = sim_maker(params, tools);

local depos = sim.tracks([{
    time: 0,
    charge: t.charge,
    ray: { tail: wc.point(t.tail[0], t.tail[1], t.tail[2], wc.cm),
           head: wc.point(t.head[0], t.head[1], t.head[2], wc.cm) },
} for t in tracks], step = step_mm * wc.mm);

// ADC count the Digitizer adds per plane: (baseline_V - fullscale_lo) / (hi - lo) * (2^res - 1)
local adc_baseline(iplane) =
    (params.adc.baselines[iplane] - params.adc.fullscale[0])
    / (params.adc.fullscale[1] - params.adc.fullscale[0]) * (std.pow(2, params.adc.resolution) - 1);

local sp = sp_maker(params, tools);

local anode_pipe(i) =
    local anode = tools.anodes[i];
    local ident = anode.data.ident;
    local ch0 = ident * 2560;
    local chndb_fix = [
        { channels: std.range(ch0, ch0 + 800 - 1),          nominal_baseline: adc_baseline(0), freqmasks: [], response: {}, response_offset: 0.0, min_rms_cut: 0.0, max_rms_cut: 1.0e9 },
        { channels: std.range(ch0 + 800, ch0 + 1600 - 1),   nominal_baseline: adc_baseline(1), freqmasks: [], response: {}, response_offset: 0.0, min_rms_cut: 0.0, max_rms_cut: 1.0e9 },
        { channels: std.range(ch0 + 1600, ch0 + 2560 - 1),  nominal_baseline: adc_baseline(2), min_rms_cut: 0.0, max_rms_cut: 1.0e9 },
    ];
    local chndb = {
        type: 'OmniChannelNoiseDB',
        name: 'ocndbsim%d' % ident,
        data: chndb_base(params, anode, tools.field, ident, rms_cuts=chndb_fix) { bad: [], dft: wc.tn(tools.dft) },
        uses: [anode, tools.field, tools.dft],
    };
    local nf = nf_maker(params, anode, chndb, ident, name='nf%d' % ident);
    local sp_node = sp.make_sigproc(anode, name='sp%d' % ident);
    local sink(label, tags) = g.pnode({
        type: 'FrameFileSink',
        name: 'sink_%s_%d' % [label, ident],
        data: {
            outname: if label == 'sp' then '%s-anode%d.tar.bz2' % [output_prefix, ident]
                     else '%s-%s-anode%d.tar.bz2' % [output_prefix, label, ident],
            tags: tags,
            digitize: false,
            masks: true,
        },
    }, nin=1, nout=0);
    local raw_tap = g.fan.tap('FrameFanout', sink('raw', ['raw%d' % ident]), 'rawtap%d' % ident);
    local sim_pipe = if noise then sim.splusn_pipelines[i] else sim.signal_pipelines[i];
    g.pipeline([sim_pipe, nf] + (if save_raw then [raw_tap] else [])
               + [sp_node, sink('sp', ['gauss%d' % ident, 'wiener%d' % ident])], 'simpipe%d' % ident);

local pipes = [anode_pipe(i) for i in anode_indices];

local depo_sink = g.pnode({
    type: 'DepoFileSink',
    name: 'deposink',
    data: { outname: depo_outname },
}, nin=1, nout=0);

// DepoSetFanout to the per-anode pipes + the depo file (every pipe ends in a sink).
local fan = g.fan.sink('DepoSetFanout', pipes + [depo_sink], 'simfan');

local graph = g.pipeline([depos, sim.drifter, sim.make_bagger(), fan]);

local patched = [
    if c.type == 'Random' then c { data+: { seeds: [seed, 1, 2, 3, 4] } }
    else c
    for c in g.uses(graph)
];
local app = { type: 'Pgrapher', data: { edges: g.edges(graph) } };
local cmdline = {
    type: 'wire-cell',
    data: {
        plugins: ['WireCellGen', 'WireCellPgraph', 'WireCellSio', 'WireCellSigProc',
                  'WireCellAux', 'WireCellTbb'],
        apps: ['Pgrapher'],
    },
};

[cmdline] + patched + [app]
