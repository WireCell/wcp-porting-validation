// wct-sim-depofile-nf-sp.jsonnet -- DUNE FD-HD 1x2x6 "workspace" standalone simulation from an
// EXTERNAL depo file (wcfm/docs/07 E1: the WC_FM_Sim pilot GENIE+G4 depos, events 101+).
//
// DepoFileSource (un-drifted G4 depos, one depo_data_0/depo_info_0 set, electrons negative)
// -> DepoSetDrifter (the same Drifter as the gun sim) -> DepoSetFanout -> per anode
// [DepoTransform -> Reframer -> AddNoise -> Digitizer -> OmnibusNoiseFilter -> OmnibusSigProc
// -> FrameFileSink (gauss+wiener, masks)] and one DepoFileSink port carrying the DRIFTED
// depos (+ their undrifted priors) for the BlobDepoFill truth stage.
//
// Forked BY DUPLICATION from wct-sim-iso-track-nf-sp.jsonnet (untouched; events 1-100 keep
// that file).  The only differences: the `tracks`/`step_mm` arguments are replaced by
// `depo_inname`; the TrackDepos source, per-depo Drifter and DepoBagger head of the graph is
// replaced by DepoFileSource -> DepoSetDrifter (a DepoFileSource emits a whole IDepoSet, the
// per-depo Drifter cannot take it; DepoSetDrifter wraps the identical Drifter component, so
// drift speed / DL / DT / lifetime / fluctuation are those of events 1-100).  The bagger's
// readout gate is not applied: depos drifting in after the 3 ms readout simply fall outside
// the frame and outside every slice.  Everything below the fan is byte-identical to the gun
// sim.  Output names are identical too, so run_img_evt.sh / run_fm_evt.sh / gnn_dataset.py
// run unchanged on these events.
//
// Usage (run_sim_depo_evt.sh):
//   wire-cell --tla-str depo_inname=work/000001_201/e1-depos-in.tar.bz2 \
//             --tla-code anode_indices=[0,2,4] --tla-str output_prefix=work/000001_201/sim-frames \
//             --tla-str depo_outname=work/000001_201/sim-depos.tar.bz2 --tla-code seed=1201 \
//             -c wct-sim-depofile-nf-sp.jsonnet
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
    depo_inname = 'e1-depos-in.tar.bz2',   // WCT depo file: depo_data_0/depo_info_0, t ns, q<0 electrons, xyz mm
    anode_indices = [0],        // indices into tools.anodes (== anode idents for this wires file)
    output_prefix = 'sim-frames',
    depo_outname = 'sim-depos.tar.bz2',
    seed = 0,                   // Random seeds[0] (noise + fluctuations)
    noise = true,               // false: sim.signal_pipelines (no AddNoise)
    save_raw = false,           // also save the NF output (raw<N>) frames
)

local params = P.params;
local tools = tools_maker(params);
local sim = sim_maker(params, tools);

local depos = g.pnode({
    type: 'DepoFileSource',
    name: 'e1deposrc',
    data: { inname: depo_inname, scale: 1.0 },   // the file already carries electrons < 0
}, nin=0, nout=1);

// DepoSetDrifter wraps the gun sim's Drifter (same component, same parameters).
local setdrifter = g.pnode({
    type: 'DepoSetDrifter',
    name: 'e1setdrifter',
    data: { drifter: 'Drifter' },      // the gun sim's Drifter carries no name (type-only tn)
}, nin=1, nout=1, uses=[sim.drifter]);

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

local graph = g.pipeline([depos, setdrifter, fan]);

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
