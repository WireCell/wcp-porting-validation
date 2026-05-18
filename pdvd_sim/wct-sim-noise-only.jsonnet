// wct-sim-noise-only.jsonnet
//
// Electronics-noise-ONLY simulation for ProtoDUNE Vertical Drift (PDVD).
// No track, no signal, no noise filtering.  Generates electronics noise on
// every channel of each selected anode and writes the raw DIGITIZED frame
// (ADC counts, frame tag 'orig<ident>') to one tar.bz2 per anode.
//
// Per-anode graph:
//   SilentNoise -> Reframer(fill:0) -> AddNoise -> Digitizer -> FrameFileSink
//
// The Reframer rebuilds the full channel set from the anode and zero-fills
// (gen/src/Reframer.cxx), so every real channel gets a trace; AddNoise then
// adds an independent noise waveform to each (gen/src/AddNoise.cxx).  The
// bottom (ident<4) / top (ident>=4) bifurcation -- noise spectra file, ADC
// baselines and fullscale -- replicates sim.jsonnet.
//
// Usage:
//   wire-cell \
//     --tla-str  output_prefix=work/noise/all/pdvd-noise-sim \
//     --tla-code anode_indices='[0,1,2,3,4,5,6,7]' \
//     -c wct-sim-noise-only.jsonnet
//
// Parameters
// ----------
//   anode_indices : jsonnet array of anode indices to simulate (default all 0..7)
//   output_prefix : prefix for output tar.bz2 files
//                   output: <output_prefix>-anode<ident>.tar.bz2

local g  = import 'pgraph.jsonnet';
local wc = import 'wirecell.jsonnet';

local tools_maker = import 'pgrapher/common/tools.jsonnet';
local params      = import 'pgrapher/experiment/protodunevd/simparams.jsonnet';
local tools_all   = tools_maker(params);

function(
    anode_indices = std.range(0, std.length(tools_all.anodes) - 1),
    output_prefix = 'work/noise/pdvd-noise-sim',
)

// -- selected anodes ---------------------------------------------------------
local anodes     = [tools_all.anodes[i] for i in anode_indices];
local anode_iota = std.range(0, std.length(anodes) - 1);

// -- silent (zero) frame source ----------------------------------------------
// SilentNoise emits one empty frame; the Reframer below rebuilds the full
// channel set from the anode.  nchannels MUST be 0: any trace SilentNoise
// emits carries a sequential channel id (0,1,...) that Reframer keeps as an
// extra trace -- a channel id not on this anode, which AddNoise then cannot
// map to a wire plane ("illegal plane index").
local silent(ident) = g.pnode({
    type: 'SilentNoise',
    name: 'silent%d' % ident,
    data: {
        noutputs: 1,
        nchannels: 0,
    },
}, nin=0, nout=1);

// -- reframer: produce a full zero frame over ALL anode channels -------------
// fill:0.0 + no input traces => every anode channel gets a zero waveform.
// tbin:0/toffset:0 because there is no early-arriving signal to chop.
local reframer(anode) = g.pnode({
    type: 'Reframer',
    name: 'reframer-' + anode.name,
    data: {
        anode: wc.tn(anode),
        tags: [],
        fill: 0.0,
        tbin: 0,
        toffset: 0,
        nticks: params.daq.nticks,   // 10000 (simparams override)
    },
}, nin=1, nout=1, uses=[anode]);

// -- empirical noise model (PDVD: bottom spectra ident<4, top ident>=4) ------
local noise_model(anode) = {
    type: 'EmpiricalNoiseModel',
    name: 'empiricalnoise-' + anode.name,
    data: {
        anode: wc.tn(anode),
        dft: wc.tn(tools_all.dft),
        chanstat: '',
        spectra_file: params.files.noises[if anode.data.ident < 4 then 0 else 1],
        nsamples: params.daq.nticks,
        period: params.daq.tick,
        wire_length_scale: 1.0 * wc.cm,
    },
    uses: [anode, tools_all.dft],
};

// -- AddNoise pnode ----------------------------------------------------------
local add_noise(anode, model) = g.pnode({
    type: 'AddNoise',
    name: 'addnoise-' + anode.name,
    data: {
        rng: wc.tn(tools_all.random),
        dft: wc.tn(tools_all.dft),
        model: wc.tn(model),
        nsamples: params.daq.nticks,
        replacement_percentage: 0.02,
    },
}, nin=1, nout=1, uses=[tools_all.random, tools_all.dft, model]);

// -- digitizer (PDVD baselines/fullscale bifurcation, as in sim.jsonnet) -----
local digitizer(anode) = g.pnode({
    type: 'Digitizer',
    name: 'digitizer-' + anode.name,
    data: params.adc {
        anode: wc.tn(anode),
        frame_tag: 'orig%d' % anode.data.ident,
        baselines: if anode.data.ident < 4
                   then [1003.4*wc.millivolt, 1003.4*wc.millivolt, 507.7*wc.millivolt]
                   else [1.0*wc.volt, 1.0*wc.volt, 1.0*wc.volt],
        fullscale: if anode.data.ident < 4
                   then [0.2*wc.volt, 1.6*wc.volt]
                   else [0.0*wc.volt, 2.0*wc.volt],
    },
}, nin=1, nout=1, uses=[anode]);

// -- FrameFileSink: save digitized 'orig<ident>' frame -----------------------
// Digitizer tags the FRAME (not traces) 'orig<ident>'; FrameFileSink's
// tagged_traces() falls back to all traces when the tag is a frame tag.
// digitize:false -> write the already-ADC float values as float32 as-is.
local frame_sink(anode) = g.pnode({
    type: 'FrameFileSink',
    name: 'noisesink%d' % anode.data.ident,
    data: {
        outname: '%s-anode%d.tar.bz2' % [output_prefix, anode.data.ident],
        tags: ['orig%d' % anode.data.ident],
        digitize: false,
        masks: false,
    },
}, nin=1, nout=0);

// -- per-anode pipelines -----------------------------------------------------
local pipes = [
    local a = anodes[n];
    local nm = noise_model(a);
    g.pipeline([
        silent(a.data.ident),
        reframer(a),
        add_noise(a, nm),
        digitizer(a),
        frame_sink(a),
    ], 'noise_pipe_%d' % a.data.ident)
    for n in anode_iota
];

// The N pipelines are independent (each has its own SilentNoise head): a
// forest.  Pgrapher accepts a disconnected graph; flatten edges/uses directly.
local app = {
    type: 'Pgrapher',
    data: { edges: std.flattenArrays([g.edges(p) for p in pipes]) },
};

local cmdline = {
    type: 'wire-cell',
    data: {
        plugins: ['WireCellGen', 'WireCellPgraph', 'WireCellSio', 'WireCellTbb'],
        apps: ['Pgrapher'],
    },
};

[cmdline] + std.flattenArrays([g.uses(p) for p in pipes]) + [app]
