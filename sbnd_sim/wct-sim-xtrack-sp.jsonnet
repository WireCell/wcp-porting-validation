// wct-sim-xtrack-sp.jsonnet  (SBND)
//
// doc pdvd/47 -- controlled straight tracks ALONG THE DRIFT DIRECTION through the full
// Wire-Cell chain: TrackDepos -> Drifter -> DepoBagger -> [DepoTransform -> Reframer ->
// (AddNoise) -> Digitizer] -> OmnibusNoiseFilter -> OmnibusSigProc (production
// pgrapher/experiment/sbnd/sp.jsonnet) -> FrameFileSink, ONE anode per run.  Every
// time tick of such a track samples a different drift distance and its transverse
// coordinate in every plane is a point, so the SP output's transverse width vs drift
// time is measured directly (scripts/d47_sim_transverse_profile.py).
//
// Forked BY DUPLICATION from pdhd_sim/wct-sim-xtrack-sp.jsonnet (the PDHD twin); the SBND
// pieces follow cfg/pgrapher/experiment/sbnd/wct-sim-check.jsonnet (dead: fileio.jsonnet).  SP is built from
// the SAME params object as the simulation (simparams: 500 ns tick) -- feeding sim frames
// to pdhd/wct-nf-sp.jsonnet would deconvolve with the 512 ns data binning.
//
// Arm switches (all TLAs; the production-like arm is the default):
//   splat=true         DepoFluxSplat (no field/electronics response, no noise, no SP):
//                      the pure binned diffusion Gaussian = the estimator's calibration
//   noise=false        sim.signal_pipelines instead of splusn_pipelines
//   fluctuate=false    no lifetime-absorption / per-bin charge fluctuation
//   diffusion=false    lar.DL = lar.DT = 1e-4 cm2/s (NOT 0: keeps GaussianDiffusion on
//                      its regular path); the constant term alone
//   lar_DL/lar_DT/lar_drift/lar_lifetime   transport (defaults = what the PDHD track fit
//                      assumes: sbnd_track_fitting.json = simparams DL 4.0 DT 8.8 cm2/s,
//                      simparams 1.563 mm/us; lifetime 1e4 ms so charge/tick is flat)
//   nsigma             DepoTransform / DepoFluxSplat Gaussian truncation (production 3)
//   wire_filter='passthrough'   replace Wire_ind/Wire_col by an all-pass HfFilter
//   dump_rawdecon=true          also save the pre-wire-filter, pre-ROI deconvolved frame
//   seed               Random seeds[0]
//   noise_tag='x0.5'   noise spectra amplitudes scaled (ROI-threshold environment test)
//   nf=false           bypass the noise filter (OmnibusSigProc reads every trace of the frame)
//
// Usage (see run_xtrack_arms.sh):
//   wire-cell --tla-code "tracks=$(cat tracks.json)" --tla-code anode_index=1 \
//             --tla-str output_prefix=/home/xqian/tmp/xtrack/pdhd/S1 [-S ...] -c wct-sim-xtrack-sp.jsonnet
// Output: <output_prefix>-anode<N>-{raw,sp,splat}.tar.bz2 (FrameFileSink, float frames)

local g = import 'pgraph.jsonnet';
local wc = import 'wirecell.jsonnet';
local tools_maker = import 'pgrapher/common/tools.jsonnet';
local base = import 'pgrapher/experiment/sbnd/simparams.jsonnet';
local sim_maker = import 'pgrapher/experiment/sbnd/sim.jsonnet';
local perfect = import 'pgrapher/experiment/sbnd/chndb-base.jsonnet';
local nf_maker = import 'pgrapher/experiment/sbnd/nf.jsonnet';
local sp_maker = import 'pgrapher/experiment/sbnd/sp.jsonnet';

function(
    tracks = [],                 // [{tail:[x,y,z], head:[x,y,z], charge:-500}] in cm; charge<=0 = electrons PER 0.1 mm step
    anode_index = 0,             // 0 east / 1 west TPC
    output_prefix = '/home/xqian/tmp/xtrack/sbnd/S1',
    noise = true,
    fluctuate = false,           // sbnd/simparams.jsonnet production value
    diffusion = true,
    lar_DL = 4.0,                // cm2/s
    lar_DT = 8.8,                // cm2/s
    lar_drift = 1.563,           // mm/us
    lar_lifetime = 1.0e4,        // ms
    splat = false,
    nsigma = 3,
    wire_filter = 'production',  // | 'passthrough'
    dump_rawdecon = false,
    l1sp_mode = '',              // ignored on SBND (no L1SPFilterPD)
    seed = 0,
    save_raw = true,
    noise_tag = '',              // 'x0.5' | 'x2': amplitude-scaled noise spectra (d47_scale_noise.py) in /home/xqian/tmp/xtrack/noise
    nf = true,                   // false: skip OmnibusNoiseFilter (noiseless arms on SBND: its per-channel
                                 // filters flag every flat channel bad regardless of the RMS cuts)
)

local params = base {
    lar: super.lar {
        DL: (if diffusion then lar_DL else 1e-4) * wc.cm2 / wc.s,
        DT: (if diffusion then lar_DT else 1e-4) * wc.cm2 / wc.s,
        drift_speed: lar_drift * wc.mm / wc.us,
        lifetime: lar_lifetime * wc.ms,
    },
    sim: super.sim { fluctuate: fluctuate },
    files: super.files { noise: if noise_tag == '' then super.noise else '/home/xqian/tmp/xtrack/noise/' + std.strReplace(super.noise, '.json.bz2', '.' + noise_tag + '.json.bz2') },
};
local tools = tools_maker(params);
local sim = sim_maker(params, tools);
local anode = tools.anodes[anode_index];
local ident = anode.data.ident;
// the Reframer's output frame time (= tick0_time up to the response_nticks rounding)
local frame_t0 = params.sim.ductor.start_time + params.sim.reframer.tbin * params.daq.tick;

local depos = sim.tracks([{
    time: 0,
    charge: t.charge,
    ray: { tail: wc.point(t.tail[0], t.tail[1], t.tail[2], wc.cm),
           head: wc.point(t.head[0], t.head[1], t.head[2], wc.cm) },
} for t in tracks], step = 0.1 * wc.mm);

local sink(label, tags) = g.pnode({
    type: 'FrameFileSink',
    name: 'sink_%s_%d' % [label, ident],
    data: {
        outname: '%s-anode%d-%s.tar.bz2' % [output_prefix, ident, label],
        tags: tags,
        digitize: false,
        masks: false,
    },
}, nin=1, nout=0);

// ---- S0: truth splat (no response, no noise, no SP) ----
local splat_node = g.pnode({
    type: 'DepoFluxSplat',
    name: 'splat%d' % ident,
    data: {
        anode: wc.tn(anode),
        field_response: wc.tn(tools.fields[0]),
        drift_speed: params.lar.drift_speed,         // override the FR file's speed
        response_plane: params.det.response_plane,   // override the FR file's origin
        tick: params.daq.tick,
        window_start: frame_t0,
        window_duration: params.daq.nticks * params.daq.tick,
        reference_time: 0,
        // sparse=true (one trace per depo patch; FrameFileSink SUMS overlapping traces).  The
        // dense accumulator (DepoFluxSplat.cxx DenseAccumulator::add) silently DROPS a patch whose
        // tick range lies inside the trace already held -- 7 of every 8 depos here (2026-09-05).
        sparse: true,
        nsigma: nsigma,
        smear_long: 0.0,
        smear_tran: 0.0,
    },
}, nin=1, nout=1, uses=[anode, tools.fields[0]]);

// ---- S1..: sim (+noise) -> NF -> SP ----
local sim_pipe = if noise then sim.splusn_pipelines[anode_index] else sim.signal_pipelines[anode_index];
// The "perfect" channel DB still carries the DATA bad-channel list and a min_rms_cut
// (PDHD 10 ADC) that flags every NOISELESS channel as bad and zeroes it in NF.  In a
// controlled simulation neither belongs: no bad channels, no RMS cuts (last entry wins).
local all_chans = std.range(ident * 5638, ident * 5638 + 5637);   // sbnd/chndb-base.jsonnet first entry
local chndb = {
    type: 'OmniChannelNoiseDB',
    name: 'ocndbperfect%d' % ident,
    data: perfect(params, anode, tools.field, ident,
                  rms_cuts=[{ channels: all_chans, min_rms_cut: 0.0, max_rms_cut: 1.0e9 }])
          { bad: [], dft: wc.tn(tools.dft) },
    uses: [anode, tools.field, tools.dft],
};
local nf_node = nf_maker(params, anode, chndb, ident, name='nf%d' % ident);

local wire_pass = {
    type: 'HfFilter', name: 'Wire_pass',
    data: { max_freq: 1, power: 2, flag: false, sigma: 1e9 },   // exp(-0.5 (f/1e9)^2) == 1.0 in float
};
local sp_override = { sparse: false }
    + (if wire_filter == 'passthrough' then { Wire_filters: ['Wire_pass', 'Wire_pass', 'Wire_pass'] } else {})
    + (if dump_rawdecon then { rawdecon_tag: 'rawdecon%d' % ident } else {});   // sbnd/sp.jsonnet has no dump_rawdecon arg
local sp = sp_maker(params, tools, sp_override);
local sp_pipe = sp.make_sigproc(anode);   // SBND has no L1SPFilterPD; l1sp_mode is accepted and ignored

local raw_tap = g.fan.tap('FrameFanout', sink('raw', ['raw%d' % ident]), 'rawtap%d' % ident);
local sp_sink = sink('sp', ['gauss%d' % ident, 'wiener%d' % ident]
                           + (if dump_rawdecon then ['rawdecon%d' % ident] else []));

local pipe = if splat
    then g.pipeline([splat_node, sink('splat', [])], 'splatpipe%d' % ident)
    else g.pipeline([sim_pipe] + (if nf then [nf_node] + (if save_raw then [raw_tap] else []) else []) + [sp_pipe, sp_sink], 'xpipe%d' % ident);

local graph = g.pipeline([depos, sim.drifter, sim.make_bagger(), pipe]);

local patched = [
    if c.type == 'DepoTransform' then c { data+: { nsigma: nsigma } }
    else if c.type == 'Random' then c { data+: { seeds: [seed, 1, 2, 3, 4] } }
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

[cmdline] + patched + (if wire_filter == 'passthrough' then [wire_pass] else []) + [app]
