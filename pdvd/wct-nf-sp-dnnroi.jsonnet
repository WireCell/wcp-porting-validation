// DNN-ROI variant of wct-nf-sp.jsonnet for ProtoDUNE-VD.
//
// Inserts a per-anode DNN-ROI subgraph (DNN_ROI_SP model exported as
// wire-cell-data/dnnroi/pdvd/*.ts) after the standard SP pipeline.  The
// PDVD models are 4-channel and stacked-multi-plane: the two induction
// planes U+V are fed to DNNROIFindingMultiPlane in one call, the W
// collection plane is passed through from SP gauss.
//
// All other behaviour mirrors wct-nf-sp.jsonnet.  L1SP-after-DNN is not
// wired (PDVD has no L1SP-after-DNN config); L1SP inside sp.jsonnet stays
// OFF in the DNN chain so the DNN-ROI debug input tags survive.
//
// Run example (single anode, single event):
//   wire-cell -l stdout -L debug \
//     --tla-str orig_prefix="protodune-orig-frames" \
//     --tla-str sp_prefix="protodune-sp-dnnroi-frames" \
//     --tla-code anode_indices='[0]' \
//     --tla-str dnnroi_device="cpu" \
//     -c pgrapher/experiment/protodunevd/wct-nf-sp-dnnroi.jsonnet
//
// To skip the DNN step and behave like wct-nf-sp.jsonnet pass
//   --tla-code use_dnnroi='false'.

local g = import 'pgraph.jsonnet';
local wc = import 'wirecell.jsonnet';

local params = import 'pgrapher/experiment/protodunevd/params.jsonnet';

local tools_maker = import 'pgrapher/common/tools.jsonnet';
local tools_all = tools_maker(params);

function(
  orig_prefix   = 'protodune-orig-frames',  // input prefix; reads {prefix}-anode{N}.tar.bz2
  sp_prefix     = 'protodune-sp-dnnroi-frames',  // output prefix for post-DNN frames
  reality       = 'data',                   // 'data' enables the 512->500 ns Resampler on bottom anodes (n<4)
  sigoutform    = 'dense',                  // 'sparse' or 'dense'
  anode_indices = std.range(0, std.length(tools_all.anodes) - 1),
  use_freqmask  = true,
  debug_dump_path = '',
  debug_dump_groups = [],
  shield_dump_path = '',

  // DNN-ROI specific
  use_dnnroi    = true,
  dnnroi_model  = 'dnnroi/pdvd/kd_distill_transformer_4ch.ts',  // resolved via WIRECELL_PATH
  dnnroi_device = 'cpu',                    // 'cpu' or 'gpu'
  dnnroi_nchan  = 4,                        // PDVD models are 4-channel
  dnnroi_concurrency = 1,
  dnnroi_nticks = 6000,
  dnnroi_tick_per_slice = 4,                // training rebin=4
  dnnroi_output_scale = 1.0,
  dnnroi_mask_thresh = 0.5,
  dnnroi_nchunks = 1,
  dnnroi_debugfile = '',                    // if non-empty, C++ node dumps per-call .pt
)

  local tools = tools_all;
  local use_resampler = (reality == 'data');

  local base = import 'pgrapher/experiment/protodunevd/chndb-base.jsonnet';
  local chndb = [{
    type: 'OmniChannelNoiseDB',
    name: 'ocndbperfect%d' % n,
    data: base(params, tools.anodes[n], tools.field, tools.anodes[n].data.ident, use_freqmask=use_freqmask) { dft: wc.tn(tools.dft) },
    uses: [tools.anodes[n], tools.field, tools.dft],
  } for n in std.range(0, std.length(tools.anodes) - 1)];

  local nf_maker = import 'pgrapher/experiment/protodunevd/nf.jsonnet';
  local nf_pipes = [nf_maker(params, tools.anodes[n], chndb[n], tools.anodes[n].data.ident, name='nf%d' % tools.anodes[n].data.ident,
                             debug_dump_path=debug_dump_path, debug_dump_groups=debug_dump_groups,
                             shield_dump_path=shield_dump_path)
                    for n in std.range(0, std.length(tools.anodes) - 1)];

  // DNN-ROI input tags (loose_lf*, mp2_roi*, mp3_roi*, decon_charge*) require
  // OmnibusSigProc to be in debug + multi-plane-protection mode.
  local sp_maker = import 'pgrapher/experiment/protodunevd/sp.jsonnet';
  local sp_override = { sparse: sigoutform == 'sparse' }
                      + (if use_dnnroi
                         then { use_roi_debug_mode: true, use_multi_plane_protection: true }
                         else {});
  local sp = sp_maker(params, tools, sp_override);
  // L1SP OFF (l1sp_pd_mode='') so the DNN-ROI debug input tags survive.
  local sp_pipes = [sp.make_sigproc(a, l1sp_pd_mode='') for a in tools.anodes];

  // TorchService instance shared by all per-anode DNN-ROI nodes.
  local ts = {
    type: 'TorchService',
    name: 'dnnroi_pdvd',
    data: {
      model: dnnroi_model,
      device: dnnroi_device,
      concurrency: dnnroi_concurrency,
    },
  };

  local dnnroi_maker = import 'pgrapher/experiment/protodunevd/dnnroi_mp.jsonnet';
  // Per-anode debug-file basename; when empty, the C++ node skips the dump.
  local _per_anode_dbg(n) =
    if dnnroi_debugfile == '' then ''
    else '%s_anode%d' % [dnnroi_debugfile, n];
  local dnnroi_inner_pipes = [dnnroi_maker(tools.anodes[n], ts,
                                           nticks=dnnroi_nticks,
                                           tick_per_slice=dnnroi_tick_per_slice,
                                           output_scale=dnnroi_output_scale,
                                           mask_thresh=dnnroi_mask_thresh,
                                           nchunks=dnnroi_nchunks,
                                           nchan=dnnroi_nchan,
                                           debugfile=_per_anode_dbg(tools.anodes[n].data.ident))
                              for n in std.range(0, std.length(tools.anodes) - 1)];

  local resamplers_config = import 'pgrapher/common/resamplers.jsonnet';
  local load_resamplers = resamplers_config(g, wc, tools);
  local resamplers = load_resamplers.resamplers;

  // Final frame sink: write the post-DNN frame with standard SP trace tags
  // gauss%d (= DNN-ROI output, U+V from the model + W passthrough) and
  // wiener%d (= SP Wiener, carrying the per-channel threshold summary).
  // dnnroi_mp.jsonnet builds those two tags so the DNN-ROI archive is
  // structurally a standard SP archive; the non-DNN branch writes the same.
  local final_frame_sink = function(n)
    g.pnode({
      type: 'FrameFileSink',
      name: 'dnnroiframesink%d' % n,
      data: {
        outname: '%s-anode%d.tar.bz2' % [sp_prefix, n],
        tags: ['gauss%d' % n, 'wiener%d' % n],
        digitize: false,
        masks: true,
      },
    }, nin=1, nout=0);

  local per_anode_graph(n) =
    local anode = tools.anodes[n];
    local aid = anode.data.ident;

    local src = g.pnode({
      type: 'FrameFileSource',
      name: 'origframesrc%d' % aid,
      data: {
        inname: '%s-anode%d.tar.bz2' % [orig_prefix, aid],
        tags: ['orig'],
      } + (
        // Top-CRP (n>=4) orig frames are mislabeled 512 ns by the upstream
        // extraction; the top is physically 500 ns.  Re-stamp the tick at
        // read time so NF/SP run on the correct grid.  See wct-nf-sp.jsonnet.
        if use_resampler && n >= 4 then { tick: 500 * wc.ns } else {}
      ),
    }, nin=0, nout=1);

    g.pipeline(
      [src]
      + (if use_resampler && n < 4 then [resamplers[n]] else [])
      + [nf_pipes[n]]
      + [sp_pipes[n]]
      + (if use_dnnroi then [dnnroi_inner_pipes[n]] else [])
      + [final_frame_sink(aid)],
      'nfspdnn_pipe_%d' % n);

  local graphs = [per_anode_graph(n) for n in anode_indices];

  local all_edges = std.foldl(function(acc, gr) acc + g.edges(gr), graphs, []);
  local all_uses  = std.foldl(function(acc, gr) acc + g.uses(gr),  graphs, []);

  local app = {
    type: 'Pgrapher',
    data: { edges: all_edges },
  };

  local cmdline = {
    type: 'wire-cell',
    data: {
      plugins: [
        'WireCellGen',
        'WireCellPgraph',
        'WireCellSio',
        'WireCellSigProc',
        'WireCellAux',
        'WireCellPytorch',
      ],
      apps: ['Pgrapher'],
    },
  };

  [cmdline] + all_uses + [app]
