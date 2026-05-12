// DNN-ROI variant of wct-nf-sp.jsonnet.
//
// Inserts a DNN-ROI subgraph (DNN_ROI_SP/checkpoints model exported as
// wire-cell-data/dnnroi/pdhd/CP43.ts) after the standard SP pipeline,
// per anode.  Two wiring modes selectable via TLA `dnnroi_mode`:
//   'pp' (default): per-plane sequential — two DNNROIFinding calls
//                   at (1, 3, 800, 1500) each, sharing CP43.ts.
//   'mp':           stacked U+V via DNNROIFindingMultiPlane at
//                   (1, 3, 1600, 1500); legacy, kept for comparison.
// All other behaviour is identical to wct-nf-sp.jsonnet.
//
// Run example (single anode, single event):
//   wire-cell -l stdout -L debug \
//     --tla-str orig_prefix="protodunehd-orig-frames" \
//     --tla-str sp_prefix="protodunehd-sp-dnnroi-frames" \
//     --tla-code anode_indices='[0]' \
//     --tla-str dnnroi_device="cpu" \
//     -c pgrapher/experiment/pdhd/wct-nf-sp-dnnroi.jsonnet
//
// To skip the DNN step and behave like wct-nf-sp.jsonnet pass
//   --tla-code use_dnnroi='false'.

local g = import 'pgraph.jsonnet';
local wc = import 'wirecell.jsonnet';

local params = import 'pgrapher/experiment/pdhd/params.jsonnet';

local tools_maker = import 'pgrapher/common/tools.jsonnet';
local tools_all = tools_maker(params);

function(
  orig_prefix   = 'protodunehd-orig-frames',
  raw_prefix    = 'protodunehd-sp-frames-raw',
  sp_only_prefix = 'protodunehd-sp-frames',  // baseline (pre-DNN) SP tap output
  sp_prefix     = 'protodunehd-sp-dnnroi-frames',
  reality       = 'data',
  anode_indices = std.range(0, std.length(tools_all.anodes) - 1),
  use_freqmask  = true,
  debug_dump_path = '',
  debug_dump_groups = [],
  // L1SPFilterPD's FrameMerger only preserves raw / gauss / wiener / rawdecon
  // trace tags.  The DNN-ROI input tags (loose_lf, mp2_roi, mp3_roi,
  // decon_charge) get dropped through it, so the L1SP chain must be bypassed
  // here.  If the user wants L1SP active *and* DNN-ROI, the merger rules in
  // sp.jsonnet need to be extended to pass the debug tags through.
  l1sp_pd_mode = '',
  l1sp_pd_dump_path = '',
  l1sp_pd_wf_dump_path = '',
  l1sp_pd_dump_all_rois = false,
  l1sp_pd_adj_enable = true,
  l1sp_pd_adj_max_hops = 3,
  dump_rawdecon = false,

  // DNN-ROI specific
  use_dnnroi    = true,
  dnnroi_mode   = 'pp',                     // 'pp' = per-plane sequential (default)
                                            // 'mp' = stacked multi-plane (legacy)
  dnnroi_model  = 'dnnroi/pdhd/CP43.ts',    // resolved via WIRECELL_PATH
  dnnroi_device = 'cpu',                    // 'cpu' or 'gpu'
  dnnroi_concurrency = 1,
  dnnroi_nticks = 6000,
  dnnroi_tick_per_slice = 4,                // training rebin=4
  dnnroi_output_scale = 1.0,
  dnnroi_mask_thresh = 0.5,
  dnnroi_nchunks = 1,
  dnnroi_debugfile = '',   // if non-empty, C++ node dumps per-call .pt

  // L1SP-after-DNN: when true, run L1SPFilterPD after the DNN-ROI subgraph,
  // feeding the DNN output as L1SP's sigtag.  See dnnroi_l1sp.md.
  use_l1sp_dnn      = true,
  l1sp_pd_adj_enable    = true,
  l1sp_pd_adj_max_hops  = 3,
)

  local tools = tools_all;
  local use_resampler = (reality == 'data');

  local base = import 'pgrapher/experiment/pdhd/chndb-base.jsonnet';
  local chndb = [{
    type: 'OmniChannelNoiseDB',
    name: 'ocndbperfect%d' % n,
    data: base(params, tools.anodes[n], tools.field, n, use_freqmask=use_freqmask) { dft: wc.tn(tools.dft) },
    uses: [tools.anodes[n], tools.field, tools.dft],
  } for n in std.range(0, std.length(tools.anodes) - 1)];

  local nf_maker = import 'pgrapher/experiment/pdhd/nf.jsonnet';
  local nf_pipes = [nf_maker(params, tools.anodes[n], chndb[n], n, name='nf%d' % n,
                             debug_dump_path=debug_dump_path, debug_dump_groups=debug_dump_groups)
                    for n in std.range(0, std.length(tools.anodes) - 1)];

  local sp_maker = import 'pgrapher/experiment/pdhd/sp.jsonnet';
  // DNN-ROI input tags (loose_lf*, mp2_roi*, mp3_roi*, decon_charge*) require
  // OmnibusSigProc to be in debug + multi-plane-protection mode.
  local sp_override = { sparse: false }
                      + (if use_dnnroi
                         then { use_roi_debug_mode: true, use_multi_plane_protection: true }
                         else {});
  local sp = sp_maker(params, tools, sp_override);
  local sp_pipes = [sp.make_sigproc(a,
                                    l1sp_pd_mode=l1sp_pd_mode,
                                    l1sp_pd_dump_path=l1sp_pd_dump_path,
                                    l1sp_pd_wf_dump_path=l1sp_pd_wf_dump_path,
                                    l1sp_pd_dump_all_rois=l1sp_pd_dump_all_rois,
                                    l1sp_pd_adj_enable=l1sp_pd_adj_enable,
                                    l1sp_pd_adj_max_hops=l1sp_pd_adj_max_hops,
                                    dump_rawdecon=dump_rawdecon)
                    for a in tools.anodes];

  // TorchService instance shared by all per-anode DNN-ROI nodes.
  local ts = {
    type: 'TorchService',
    name: 'dnnroi_pdhd',
    data: {
      model: dnnroi_model,
      device: dnnroi_device,
      concurrency: dnnroi_concurrency,
    },
  };

  local dnnroi_maker =
    if dnnroi_mode == 'pp'
    then import 'pgrapher/experiment/pdhd/dnnroi_pp.jsonnet'
    else import 'pgrapher/experiment/pdhd/dnnroi_mp.jsonnet';
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
                                           debugfile=_per_anode_dbg(n))
                              for n in std.range(0, std.length(tools.anodes) - 1)];

  // L1SP-after-DNN envelope.  When use_l1sp_dnn=true, the envelope wraps
  // SP + DNN-ROI + L1SP into a single per-anode subgraph (the FrameSplitter
  // must sit BEFORE SP since OmnibusSigProc drops raw%d from its output).
  // In that mode, the top-level pipeline below uses the envelope IN PLACE
  // OF the separate sp_pipe + sp_frame_tap + dnnroi_pipe sequence.
  local l1sp_dnn_maker = import 'pgrapher/experiment/pdhd/l1sp_after_dnnroi.jsonnet';

  local resamplers_config = import 'pgrapher/common/resamplers.jsonnet';
  local load_resamplers = resamplers_config(g, wc, tools);
  local resamplers = load_resamplers.resamplers;

  local raw_frame_tap = function(n)
    g.fan.tap('FrameFanout',
      g.pnode({
        type: 'FrameFileSink',
        name: 'rawframesink%d' % n,
        data: {
          outname: '%s-anode%d.tar.bz2' % [raw_prefix, n],
          tags: ['raw%d' % n],
          digitize: false,
          masks: true,
        },
      }, nin=1, nout=0),
      'rawframetap%d' % n);

  // SP-only frame tap (the standard gauss/wiener output, before DNN-ROI).
  // Useful for direct SP vs DNN-ROI comparison.
  local sp_frame_tap = function(n)
    g.fan.tap('FrameFanout',
      g.pnode({
        type: 'FrameFileSink',
        name: 'spframesink%d' % n,
        data: {
          outname: '%s-anode%d.tar.bz2' % [sp_only_prefix, n],
          tags: ['gauss%d' % n, 'wiener%d' % n]
                + (if dump_rawdecon then ['rawdecon%d' % n] else []),
          digitize: false,
          masks: true,
        },
      }, nin=1, nout=0),
      'spframetap%d' % n);

  // Final frame sink: write the post-DNN frame (one trace tag per plane
  // plus the merged 'dnnspN' frame tag) when DNN-ROI is enabled,
  // otherwise just write the SP frame.
  local final_frame_sink = function(n)
    g.pnode({
      type: 'FrameFileSink',
      name: 'dnnroiframesink%d' % n,
      data: {
        outname: '%s-anode%d.tar.bz2' % [sp_prefix, n],
        tags: if use_dnnroi
              then (if use_l1sp_dnn
                    then ['gauss%d' % n, 'wiener%d' % n, 'raw%d' % n]
                    else ['dnnsp%d' % n, 'dnnsp%du' % n, 'dnnsp%dv' % n, 'dnnsp%dw' % n,
                          'gauss%d' % n, 'wiener%d' % n])
              else ['gauss%d' % n, 'wiener%d' % n],
        digitize: false,
        masks: true,
      },
    }, nin=1, nout=0);

  local per_anode_graph(n) =
    local src = g.pnode({
      type: 'FrameFileSource',
      name: 'origframesrc%d' % n,
      data: {
        inname: '%s-anode%d.tar.bz2' % [orig_prefix, n],
        tags: [],
      },
    }, nin=0, nout=1);

    // L1SP-after-DNN envelope wraps SP + DNN + L1SP into one node; when
    // disabled, the legacy three-step (SP → SP-tap → DNN) is used directly.
    local sp_dnn_l1sp_segment =
      if use_dnnroi && use_l1sp_dnn
      then [l1sp_dnn_maker(tools.anodes[n], sp_pipes[n], dnnroi_inner_pipes[n],
                           tools, params,
                           sp_frame_tap=sp_frame_tap(n),
                           l1sp_pd_adj_enable=l1sp_pd_adj_enable,
                           l1sp_pd_adj_max_hops=l1sp_pd_adj_max_hops)]
      else [sp_pipes[n], sp_frame_tap(n)]
           + (if use_dnnroi then [dnnroi_inner_pipes[n]] else []);

    g.pipeline(
      [src]
      + (if use_resampler then [resamplers[n]] else [])
      + [nf_pipes[n]]
      + [raw_frame_tap(n)]
      + sp_dnn_l1sp_segment
      + [final_frame_sink(n)],
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
