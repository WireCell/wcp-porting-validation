// DNN-ROI variant of wct-nf-sp.jsonnet.
//
// Inserts a DNN-ROI subgraph (model in wire-cell-data/dnnroi/pdhd/)
// after the standard SP pipeline, per anode.  Per-plane sequential
// wiring via dnnroi_pp.jsonnet: two DNNROIFinding calls per APA at
// (1, 6, 800, 1500) each, sharing one TorchService.
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
  sp_prefix     = 'protodunehd-sp-dnnroi-frames',
  reality       = 'data',
  anode_indices = std.range(0, std.length(tools_all.anodes) - 1),
  use_freqmask  = true,
  debug_dump_path = '',
  debug_dump_groups = [],
  // L1SP inside sp.jsonnet stays off in the DNN chain (the DNN-ROI debug
  // input tags would be dropped through its FrameMerger).  L1SP-after-DNN
  // is wired separately via the use_l1sp_dnn TLA below.
  l1sp_pd_mode = '',
  l1sp_pd_dump_path = '',
  l1sp_pd_wf_dump_path = '',
  l1sp_pd_dump_all_rois = false,
  l1sp_pd_adj_enable = true,
  l1sp_pd_adj_max_hops = 3,
  dump_rawdecon = false,

  // APA0 W-plane induction-path ROI-refinement tune (sp.jsonnet
  // apa0_w_roi_tune).  Default true in the DNN chain: recovers the
  // gap-prone APA0 W signal (no BreakROI split, wider ShrinkROI pad,
  // lower refine threshold; applies to APA0 U+W, V untouched).  Set
  // false to recover the pre-tune, bit-identical APA0 SP behaviour.
  apa0_w_roi_tune = true,

  // Prolonged-W-signal fix passthroughs (sp.jsonnet roi_mad_rms /
  // w_col_break_roi_tune; both default true there).  Set false for the
  // bit-identical pre-fix SP.  See docs/sp-w-collection-roi-break.md.
  sp_roi_mad_rms = true,
  sp_w_col_break_roi_tune = true,

  // PRE-FLIP coherent-noise grouping override (run-027409 etc. decoded with the
  // pre-2025-06-30 channel map).  Default false = use the toolkit's latest
  // (post-flip, map-derived) groups -> bit-identical for colleagues.  The run
  // scripts set this true when the local `.coh_preflip` sentinel is present.
  // See pdhd-coh-groups-preflip.jsonnet.
  coh_groups_preflip = false,

  // DNN-ROI specific
  use_dnnroi    = true,
  // Default = FP32 best KD (6-ch).  Resolved via WIRECELL_PATH.
  //   FP32 best KD:    pipe_distill_transformer_6ch.ts       (Dice 0.9107)
  //   INT8 best QAT:   pipe_qat_transformer_6ch_int8.ts      (Dice 0.8900, CPU only)
  //   Legacy 3-ch:     CP43.ts                               (also set dnnroi_nchan=3)
  // The shell wrapper run_nf_sp_dnnroi_evt.sh -P fp32|int8 sets this via tla-str.
  dnnroi_model  = 'dnnroi/pdhd/pipe_distill_transformer_6ch.ts',
  // dnnroi_model = 'dnnroi/pdhd/pipe_qat_transformer_6ch_int8.ts',  // INT8: requires dnnroi_device='cpu'
  dnnroi_device = 'cpu',                    // 'cpu' or 'gpu'.  INT8 graph is CPU-only.
  dnnroi_nchan  = 6,                        // 6 = production KD/QAT models; 3 = legacy CP43.ts
  dnnroi_concurrency = 1,
  dnnroi_nticks = 6000,
  dnnroi_tick_per_slice = 4,                // training rebin=4
  dnnroi_output_scale = 1.0,
  dnnroi_mask_thresh = 0.2,
  dnnroi_nchunks = 1,
  dnnroi_debugfile = '',   // if non-empty, C++ node dumps per-call .pt

  // L1SP-after-DNN: when true, run L1SPFilterPD after the DNN-ROI subgraph,
  // feeding the DNN output as L1SP's sigtag.  See dnnroi_l1sp.md.
  use_l1sp_dnn      = true,
  l1sp_pd_adj_enable    = true,
  l1sp_pd_adj_max_hops  = 3,

  // ── ML L1SP tagger (l1sp_pd_mode == 'dnn') ─────────────────────────
  // When l1sp_pd_mode == 'dnn', L1SPFilterPD replaces its 5-arm
  // heuristic decide_trigger with a TorchScript model call.  Polarity
  // is still sign(raw_asym_wide); only the fire/no-fire cut changes.
  // Model file is resolved via WIRECELL_PATH and ships with
  // wire-cell-data/l1sp/pdhd/.  Default threshold 0.5, matching the
  // 2026-05-25 deployed value (raised 0.10 → 0.35 → 0.5).  See
  // l1sp_dl_tagger/docs/13-l1sp-deploy-thresh0p5.md and
  // l1sp_dl_tagger/experiments/stage_a_pu_round4/deploy_round4.md.
  l1sp_pd_dnn_model        = 'l1sp/pdhd/l1sp_dnn_pdhd_v1.ts',
  l1sp_pd_dnn_device       = 'cpu',
  l1sp_pd_dnn_concurrency  = 1,
  l1sp_pd_dnn_threshold    = 0.5,
  l1sp_pd_dnn_window_ticks = 256,
  // Run DNN veto on adjacency-promoted ROIs too (default true since
  // 2026-05-25).  Without it the heuristic-driven adjacency chain
  // bypasses the DNN entirely.
  l1sp_pd_adj_dnn_veto     = true,
  // When non-empty, L1SPFilterPD writes per-ROI (waveform, scalars,
  // score, polarity, fired) to one NPZ per operator() call so the
  // l1sp_dl_tagger Python validator can assert score parity.
  l1sp_pd_dnn_debug_path   = '',
  // ── Loose-heur pre-filter overrides (DNN-chain only) ───────────────
  // Defaults match C++ defaults (trad-chain values). Loosen these in
  // the DNN chain because (1) DNN ROIs are typically shorter/weaker
  // than trad ROIs for the same signal -- which fails the default
  // pre-filters -- and (2) the DNN L1SP tagger refines downstream, so
  // heuristic false positives are vetoed there. See
  // l1sp_dl_tagger/experiments/stage_a_pu_round4/veto_mode_design.md
  // Phase-B/C analysis.
  l1sp_pd_gmax_min         = 1500.0,
  l1sp_pd_min_length       = 30,
  l1sp_pd_energy_frac_thr  = 0.66,

  // SP ROI-stage diagnosis sink.  When true, the OmnibusSigProc debug +
  // multi-plane-protection modes are forced ON (matching the production
  // DNN-chain sp_override, so the SP-internal ROI processing is identical)
  // and the per-stage ROI tags (tight_lf, cleanup_roi, break_roi_1st/2nd,
  // shrink_roi, extend_roi, decon_charge, mp2/mp3_roi) are added to the
  // FrameFileSink.  Use together with use_dnnroi=false: the DNN-ROI and
  // L1SP envelopes contain FrameMergers whose mergemaps drop these tags,
  // so the SP frame must reach the sink directly.  See
  // docs/sp-w-collection-breakroi.md.
  sp_roi_debug_sink = false,
)

  local tools = tools_all;
  local use_resampler = (reality == 'data');

  local base = import 'pgrapher/experiment/pdhd/chndb-base.jsonnet';
  local coh_preflip = import 'pdhd-coh-groups-preflip.jsonnet';
  local chndb = [{
    type: 'OmniChannelNoiseDB',
    name: 'ocndbperfect%d' % n,
    data: base(params, tools.anodes[n], tools.field, n, use_freqmask=use_freqmask) { dft: wc.tn(tools.dft) }
          + (if coh_groups_preflip
             then { groups: coh_preflip.groups(n), femb_negpulse_groups: coh_preflip.negpulse_groups }
             else {}),
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
                      + (if use_dnnroi || sp_roi_debug_sink
                         then { use_roi_debug_mode: true, use_multi_plane_protection: true }
                         else {});
  local sp = sp_maker(params, tools, sp_override);
  // When DNN-ROI is in the chain, sp_pipes must NOT include an
  // L1SPFilterPD node: L1SP belongs strictly AFTER DNN-ROI, inside the
  // l1sp_after_dnnroi envelope.  An L1SP node placed before DNN-ROI
  // drops the SP auxiliary tags (decon_charge, loose_lf, mp2_roi,
  // mp3_roi, tight_lf) that DNN-ROI's 6-channel input model needs, so
  // DNN-ROI ends up fed mostly zeros and outputs zero on the U/V planes
  // it's responsible for.  Forcing l1sp_pd_mode='' here keeps
  // sp.make_sigproc emitting bare SP only.  When DNN-ROI is bypassed
  // (use_dnnroi=false), fall back to the legacy mix that lets l1sp run
  // inside sp_pipes if the caller requested it.
  local sp_pipes = [sp.make_sigproc(a,
                                    l1sp_pd_mode=if use_dnnroi then '' else l1sp_pd_mode,
                                    l1sp_pd_dump_path=l1sp_pd_dump_path,
                                    l1sp_pd_wf_dump_path=l1sp_pd_wf_dump_path,
                                    l1sp_pd_dump_all_rois=l1sp_pd_dump_all_rois,
                                    l1sp_pd_adj_enable=l1sp_pd_adj_enable,
                                    l1sp_pd_adj_max_hops=l1sp_pd_adj_max_hops,
                                    apa0_w_roi_tune=apa0_w_roi_tune,
                                    roi_mad_rms=sp_roi_mad_rms,
                                    w_col_break_roi_tune=sp_w_col_break_roi_tune,
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

  local dnnroi_maker = import 'pgrapher/experiment/pdhd/dnnroi_pp.jsonnet';
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
                                           debugfile=_per_anode_dbg(n))
                              for n in std.range(0, std.length(tools.anodes) - 1)];

  // L1SP-after-DNN envelope.  When use_l1sp_dnn=true, the envelope wraps
  // SP + DNN-ROI + L1SP into a single per-anode subgraph (the FrameSplitter
  // must sit BEFORE SP since OmnibusSigProc drops raw%d from its output).
  // In that mode, the top-level pipeline below uses the envelope IN PLACE
  // OF the separate sp_pipe + sp_frame_tap + dnnroi_pipe sequence.
  local l1sp_dnn_maker = import 'pgrapher/experiment/pdhd/l1sp_after_dnnroi.jsonnet';

  // TorchService for the ML L1SP tagger.  Built when l1sp_pd_mode is
  // 'dnn' OR 'hybrid' (hybrid also calls DNN, just gated by heuristic).
  // null otherwise so the heuristic-only path doesn't pull libtorch.
  local l1sp_torch_service = if (l1sp_pd_mode == 'dnn' || l1sp_pd_mode == 'hybrid') then {
    type: 'TorchService',
    name: 'l1sp_dnn_pdhd',
    data: {
      model:       l1sp_pd_dnn_model,
      device:      l1sp_pd_dnn_device,
      concurrency: l1sp_pd_dnn_concurrency,
    },
  } else null;

  local resamplers_config = import 'pgrapher/common/resamplers.jsonnet';
  local load_resamplers = resamplers_config(g, wc, tools);
  local resamplers = load_resamplers.resamplers;

  // Final frame sink: write the post-DNN frame (one trace tag per plane
  // plus the merged 'dnnspN' frame tag) when DNN-ROI is enabled,
  // otherwise just write the SP frame.
  local final_frame_sink = function(n)
    g.pnode({
      type: 'FrameFileSink',
      name: 'dnnroiframesink%d' % n,
      data: {
        outname: '%s-anode%d.tar.bz2' % [sp_prefix, n],
        tags: (if use_dnnroi
               then (if use_l1sp_dnn
                     then ['gauss%d' % n, 'wiener%d' % n, 'raw%d' % n]
                     else ['dnnsp%d' % n, 'dnnsp%du' % n, 'dnnsp%dv' % n, 'dnnsp%dw' % n,
                           'gauss%d' % n, 'wiener%d' % n])
               else ['gauss%d' % n, 'wiener%d' % n])
              // Per-stage ROI diagnosis tags (effective with use_dnnroi=false
              // only — the DNN/L1SP FrameMergers drop them otherwise).
              + (if sp_roi_debug_sink
                 then ['tight_lf%d' % n, 'loose_lf%d' % n, 'cleanup_roi%d' % n,
                       'break_roi_1st%d' % n, 'break_roi_2nd%d' % n,
                       'shrink_roi%d' % n, 'extend_roi%d' % n,
                       'decon_charge%d' % n, 'mp3_roi%d' % n, 'mp2_roi%d' % n]
                 else []),
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
    // disabled, the legacy two-step (SP → DNN) is used directly.
    local sp_dnn_l1sp_segment =
      if use_dnnroi && use_l1sp_dnn
      then [l1sp_dnn_maker(tools.anodes[n], sp_pipes[n], dnnroi_inner_pipes[n],
                           tools, params,
                           l1sp_pd_adj_enable=l1sp_pd_adj_enable,
                           l1sp_pd_adj_max_hops=l1sp_pd_adj_max_hops,
                           l1sp_pd_dump_mode=l1sp_pd_mode,
                           l1sp_pd_dump_path=l1sp_pd_dump_path,
                           l1sp_pd_wf_dump_path=l1sp_pd_wf_dump_path,
                           l1sp_pd_dump_all_rois=l1sp_pd_dump_all_rois,
                           l1sp_pd_torch_service=l1sp_torch_service,
                           l1sp_pd_dnn_threshold=l1sp_pd_dnn_threshold,
                           l1sp_pd_adj_dnn_veto=l1sp_pd_adj_dnn_veto,
                           l1sp_pd_dnn_window_ticks=l1sp_pd_dnn_window_ticks,
                           l1sp_pd_dnn_debug_path=l1sp_pd_dnn_debug_path,
                           l1sp_pd_gmax_min=l1sp_pd_gmax_min,
                           l1sp_pd_min_length=l1sp_pd_min_length,
                           l1sp_pd_energy_frac_thr=l1sp_pd_energy_frac_thr)]
      else [sp_pipes[n]]
           + (if use_dnnroi then [dnnroi_inner_pipes[n]] else []);

    g.pipeline(
      [src]
      + (if use_resampler then [resamplers[n]] else [])
      + [nf_pipes[n]]
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
