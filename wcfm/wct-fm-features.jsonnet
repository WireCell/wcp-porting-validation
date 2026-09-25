// wcfm/wct-fm-features.jsonnet -- the FM feature stage for the DUNE FD-HD 1x2x6 "workspace"
// (wcfm/docs/01 sec 4.2, wcfm/docs/04, F3).  Reads per-anode SP frames sim-frames-anode<N>.tar.bz2
// (the wct-sim-iso-track-nf-sp.jsonnet output), packs each wire plane into the FM input canvas
// (FMFeatureExtract: 4-tick sum x 0.25, VIEW_NORM log map, channel-order rows, tight bbox
// floored at 64), runs the scripted MBV3 student through one shared TorchService, and writes
// the feature sidecar
//
//   <output_dir>/fm-features-anode<N>.tar.gz   (TensorFileSink, prefix 'fm_': per plane
//        coords (N,2) i4 [channel ident, slice] + feat_half (N,128) u2 (or feat f4), set metadata)
//
//   wire-cell --tla-str input_prefix=work/000001_1/sim-frames --tla-code anode_indices=[10] \
//             --tla-str output_dir=work/000001_1 -c wct-fm-features.jsonnet
//
// Nothing here touches the imaging / clustering chain; the sidecar is consumed by F4 later.

local g = import 'pgraph.jsonnet';
local wc = import 'wirecell.jsonnet';
local P = import 'wcfm_params.jsonnet';
local params = P.params;

local tools_maker = import 'pgrapher/common/tools.jsonnet';
local tools_all = tools_maker(params);

function(
  input_prefix = 'sim-frames',
  anode_indices = std.range(0, std.length(tools_all.anodes) - 1),
  output_dir = '',
  // model file, resolved through WIRECELL_PATH (wire-cell-data), and its sha256 for provenance
  fm_model = 'fm/dune10kt-1x2x6/kd_uni_mbv3_a1_inf01.ts',
  fm_model_sha = '',
  fm_device = 'cpu',          // TorchService: cpu | gpu | gpuN
  // packing constants (wcfm/docs/04 sec 1: the pack builder's 0.005 x 50 x sum of 4 ticks)
  input_scale = 0.25,
  tick_span = P.tick_span,
  active_threshold = 0.0,
  bbox_pad = 1,
  max_dense_pixels = 0,       // 0 = never tile
  halo = 64,
  store_half = true,          // f16 bits on disk (u2); false = f4 (the parity gate arm)
  planes = [0, 1, 2],
)

  local anodes = [tools_all.anodes[i] for i in anode_indices];

  local torch = {
    type: 'TorchService',
    name: 'fm',
    data: { model: fm_model, device: fm_device },
  };

  local per_anode_graph(anode) =
    local aid = anode.data.ident;
    local src = g.pnode({
      type: 'FrameFileSource',
      name: 'frame_source_anode%d' % aid,
      data: {
        inname: '%s-anode%d.tar.bz2' % [input_prefix, aid],
        tags: ['gauss%d' % aid],
      },
    }, nin=0, nout=1);
    local fm = g.pnode({
      type: 'FMFeatureExtract',
      name: 'fm_anode%d' % aid,
      data: {
        anode: wc.tn(anode),
        forward: wc.tn(torch),
        input_tag: 'gauss%d' % aid,
        planes: planes,
        tick0: 0,
        nticks: params.daq.nticks,
        tick_span: tick_span,
        input_scale: input_scale,
        input_offset: 0.0,
        active_threshold: active_threshold,
        // VIEW_NORM (WC_FM_DINO/dino/transforms.py): U, V, W
        view_norm: [[2.77, 144977.0], [2.97, 157944.0], [3.75, 83861.2]],
        min_canvas: 64,
        bbox_pad: bbox_pad,
        feature_dim: 128,
        max_dense_pixels: max_dense_pixels,
        halo: halo,
        store_half: store_half,
        provenance: { model: fm_model, model_sha256: fm_model_sha, arch: 'mbv3_uni', device: fm_device },
      },
    }, nin=1, nout=1, uses=[anode, torch]);
    local sink = g.pnode({
      type: 'TensorFileSink',
      name: 'fm_sink_anode%d' % aid,
      data: {
        outname: '%s/fm-features-anode%d.tar.gz' % [output_dir, aid],
        prefix: 'fm_',
      },
    }, nin=1, nout=0);
    g.pipeline([src, fm, sink], 'fm_graph_anode%d' % aid);

  local graphs = [per_anode_graph(a) for a in anodes];
  local all_edges = std.foldl(function(acc, gr) acc + g.edges(gr), graphs, []);
  local all_uses  = std.foldl(function(acc, gr) acc + g.uses(gr),  graphs, []);

  local app = { type: 'Pgrapher', data: { edges: all_edges } };
  local cmdline = {
    type: 'wire-cell',
    data: {
      plugins: ['WireCellGen', 'WireCellPgraph', 'WireCellSio', 'WireCellAux', 'WireCellPytorch'],
      apps: ['Pgrapher'],
    },
  };

  [cmdline] + all_uses + [app]
