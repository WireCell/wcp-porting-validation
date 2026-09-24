// wcfm/wct-img-all.jsonnet -- standalone imaging for the DUNE FD-HD 1x2x6 "workspace"
// (wcfm/docs/02 sec 3).  Reads per-anode SP frames sim-frames-anode<N>.tar.bz2 (the
// wct-sim-iso-track-nf-sp.jsonnet output), images BOTH faces of every anode (they are
// real, opposite-drift faces here -- no PDHD "null-face restore"), and, when `depos` is
// given, attaches the W4 BlobDepoFill truth tiers (wcfm/img.jsonnet).
//
// Forked BY DUPLICATION from toolkit cfg/pgrapher/experiment/pdhd/wct-img-all.jsonnet.
//
//   wire-cell --tla-str input_prefix=work/000001_1/sim-frames --tla-code anode_indices=[8] \
//             --tla-str output_dir=work/000001_1 --tla-str depos=work/000001_1/sim-depos.tar.bz2 \
//             -c wct-img-all.jsonnet
// Output: <output_dir>/clusters-apa-anode<N>-ms-{active,masked}.tar.gz and, with depos,
//         <output_dir>/clusters-{tru0,tru}-anode<N>-ms-active.tar.gz

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
  nticks = 0,
  // drifted-depo file for the truth tiers ('' = plain imaging, PDHD chain)
  depos = '',
  // BlobDepoFill depo-time -> frame-time offset (WCT units); default from wcfm_params
  time_offset = P.depofill_time_offset,
  // BlobDepoFill primary plane (0,1,2); diagnostics only
  pindex = 2,
)

  local anodes = [tools_all.anodes[i] for i in anode_indices];

  local img = import 'img.jsonnet';
  local img_maker = img({
      nthreshold: [1e-6, 1e-6, 1e-6],
      depos: depos,
      depofill_speed: P.drift_speed,
      depofill_time_offset: time_offset,
      depofill_nsigma: P.depofill_nsigma,
      depofill_pindex: pindex,
  });

  local per_anode_graph(anode) =
    local aid = anode.data.ident;
    local src = g.pnode({
      type: 'FrameFileSource',
      name: 'frame_source_anode%d' % aid,
      data: {
        inname: '%s-anode%d.tar.bz2' % [input_prefix, aid],
        tags: ['gauss%d' % aid, 'wiener%d' % aid],
      },
    }, nin=0, nout=1);
    local reframer = g.pnode({
      type: 'Reframer',
      name: 'reframer_anode%d' % aid,
      data: {
        anode: wc.tn(anode),
        tags: ['gauss%d' % aid, 'wiener%d' % aid],
        tbin: 0,
        nticks: if nticks > 0 then nticks else params.daq.nticks,
        fill: 0.0,
        keep_masks: true,
      },
    }, nin=1, nout=1, uses=[anode]);
    g.pipeline([src, reframer, img_maker.per_anode(anode, "multi", output_dir)],
               'img_graph_anode%d' % aid);

  local graphs = [per_anode_graph(a) for a in anodes];
  local all_edges = std.foldl(function(acc, gr) acc + g.edges(gr), graphs, []);
  local all_uses  = std.foldl(function(acc, gr) acc + g.uses(gr),  graphs, []);

  local app = { type: 'Pgrapher', data: { edges: all_edges } };
  local cmdline = {
    type: 'wire-cell',
    data: {
      plugins: ['WireCellGen', 'WireCellPgraph', 'WireCellSio', 'WireCellSigProc', 'WireCellImg', 'WireCellClus'],
      apps: ['Pgrapher'],
    },
  };

  [cmdline] + all_uses + [app]
