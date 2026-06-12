// NF-only pipeline for ProtoDUNE-VD (no signal processing).
//
// Reads per-anode orig frames and runs noise filtering only.
// Output file naming matches run-nfsp (wct-nf-sp.jsonnet):
//   {raw_prefix}-anode{N}.tar.bz2
//
// The maskmap parameter controls which NF mask tags are saved and under
// what names in the output frame.  Pass as a jsonnet object via --tla-code,
// e.g.:
//   --tla-code 'maskmap={chirp:"chirp",noisy:"noisy"}'
// The default keeps each tag under its own name (no renaming to "bad").
// To merge everything to "bad" as the old behaviour did:
//   --tla-code 'maskmap={chirp:"bad",noisy:"bad"}'

local g = import 'pgraph.jsonnet';
local wc = import 'wirecell.jsonnet';

local params = import 'pgrapher/experiment/protodunevd/params.jsonnet';

local tools_maker = import 'pgrapher/common/tools.jsonnet';
local tools_all = tools_maker(params);

function(
  orig_prefix   = 'protodune-orig-frames',    // input prefix; reads {prefix}-anode{N}.tar.bz2
  raw_prefix    = 'protodune-sp-frames-raw',  // output prefix for NF (raw) frames
  reality       = 'data',                     // 'data' enables the 512->500 ns Resampler on bottom anodes (n<4); 'sim' disables it
  anode_indices = std.range(0, std.length(tools_all.anodes) - 1),
  use_freqmask  = true,
  // maskmap: keys are NF tag names, values are the tag names written to output.
  // Default: preserve each tag with its original name so they stay distinct.
  maskmap       = {  noisy: 'noisy' },
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
  local nf_pipes = [nf_maker(params, tools.anodes[n], chndb[n], tools.anodes[n].data.ident,
                             name='nf%d' % tools.anodes[n].data.ident,
                             maskmap=maskmap)
                    for n in std.range(0, std.length(tools.anodes) - 1)];

  local resamplers_config = import 'pgrapher/common/resamplers.jsonnet';
  local load_resamplers = resamplers_config(g, wc, tools);
  local resamplers = load_resamplers.resamplers;

  // Save NF output (raw) frame per anode
  local raw_frame_sink = function(anode_ident)
    g.pnode({
      type: 'FrameFileSink',
      name: 'rawframesink%d' % anode_ident,
      data: {
        outname: '%s-anode%d.tar.bz2' % [raw_prefix, anode_ident],
        tags: ['raw%d' % anode_ident],
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
      },
    }, nin=0, nout=1);

    g.pipeline(
      [src]
      + (if use_resampler && n < 4 then [resamplers[n]] else [])
      + [nf_pipes[n]]
      + [raw_frame_sink(aid)],
      'nf_pipe_%d' % n);

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
      ],
      apps: ['Pgrapher'],
    },
  };

  [cmdline] + all_uses + [app]
