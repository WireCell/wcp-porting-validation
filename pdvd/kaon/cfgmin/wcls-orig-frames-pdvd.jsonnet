// Decode-only WC/LS job: dump the per-anode "orig" frames of a PDVD event
// exactly where xning's wcls-nf-out.jsonnet (toolkit 941a261c) taps them --
// wclsRawFrameSource -> per-anode ChannelSelector -> FrameFileSink, before any
// resampler/NF/SP -- without running NF/SP inside lar.  Written for doc
// pdvd/118 because the local WCT libs that recipe needs no longer load in the
// SL7 container (GLIBC_2.29).  Same art tag, tick rule, channel split,
// outname, tags, digitize/masks flags as the 941a261c tap.
local wc = import 'wirecell.jsonnet';
local g = import 'pgraph.jsonnet';

local raw_input_label = std.extVar('raw_input_label');
local use_resampler = std.extVar('use_resampler');

// protodunevd/funcs.jsonnet anode_channels (identical in 941a261c and HEAD)
local anode_channels(n) = {
  local crp = (n - n % 2) / 2,
  local cru0 = std.range(0, 475) + std.range(952, 1427) + std.range(1904, 2487),
  local cru1 = std.range(476, 951) + std.range(1428, 1903) + std.range(2488, 3071),
  local channels = if n % 2 == 0 then cru0 else cru1,
  ret: [x + 3072 * crp for x in channels],
}.ret;

local anodes = std.range(0, 7);

local source = g.pnode({
  type: 'wclsRawFrameSource',
  name: '',
  data: {
    art_tag: raw_input_label,
    frame_tags: ['orig'],
    tick: if use_resampler == 'true' then 512 * wc.ns else 500 * wc.ns,
  },
}, nin=0, nout=1);

local fanout = g.pnode({
  type: 'FrameFanout',
  name: 'origfanout',
  data: { multiplicity: std.length(anodes) },
}, nin=1, nout=std.length(anodes));

local sinks = [
  g.pipeline([
    g.pnode({
      type: 'ChannelSelector',
      name: 'chsel%d' % n,
      data: { channels: anode_channels(n) },
    }, nin=1, nout=1),
    g.pnode({
      type: 'FrameFileSink',
      name: 'origframesink%d' % n,
      data: {
        outname: 'protodune-orig-frames-anode%d.tar.bz2' % n,
        tags: ['orig'],
        digitize: false,
        masks: false,
      },
    }, nin=1, nout=0),
  ], 'origpipe%d' % n)
  for n in anodes
];

local graph = g.intern(innodes=[source],
                       centernodes=[fanout],
                       outnodes=sinks,
                       edges=[g.edge(source, fanout)] +
                             [g.edge(fanout, sinks[n], n, 0) for n in anodes]);

local app = { type: 'Pgrapher', data: { edges: g.edges(graph) } };
g.uses(graph) + [app]
