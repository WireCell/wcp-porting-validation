// Validation-only PDVD light job: one population branch, raw ->
// OpDecon(WI) -> FrameFileSink dumping the "raw" and "decon" dense
// frames for comparison against the standalone python chain
// (pd_plot/wct_wi_validate.py).
//
//   wcsonnet -A input_file=raw.root -A output_file=frames.tar.bz2 \
//            -S run=39252 -S event=298567 -S branch=pmt \
//            -o cfg.json wct-light-frames.jsonnet
//
// branch: cathode | membrane | pmt (selects opch block, samples, sigma).

local g = import 'pgraph.jsonnet';
local flash = import 'pgrapher/experiment/protodunevd/flash.jsonnet';

local BRANCH = {
    cathode: { lo: 1000, hi: 1999, samples: 468864, sigma: 1.25 },
    membrane: { lo: 2000, hi: 2999, samples: 1024, sigma: 1.0 },
    pmt: { lo: 3000, hi: 3999, samples: 1024, sigma: 3.5 },
};

function(input_file, output_file, run=39252, event=298567, branch='pmt')

  local run_n = if std.type(run) == 'string' then std.parseInt(run) else run;
  local evt_n = if std.type(event) == 'string' then std.parseInt(event) else event;
  local b = BRANCH[branch];

  local src = flash.opwaveform_source(input_file, run_n, evt_n, opch_lo=b.lo, opch_hi=b.hi, name=branch);
  local decon = flash.opdecon(name=branch, samples=b.samples, wi_sigma=b.sigma);
  local sink = flash.waveform_sink(output_file, tags=['raw', 'decon'], name=branch);

  local graph = g.pipeline([src, decon, sink]);
  local app = { type: 'Pgrapher', data: { edges: g.edges(graph) } };
  local cmdline = {
    type: 'wire-cell',
    data: {
      plugins: ['WireCellRoot', 'WireCellFlash', 'WireCellGen', 'WireCellSio', 'WireCellAux', 'WireCellPgraph'],
      apps: ['Pgrapher'],
    },
  };

  [cmdline] + g.uses(graph) + [app]
