// Dump raw + deconvolved cathode full-stream waveforms for ONE event.
// Validation input for saturation_recovery_study.py (NumPy replica of the
// OpDecon wiener-inspired filter must match this to <1%).  Read-only wrt
// production configs: same builders as wct-light-reco.jsonnet, no OpRoi /
// OpHit (the study integrates the decon directly).
//
//   wcsonnet -A input_file=... -A output_file=out.tar.gz \
//            -S run=39252 -S event=298567 -o cfg.json wct-decon-dump.jsonnet
//   wire-cell -l stderr -c cfg.json

local g = import 'pgraph.jsonnet';
local flash = import 'pgrapher/experiment/protodunevd/flash.jsonnet';

local FULLSTREAM_SAMPLES = 468864;

function(input_file, output_file, run=39252, event=298567)

  local run_n = if std.type(run) == 'string' then std.parseInt(run) else run;
  local evt_n = if std.type(event) == 'string' then std.parseInt(event) else event;

  local src = flash.opwaveform_source(input_file, run_n, evt_n,
                                      opch_lo=1000, opch_hi=1999, name='cath');
  local decon = flash.opdecon(name='cath', samples=FULLSTREAM_SAMPLES, wi_sigma=1.25);
  local sink = flash.waveform_sink(output_file, tags=['raw', 'decon'], name='cath');

  local graph = g.intern(
    innodes=[src], centernodes=[decon], outnodes=[sink],
    edges=[g.edge(src, decon), g.edge(decon, sink)],
  );

  local app = { type: 'Pgrapher', data: { edges: g.edges(graph) } };
  local cmdline = {
    type: 'wire-cell',
    data: {
      plugins: ['WireCellRoot', 'WireCellFlash', 'WireCellGen',
                'WireCellSio', 'WireCellAux', 'WireCellPgraph'],
      apps: ['Pgrapher'],
    },
  };
  [cmdline] + g.uses(graph) + [app]
