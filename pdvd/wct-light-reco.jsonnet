// WCT-native PDVD light reconstruction over ALL 40 OpDets (51 DAPHNE
// channels) in ONE processing: three population branches from the same
// raw_waveform file MERGE at the OpHit level, so a single OpFlashFinder
// builds all-PD flashes (bin_width 1 us; DAPHNE channels ganged to the
// 40 OpDet PE columns via pdvd-opch-map.json).
//
//   cathode branch (opch 10xx, full stream 468800/468864):
//     PDVDOpWaveformSource -> OpDecon(WI sigma=1.25, samples=468864)
//     -> OpRoi -> OpHitFinder(intag=decon_roi, fixed_ped_sigma)
//   membrane branch (opch 20xx, 1024-tick snippets):
//     PDVDOpWaveformSource -> OpDecon(WI sigma=1.0) -> OpHitFinder
//   PMT branch (opch 30xx, 1024-tick snippets):
//     PDVDOpWaveformSource -> OpDecon(WI sigma=3.5) -> OpHitFinder
//   all three ophits -> OpHitMerge -> OpFlashFinder
//     -> opflash_pdvd-wct.tar.gz
//
// All branches use the Wiener-INSPIRED filter (F(0)=1: PE normalization
// exact and filter-independent) and share the source's common tick
// origin (the event's earliest record over ALL channels), so their
// OpHit times are directly coincident -- pdvd/docs/pdvd-light-filter.md
// and pdvd-flash-dt.md.
//
// Compile/run (see run_light_evt.sh):
//   wcsonnet -A input_file=raw.root -A output_dir=... \
//            -S run=39252 -S event=298567 \
//            -o cfg.json wct-light-reco.jsonnet
//   wire-cell -l stderr -L debug -c cfg.json

local g = import 'pgraph.jsonnet';
local flash = import 'pgrapher/experiment/protodunevd/flash.jsonnet';

// Cathode full-stream record length: 468864 = max(nsamp); the shorter
// 468800-sample records are zero-padded by OpDecon.
local FULLSTREAM_SAMPLES = 468864;

function(input_file, output_dir='.', run=39252, event=298567, offset_us=0,
         offset_bot_us=null, offset_top_us=null,
         cath_thresh=3.7, mem_thresh=4.0, pmt_thresh=2.2,
         cath_ped_sigma=0.75, sat_pad=1024,
         // Drop OpHits overlapping a DAPHNE 14-bit rail (+- sat_pad).
         // Default true = legacy behavior, compiled config byte-identical.
         // false keeps hits from railed pulses (clipped => PE underestimated
         // instead of exactly 0).  See docs/qlmatch/
         // pdvd-pd-mapping-investigation.md sec.6-7 (42% of bright flashes
         // lose >=1 cathode channel to the veto).
         veto_saturation=true)

  local run_n = if std.type(run) == 'string' then std.parseInt(run) else run;
  local evt_n = if std.type(event) == 'string' then std.parseInt(event) else event;
  local off_us = if std.type(offset_us) == 'string' then std.parseJson(offset_us) else offset_us;
  // Per-crate light<->charge offsets from the rawwf trigoff tree (see
  // run_light_evt.sh): chain_light_t0 - charge_{bde,tde}_window_start, us.
  // Stamped verbatim into the archive metadata for downstream Q/L matching.
  local off_bot_us = if std.type(offset_bot_us) == 'string' then std.parseJson(offset_bot_us) else offset_bot_us;
  local off_top_us = if std.type(offset_top_us) == 'string' then std.parseJson(offset_top_us) else offset_top_us;
  local sat_pad_n = if std.type(sat_pad) == 'string' then std.parseInt(sat_pad) else sat_pad;
  local cath_th = if std.type(cath_thresh) == 'string' then std.parseJson(cath_thresh) else cath_thresh;
  local mem_th = if std.type(mem_thresh) == 'string' then std.parseJson(mem_thresh) else mem_thresh;
  local pmt_th = if std.type(pmt_thresh) == 'string' then std.parseJson(pmt_thresh) else pmt_thresh;
  local cath_ps = if std.type(cath_ped_sigma) == 'string' then std.parseJson(cath_ped_sigma) else cath_ped_sigma;
  local veto_sat = if std.type(veto_saturation) == 'string' then std.parseJson(veto_saturation) else veto_saturation;

  // --- cathode branch (opch 10xx, continuous full streams) ---
  local cath_src = flash.opwaveform_source(input_file, run_n, evt_n, opch_lo=1000, opch_hi=1999, name='cath');
  local cath_decon = flash.opdecon(name='cath', samples=FULLSTREAM_SAMPLES, wi_sigma=1.25,
                                   detect_saturation=true, saturation_pad=sat_pad_n);
  local cath_roi = flash.oproi(name='cath');
  local cath_hit = flash.ophit(name='cath', hit_threshold=cath_th, intag='decon_roi',
                               fixed_ped_sigma=cath_ps, veto_saturation=veto_sat);

  // --- membrane XA branch (opch 20xx, snippets; top+bottom walls share
  //     sigma=1.0, the top-wall pickup sets the threshold) ---
  local mem_src = flash.opwaveform_source(input_file, run_n, evt_n, opch_lo=2000, opch_hi=2999, name='mem');
  local mem_decon = flash.opdecon(name='mem', wi_sigma=1.0,
                                  detect_saturation=true, saturation_pad=sat_pad_n);
  local mem_hit = flash.ophit(name='mem', hit_threshold=mem_th, veto_saturation=veto_sat);

  // --- PMT branch (opch 30xx, snippets) ---
  local pmt_src = flash.opwaveform_source(input_file, run_n, evt_n, opch_lo=3000, opch_hi=3999, name='pmt');
  local pmt_decon = flash.opdecon(name='pmt', wi_sigma=3.5,
                                  detect_saturation=true, saturation_pad=sat_pad_n);
  local pmt_hit = flash.ophit(name='pmt', hit_threshold=pmt_th, veto_saturation=veto_sat);

  // --- merge OpHits and build flashes once over all 40 OpDets ---
  local merge = flash.ophit_merge(name='allpd', multiplicity=3, meta_port=0);
  local opflash_finder = flash.opflash_finder(offset_us=off_us,
                                              offset_bot_us=off_bot_us,
                                              offset_top_us=off_top_us);
  local fl_sink = flash.opflash_sink('%s/opflash_pdvd-wct.tar.gz' % output_dir, name='allpd');

  local graph = g.intern(
    innodes=[cath_src, mem_src, pmt_src],
    centernodes=[cath_decon, cath_roi, cath_hit,
                 mem_decon, mem_hit, pmt_decon, pmt_hit,
                 merge, opflash_finder],
    outnodes=[fl_sink],
    edges=[
      g.edge(cath_src, cath_decon),
      g.edge(cath_decon, cath_roi),
      g.edge(cath_roi, cath_hit),
      g.edge(mem_src, mem_decon),
      g.edge(mem_decon, mem_hit),
      g.edge(pmt_src, pmt_decon),
      g.edge(pmt_decon, pmt_hit),
      g.edge(cath_hit, merge, 0, 0),
      g.edge(mem_hit, merge, 0, 1),
      g.edge(pmt_hit, merge, 0, 2),
      g.edge(merge, opflash_finder),
      g.edge(opflash_finder, fl_sink),
    ],
  );

  local app = {
    type: 'Pgrapher',
    data: { edges: g.edges(graph) },
  };

  local cmdline = {
    type: 'wire-cell',
    data: {
      plugins: [
        'WireCellRoot',
        'WireCellFlash',
        'WireCellGen',
        'WireCellSio',
        'WireCellAux',
        'WireCellPgraph',
      ],
      apps: ['Pgrapher'],
    },
  };

  [cmdline] + g.uses(graph) + [app]
