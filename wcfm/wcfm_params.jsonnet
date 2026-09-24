// wcfm_params.jsonnet -- the one place for the DUNE FD-HD 1x2x6 "workspace" campaign
// constants (wcfm/docs/01 sec 6 W1, wcfm/docs/02).
//
// Geometry (verified against dune10kt-1x2x6-wires-larsoft-v1.json.bz2, doc 02 sec 0):
// 12 APAs, all at x = 0, laid out 2 rows in y (even ident y < 0, odd ident y > 0) x 6
// columns in z (2306.4 mm long, 2323.9 mm pitch); BOTH faces live -- face 0 (+x wires,
// W channels 2080-2559) drifts to the cathode at +3629 mm, face 1 (-x, W 1600-2079) to
// -3629 mm; U (0-799) and V (800-1599) channels are wrapped over both faces.
// The in-tree params.jsonnet had the APAs at x = +-3.63 m until 2026-09-24 (fixed);
// simparams.jsonnet always had centerline 0.  Everything here builds on simparams.
local wc = import 'wirecell.jsonnet';
local base_maker = import 'pgrapher/experiment/dune10kt-1x2x6/simparams.jsonnet';

{
    // The params object every wcfm job (sim, NF, SP, imaging, clustering) is built from.
    params: base_maker({}),

    // Readout / drift.  lar.drift_speed is the common base's 1.6 mm/us; the field file
    // (dune-garfield-1d565) says 1.565.  Kept at 1.6 everywhere so the simulation
    // (Drifter), BlobDepoFill and the clustering x agree with each other (doc 02 sec 7).
    drift_speed: $.params.lar.drift_speed,
    tick: $.params.daq.tick,          // 0.5 us
    nticks: $.params.daq.nticks,      // 6000
    tick_span: 4,                     // MaskSlices span == the FM's 4-tick pixel (doc 01 sec 2.3)

    // BlobDepoFill time offset: depo time + time_offset is matched to the frame-relative
    // slice start.  MEASURED (doc 02 sec 4.1, event 2 scan): captured true charge peaks at
    // 64-65 us (0.989/0.988; 62.5 us 0.952, 67 us 0.886) = the 10 cm response-plane transit
    // at 1.6 mm/us (62.5 us) plus ~2 us of field-response peak delay.
    depofill_time_offset: 64.5 * wc.us,
    depofill_nsigma: 3.0,

    // Anode layout helpers (mm).
    anode_row(ident):: ident % 2,                 // 0 = y < 0, 1 = y > 0
    anode_col(ident):: std.floor(ident / 2),      // z column 0..5
    z_pitch: 2323.9 * wc.mm,
    z_len: 2306.4 * wc.mm,

    // Per-face drift-volume x extents for the clustering DetectorVolumes metadata (FV_x*):
    // from the W wire plane (|x| = 30.0155 mm) to the cathode face (apa_cpa - cpa_thick/2).
    fv_x: {
        wire: 30.0155 * wc.mm,
        cathode: 3.63075 * wc.m - 0.5 * 3.175 * wc.mm,   // 3629.1625 mm
    },
    // Overall active box (wires file extents; the 15 cm insets are the PDHD convention).
    fv_overall: {
        FV_xmin: -$.fv_x.cathode, FV_xmax: $.fv_x.cathode,
        FV_ymin: -6001.2 * wc.mm + 15 * wc.cm, FV_ymax: 6001.2 * wc.mm - 15 * wc.cm,
        FV_zmin: 0 * wc.mm + 15 * wc.cm, FV_zmax: 13925.9 * wc.mm - 15 * wc.cm,
    },

    // Drift groups for clustering: all run anodes seen through face 0 (+x volume) and all
    // through face 1 (-x volume).  PDHD's ident%2 grouping is WRONG here (ident parity is
    // the y row, not the drift side).  `anodes` = the tools.anodes objects of the run.
    face_groups(anodes):: [
        { name: 'groupf0', face: 0, anodes: anodes },
        { name: 'groupf1', face: 1, anodes: anodes },
    ],

    // Bee display detector tag (no FD-HD entry exists in the toolkit's Bee writer; the
    // PDHD tag only names the geometry Bee draws).
    bee_detector: 'protodunehd',
}
