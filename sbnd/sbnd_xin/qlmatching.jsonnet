// SBND charge-light (Q/L) matching graph nodes for the standalone sbnd_xin chain.
//
// Thin wrapper over the canonical, in-tree module
// cfg/pgrapher/experiment/sbnd/qlmatching.jsonnet (single source of truth for the SBND
// matching graph and the production config). The canonical module now owns the per-PMT
// predicted-PE non-linearity correction and applies it by default (pmt_nl=true); this
// wrapper just forwards pmt_nl and adds the standalone-only cathode_diag overlay.
// Pass pmt_nl=false to disable the correction. opflash_source / flash_attach inherited.
local canonical = import 'pgrapher/experiment/sbnd/qlmatching.jsonnet';

// cathode_diag: path ('' => off) for the cathode-crossing TPC0/TPC1 offset
// diagnostic (QLMatching logs the three-vector decomposition per cross-TPC
// cathode-crossing pair). Default '' keeps production byte-identical.
local diag_on(cathode_diag) = (if cathode_diag != '' then { cathode_diag: cathode_diag } else {});

// auto_mask: per-event dynamic dead-PMT mask (QLMatching auto_mask knob; default off =>
// production byte-identical). true enables the C++ defaults (pe_low 5, K 4, pe_bright 50,
// min_contrast 1, min_flash 3), validated to mask a run-dead PMT (e.g. lan-reco2 ch69)
// with zero false positives. See match/docs/qlmatching-code.md.
local automask_on(auto_mask) = (if auto_mask then { auto_mask: true } else {});

// beam_pref: beam-window flash preference (QLMatching beam_pref knob; C++ default off =>
// production byte-identical, keys suppressed here when false). true enables the
// beam-window mechanisms on flashes inside the C++ default window (0.2, 2.2) us — the
// SBND BNB window after the frame_apply_at_caf correction:
//   1. cull_inconsistent exemption (a beam-window bundle survives the rival-consistent
//      drop and competes in the LASSO), and
//   2. per-column L1 down-weight x<w> (validated operating point 0.5: the 0.2 scan
//      point over-collects -- see the doc's population + validation tables), and
//   3. empty-flash rescue steal guard x<r> (a non-beam flash must beat a beam-window
//      match by 1/r on the light metric to re-steal its cluster; 0.2 => 5x).
// The preference only applies to bundles that could plausibly BE the beam match
// (validation round 2, doc 22): ks_dis <= max_ks AND pred >= min_pred_frac x flash
// PE. Without the gate the down-weighted beam flash sweeps up junk bundles
// (pred 8-300 PE at ks 0.4-0.8) and occasionally steals a hand-scanned true match;
// genuine beam matches in every validated case have ks <= 0.27, pred frac >= 0.17.
// w/r are function args (TLA-threaded by wct-clus-matching-perevt.jsonnet /
// BEAMPREF_WEIGHT env in run_ql_evt.sh) so the operating point can be scanned
// without editing this file. Case study + validation:
// sbnd_xin/docs/22_ql-beam-flash-preference.md (reco1 evts 246579 / 116962).
local beampref_on(beam_pref, w=0.5, r=0.2) = (if beam_pref then {
    beam_pref: true,
    beam_pref_lasso_weight: w,
    beam_pref_rescue_scale: r,
    beam_pref_max_ks: 0.3,          // C++ default 1e9 (ungated round-1 behavior)
    beam_pref_min_pred_frac: 0.02,  // C++ default 0.0 (ungated round-1 behavior)
} else {});

function(params)
    local base = canonical(params);
    base {
        matching(anode, dv, n, reality, semimodel_file, cathode_fiducial='', calib_dump='',
                 pmt_nl=true, cathode_diag='', auto_mask=false, beam_pref=false,
                 beam_pref_weight=0.5, beam_pref_rescue=0.2)::
            base.matching(anode, dv, n, reality, semimodel_file, cathode_fiducial, calib_dump,
                          pmt_nl=pmt_nl, extra=diag_on(cathode_diag) + automask_on(auto_mask)
                                               + beampref_on(beam_pref, beam_pref_weight, beam_pref_rescue)),

        matching_joint(anodes, dv, reality, semimodel_file, cathode_fiducial='', calib_dump='',
                       pmt_nl=true, cathode_diag='', auto_mask=false, beam_pref=false,
                       beam_pref_weight=0.5, beam_pref_rescue=0.2)::
            base.matching_joint(anodes, dv, reality, semimodel_file, cathode_fiducial, calib_dump,
                                pmt_nl=pmt_nl, extra=diag_on(cathode_diag) + automask_on(auto_mask)
                                                     + beampref_on(beam_pref, beam_pref_weight, beam_pref_rescue)),
    }
