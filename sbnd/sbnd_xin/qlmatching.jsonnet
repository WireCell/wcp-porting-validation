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

function(params)
    local base = canonical(params);
    base {
        matching(anode, dv, n, reality, semimodel_file, cathode_fiducial='', calib_dump='',
                 pmt_nl=true, cathode_diag='', auto_mask=false)::
            base.matching(anode, dv, n, reality, semimodel_file, cathode_fiducial, calib_dump,
                          pmt_nl=pmt_nl, extra=diag_on(cathode_diag) + automask_on(auto_mask)),

        matching_joint(anodes, dv, reality, semimodel_file, cathode_fiducial='', calib_dump='',
                       pmt_nl=true, cathode_diag='', auto_mask=false)::
            base.matching_joint(anodes, dv, reality, semimodel_file, cathode_fiducial, calib_dump,
                                pmt_nl=pmt_nl, extra=diag_on(cathode_diag) + automask_on(auto_mask)),
    }
