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

function(params)
    local base = canonical(params);
    base {
        matching(anode, dv, n, reality, semimodel_file, cathode_fiducial='', calib_dump='',
                 pmt_nl=true, cathode_diag='')::
            base.matching(anode, dv, n, reality, semimodel_file, cathode_fiducial, calib_dump,
                          pmt_nl=pmt_nl, extra=diag_on(cathode_diag)),

        matching_joint(anodes, dv, reality, semimodel_file, cathode_fiducial='', calib_dump='',
                       pmt_nl=true, cathode_diag='')::
            base.matching_joint(anodes, dv, reality, semimodel_file, cathode_fiducial, calib_dump,
                                pmt_nl=pmt_nl, extra=diag_on(cathode_diag)),
    }
