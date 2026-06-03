// SBND charge-light (Q/L) matching graph nodes for the standalone sbnd_xin chain.
//
// Thin wrapper over the canonical, in-tree module
// cfg/pgrapher/experiment/sbnd/qlmatching.jsonnet (single source of truth for the SBND
// matching graph and the production config, which keeps the PMT non-linearity OFF).
//
// This wrapper optionally enables the per-PMT predicted-PE non-linearity correction
// (study-grade; the realized scintillation NPE_true->NPE_reco curve, parameterized in
// sbnd_xin/pmt_nonlinearity_curve.py, params in cfg/.../pmt_nonlinearity_params.jsonnet;
// see match/docs/sbnd-opdetsim-chain.md). Callers pass pmt_nl=true to turn it on; default
// false => byte-identical to canonical. opflash_source / flash_attach are inherited.
local canonical = import 'pgrapher/experiment/sbnd/qlmatching.jsonnet';
local nlp = import 'pgrapher/experiment/sbnd/pmt_nonlinearity_params.jsonnet';

local nl_on = {
    pmt_nonlinearity: true,
    pmt_nl_knee: nlp.pmt_nl_knee,
    pmt_nl_beta: nlp.pmt_nl_beta,
    pmt_nl_gamma: nlp.pmt_nl_gamma,
};

function(params)
    local base = canonical(params);
    base {
        matching(anode, dv, n, reality, semimodel_file, cathode_fiducial='', calib_dump='',
                 pmt_nl=false)::
            base.matching(anode, dv, n, reality, semimodel_file, cathode_fiducial, calib_dump,
                          extra=(if pmt_nl then nl_on else {})),

        matching_joint(anodes, dv, reality, semimodel_file, cathode_fiducial='', calib_dump='',
                       pmt_nl=false)::
            base.matching_joint(anodes, dv, reality, semimodel_file, cathode_fiducial, calib_dump,
                                extra=(if pmt_nl then nl_on else {})),
    }
