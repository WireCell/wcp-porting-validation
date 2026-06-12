// SBND cathode-plane (CPA) structure-exclusion fiducial volume.
//
// The implementation lives in the canonical, in-tree module
// cfg/pgrapher/experiment/sbnd/cathode_fiducial.jsonnet (single source of truth).
// This file is a thin re-export so the standalone sbnd_xin chain keeps importing
// './cathode_fiducial.jsonnet'.
//
// Usage:  local cathode_fv = (import 'cathode_fiducial.jsonnet')(cx=.., cy=.., cz=..);
//         ... qlm.matching(..., cathode_fiducial=cathode_fv.tn) ...
//         [cmdline] + g.uses(graph) + cathode_fv.configs + [app]
import 'pgrapher/experiment/sbnd/cathode_fiducial.jsonnet'
