// SBND per-APA and all-APA clustering.
//
// The implementation now lives in the canonical, in-tree module
// cfg/pgrapher/experiment/sbnd/clus.jsonnet (single source of truth for the
// SBND clustering graph and fiducial volume).  This file is a thin re-export so
// the standalone sbnd_xin chain keeps importing './clus.jsonnet'.
//
// The constructor (output_dir, runNo, subRunNo, eventNo) and the
// per_face / per_apa / all_apa / detector_volumes methods are unchanged.
import 'pgrapher/experiment/sbnd/clus.jsonnet'
