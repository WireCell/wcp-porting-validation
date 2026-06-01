// SBND charge-light (Q/L) matching graph nodes.
//
// The implementation now lives in the canonical, in-tree module
// cfg/pgrapher/experiment/sbnd/qlmatching.jsonnet (single source of truth for the
// SBND matching graph and matching constants).  This file is a thin re-export so
// the standalone sbnd_xin chain keeps importing './qlmatching.jsonnet'.
//
// The constructor (params) and the opflash_source / flash_attach / matching
// methods are unchanged.
import 'pgrapher/experiment/sbnd/qlmatching.jsonnet'
