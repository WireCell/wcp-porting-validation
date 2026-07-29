// SBND 3D imaging on SP frames -- standalone (no LArSoft).
//
// The implementation now lives in the canonical, in-tree module
// cfg/pgrapher/experiment/sbnd/wct-img-all.jsonnet (single source of truth).  This file is a
// thin re-export so the standalone sbnd_xin chain keeps running
// `wire-cell -c wct-img-all.jsonnet`; top-level arguments (--tla-str / --tla-code) bind through
// the import unchanged, so the runner scripts need no edit.
// Promoted 2026-07-27 -- see sbnd_xin/docs/64_cfg-sync.md.
//
import 'pgrapher/experiment/sbnd/wct-img-all.jsonnet'
