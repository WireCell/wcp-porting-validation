// SBND per-APA and all-APA blob clustering -- standalone (no LArSoft).
//
// The implementation now lives in the canonical, in-tree module
// cfg/pgrapher/experiment/sbnd/wct-clustering.jsonnet (single source of truth).  This file is a
// thin re-export so the standalone sbnd_xin chain keeps running
// `wire-cell -c wct-clustering.jsonnet`; top-level arguments (--tla-str / --tla-code) bind through
// the import unchanged, so the runner scripts need no edit.
// Promoted 2026-07-27 -- see sbnd_xin/docs/64_cfg-sync.md.
//
import 'pgrapher/experiment/sbnd/wct-clustering.jsonnet'
