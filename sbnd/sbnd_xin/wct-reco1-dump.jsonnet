// Dump an SBND reco1 art/LArSoft ROOT file into the standalone interchange formats.
//
// The implementation now lives in the canonical, in-tree module
// cfg/pgrapher/experiment/sbnd/wct-reco1-dump.jsonnet (single source of truth).  This file is a
// thin re-export so the standalone sbnd_xin chain keeps running
// `wire-cell -c wct-reco1-dump.jsonnet`; top-level arguments (--tla-str / --tla-code) bind through
// the import unchanged, so the runner scripts need no edit.
// Promoted 2026-07-27 -- see sbnd_xin/docs/64_cfg-sync.md.
//
import 'pgrapher/experiment/sbnd/wct-reco1-dump.jsonnet'
