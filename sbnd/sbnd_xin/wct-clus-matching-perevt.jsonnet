// Per-event SBND clustering + charge-light (Q/L) matching -- standalone.
//
// The implementation now lives in the canonical, in-tree module
// cfg/pgrapher/experiment/sbnd/wct-clus-matching-perevt.jsonnet (single source of truth).  This file is a
// thin re-export so the standalone sbnd_xin chain keeps running
// `wire-cell -c wct-clus-matching-perevt.jsonnet`; top-level arguments (--tla-str / --tla-code) bind through
// the import unchanged, so the runner scripts need no edit.
// Promoted 2026-07-27 -- see sbnd_xin/docs/64_cfg-sync.md.
// The main_flag and lm TLA defaults are now the production operating point
// (both ON); run_ql_evt.sh passes them explicitly, so the compiled config is
// byte-identical either way.
import 'pgrapher/experiment/sbnd/wct-clus-matching-perevt.jsonnet'
