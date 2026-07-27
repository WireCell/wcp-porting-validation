// SBND charge-light (Q/L) matching graph nodes.
//
// The implementation lives in the canonical, in-tree module
// cfg/pgrapher/experiment/sbnd/qlmatching.jsonnet (single source of truth for
// the SBND matching graph and its production operating point).  This file is now
// a thin re-export: the four overlays this wrapper used to add on top of the
// canonical module were promoted into it on 2026-07-27
// (sbnd_xin/docs/64_cfg-sync.md), so there is nothing left to wrap.
//
//   cathode_diag   -> canonical arg (default '' = off), same key-suppression.
//   main_flag      -> canonical arg, now DEFAULT TRUE (flag_matched_mains);
//                     production always passed main_flag=true.
//   beam_pref /    -> canonical args, tri-state: null = inherit the production
//   beam_pref_*       operating point that match_data already sets, a value
//                     overrides it (that is what the scan overrides need).
//   auto_mask      -> canonical arg, same tri-state.  NOTE the wrapper's
//                     automask_on() was already dead: match_data sets
//                     auto_mask: true unconditionally, so passing false here
//                     never disabled it.  The canonical arg CAN disable it
//                     (auto_mask=false emits the key); the per-event job maps its
//                     own auto_mask=false TLA to null to keep the old,
//                     re-assert-only meaning of that TLA.
//
// The `lm` arg also defaults TRUE now (SBND production runs with -lm).
import 'pgrapher/experiment/sbnd/qlmatching.jsonnet'
