// PDHD WCT-native light reconstruction (stage 2) -- THIN RE-EXPORT.
//
// The implementation now lives in the canonical, in-tree job
//   cfg/pgrapher/experiment/pdhd/wct-light-reco.jsonnet
// which is the single source of truth for it, including every TLA default that
// carries a production value.  Promoted out of the working repo by doc
// sbnd_xin/120 on the owner's ask 2026-09-21 -- the move SBND made for its own
// standalone jobs in doc sbnd_xin/64 on 2026-07-27, applied to PDHD and PDVD.
//
// This file stays as a thin re-export so the runners keep working with no edit:
// run_pr_evt.sh / run_clus_evt.sh `cd` into pdhd/ and name this file bare, and
// top-level arguments (--tla-str / --tla-code) bind through the import
// unchanged.  The compiled config is byte-identical either way; the gate is doc
// 120 sec 3 -- every relocated file compiled bare AND at its runner's own TLA
// set, from this same path, before and after the move.
//
// EDIT THE IN-TREE FILE, NOT THIS ONE.

import 'pgrapher/experiment/pdhd/wct-light-reco.jsonnet'
