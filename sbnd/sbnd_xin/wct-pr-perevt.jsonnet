// Per-event SBND pattern-recognition (PR) job.
//
// The implementation now lives in the canonical, in-tree module
// cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet (single source of truth for
// the SBND PR graph and its production operating point).  This file is a thin
// re-export so the standalone sbnd_xin chain keeps running
// `wire-cell -c wct-pr-perevt.jsonnet`; top-level arguments (--tla-str /
// --tla-code) bind through the import unchanged, so run_pr_evt.sh and
// run_nusel_evt.sh need no edit.
//
// Two things changed with the promotion (sbnd_xin/docs/64_cfg-sync.md):
//   - the dQ/dx + range tables are imported as
//     'pgrapher/experiment/sbnd/particle_dataset.jsonnet' (WIRECELL_PATH) rather
//     than the relative '../particle_dataset.jsonnet', so the job no longer has
//     to be compiled from the real (non-symlinked) working directory;
//   - the TGM/FC, beam-window and pipeline_names TLA defaults are now the
//     production operating point instead of the pre-adoption values.  Every
//     runner passes these explicitly, so the compiled config is byte-identical
//     either way (gate: docs/64 Repro block).
import 'pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet'
