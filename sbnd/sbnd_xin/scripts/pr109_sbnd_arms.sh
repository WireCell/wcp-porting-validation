#!/bin/bash
# doc pr/109 -- SBND toolkit arms for the near-vertex 2-D charge residual study.
#
# Three arms off the same Q/L baseline, same binary (local/lib 2026-08-21 16:36):
#   on   fit_exclusion=true   (SBND production, wct-pr-perevt.jsonnet:196)
#   off  fit_exclusion=false  (the arm the owner's claim is about)
#   on2  fit_exclusion=true, a REPEAT of `on` -- measures the run-to-run band of
#        T_proj_data's charge_pred, which is last-writer-wins on cells owned by
#        more than one cluster (PrDisplayDump.cxx:1130-1150).  That band is the
#        systematic the ON-vs-OFF difference has to beat.
#
# The readout is the tree T_proj_data in pr_evt<ID>/tracking-pr.root
# (channel/time_slice/charge/charge_err/charge_pred per cluster) -- written by
# the default chain, no env dump and no code change needed.
#
# Events: the four pr/107-108 events plus the two most junction-rich nueCC48
# events (junction count from pr108_fit_point_compare.junctions on
# work-pr107-off-nuecc48: 256587 -> 23, 433451 -> 21).
set -eu
cd "$(dirname "$0")/.."
EVTS="10550 46363 81597 360535 256587 433451"
BASE=work-nuecc48-ql0819

run () {  # run <label> <fit_exclusion>
    local out="work-pr109-$1-nuecc48" excl="$2"
    echo "=== $(date +%T) arm=$1 fit_exclusion=$excl -> $out"
    SBND_FIT_EXCLUSION="$excl" PR_JOBS=6 \
        ./run_pr_chain_batch.sh "$BASE" "$out" data $EVTS
}

run on  true
run off false
run on2 true
echo "=== $(date +%T) done"
