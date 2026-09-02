#!/bin/bash
# doc 94 step 1 -- instrument the 5 STM events from the colleague's feedback
# sample (doc 93) with save_stm_fit, to find WHICH acceptance branch fired.
#
# No runner fork (doc 93 sec 4 assumed one was needed; it is not):
# run_pr_chain_batch.sh:1835 honors PR_EXTRA_TLA, a file of raw jsonnet TLA
# lines appended verbatim as --tla-code and deliberately placed as the LAST
# TLA block.  save_stm_fit is a top-level arg of wct-pr-perevt.jsonnet:956.
# 'stm_magnify' is absent from the default pipeline only because the knob
# defaults false (wct-pr-perevt.jsonnet:99-101), so PR_EXTRA_STAGES adds it.
#
# TRACE globally: the eval/detect_proton internals that say why a prong-less
# cluster accepted are SPDLOG_LOGGER_TRACE (verified compiled in -- the
# "eval_stm: KS value" literal is present in libWireCellClus.so).
#
# FRESH arm (M13): never write into work-stmfb8-pr.
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
export LD_LIBRARY_PATH=$HOME/tmp/prod0901b-libsnap:${LD_LIBRARY_PATH:-}
export PR_JOBS=${PR_JOBS:-4}
export PR_EXTRA_STAGES=pr_display,stm_magnify
export SBND_WCT_LOGLEVEL=trace
TLA=/home/xqian/tmp/doc94-stmfit.tla
printf 'save_stm_fit=true\n' > "$TLA"
export PR_EXTRA_TLA=$TLA
OUT=work-stmfb8-stmfit
if [ -e "$BASE/$OUT" ]; then
    echo "SKIP: $OUT already exists (M13: never write into an existing label)" >&2
    exit 1
fi
LOGD=/home/xqian/tmp/doc94; mkdir -p "$LOGD"
echo "start $(date -Is)  libsnap=$HOME/tmp/prod0901b-libsnap  jobs=$PR_JOBS"
./run_pr_chain_batch.sh work-stmfb8-ql "$OUT" sim > "$LOGD/instrument.log" 2>&1
rc=$?
n=$(find "$BASE/$OUT" -maxdepth 1 -type d -name 'pr_evt*' 2>/dev/null | wc -l)
echo "rc=$rc  pr_evt dirs=$n  end $(date -Is)"
exit $rc
