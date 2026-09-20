#!/bin/bash
# doc 115: an EXTRA stage-B worker for the nuecc sample over a RANGE of per-file sub-roots,
# run alongside the primary scripts/d115/stageB.sh so the arm finishes in reasonable time.
#
# Why it exists.  nuecc reco1 files hold ~9 events each, and stageB.sh walks the sub-roots one
# at a time, so the effective parallelism is ~9 against a PR stage that costs 18.6 s/event on
# this sample -- over four hours for 2001 events.  Running disjoint index ranges concurrently
# fixes that: each sub-root is its own out_root with no shared state, and a sub-root either
# worker has finished is skipped by the other's npr==nql guard.
#
# Ranges must be DISJOINT from what the primary is currently sweeping and from each other --
# two workers inside the same sub-root at the same time would write the same pr_evt dirs.  The
# doc-115 arm used START=224 STOP=113, then START=112 STOP=80, then START=126 STOP=113, while
# the primary climbed from f000; the boundaries were checked against each worker's live
# position before launch.
#
# Usage: START=<high> STOP=<low> ./stageB_range.sh      (descending, 2 sub-roots at a time)
set -u
SX=/home/xqian/toolkit-dev/toolkit/sbnd_xin
cd "$SX" || exit 1
export LD_LIBRARY_PATH=/home/xqian/tmp/d115-libsnap:${LD_LIBRARY_PATH:-}
export PR_EXTRA_STAGES=pr_display
STOP=${STOP:-113}
one() {
    i=$1
    qa=work-r3nue-d115/f$i; qb=work-r3nue-d115pr/f$i
    [ -d "$qa" ] || return 0
    nql=$(ls -d "$qa"/ql_evt* 2>/dev/null | wc -l)
    npr=$(ls -d "$qb"/pr_evt* 2>/dev/null | wc -l)
    [ "$nql" -gt 0 ] && [ "$npr" -eq "$nql" ] && { echo "[f$i] skip ($npr/$nql)"; return 0; }
    PR_JOBS=8 ./run_pr_chain_batch.sh "$qa" "$qb" sim > /home/xqian/tmp/d115-nue-tail-f$i.log 2>&1
    echo "[f$i] rc=$? $(ls -d "$qb"/pr_evt* 2>/dev/null | wc -l)/$nql"
}
export -f one
export LD_LIBRARY_PATH PR_EXTRA_STAGES
seq ${START:-224} -1 "$STOP" | xargs -P 2 -I{} bash -c 'printf -v n "%03d" {}; one "$n"'
echo "NUE TAIL DONE"
