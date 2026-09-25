#!/bin/bash
# doc sbnd_xin/123: an EXTRA stage-B worker over a DESCENDING range of per-file sub-roots, run
# alongside scripts/d123/stageB.sh (which walks the sub-roots upward, ~9 events each, so its
# effective parallelism is ~9 whatever PR_JOBS says -- the doc 115 stageB_range.sh reason).
# The range must stay disjoint from the primary's sweep: a sub-root already started by anyone
# (its out dir exists) is skipped here, and the caller picks STOP above where the primary will be.
#
# Usage: [JOBS=n] scripts/d123/stageB_range.sh <stageA_root> <data|sim> <START> <STOP>
set -u
cd -P "$(dirname "$0")/../.." || exit 1
SX=$PWD
A=${1:?usage: stageB_range.sh <stageA_root> <data|sim> <START> <STOP>}; REALITY=${2:?}; START=${3:?}; STOP=${4:?}
case "$A" in /*) ;; *) A=$SX/$A;; esac
B=${A}pr
P=${PIN:-$HOME/tmp/d123-libpin}
export LD_LIBRARY_PATH=$P:$P/reco1/lib:${LD_LIBRARY_PATH:-}
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES:-pr_display}
J=${JOBS:-7}
LOGD=$HOME/tmp/d123-stageB-$(basename "$A")-range; mkdir -p "$LOGD"
echo "=== d123 stage B range worker: $A -> $B f$START .. f$STOP (descending) PR_JOBS=$J"
for ((i=START; i>=STOP; i--)); do
    sub=$(printf 'f%03d' "$i"); qa=$A/$sub; qb=$B/$sub
    [ -d "$qa" ] || continue
    [ -d "$qb" ] && { echo "[$sub] already started by another worker -- skipped"; continue; }
    nql=$(ls -d "$qa"/ql_evt* 2>/dev/null | wc -l); [ "$nql" -gt 0 ] || continue
    PR_JOBS=$J "$SX/run_pr_chain_batch.sh" "$qa" "$qb" "$REALITY" > "$LOGD/$sub.log" 2>&1
    echo "[$sub] rc=$? pr_evt=$(ls -d "$qb"/pr_evt* 2>/dev/null | wc -l)/$nql"
done
echo "=== range worker done"
