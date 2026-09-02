#!/bin/bash
# doc 97 -- the sep_track_recarve ON arm over the whole validation sample:
# 3067 SBND data events (mcp1k 1000 + mcp2k 2000) + nueCC48 (48) + NCpi0 (19).
#
# Stage A (Q/L) is re-run because the knob lives in ClusteringSeparate, inside
# the Q/L job; stage B (the PR tagger tail) is then rebuilt on top of it at the
# production operating point ref/prod-2026-09-03, which is the CONFIG DEFAULT
# since the 2026-09-02 owner flip -- so no PR_EXTRA_TLA is needed and the OFF
# baseline is the existing work-<s>-r3entry.
#
# The OFF baseline for stage A is work-<s>-grp0825, proven still reproducible
# by today's binary + cfg: doc 97 sec 2, 367 events / 1468 products PASS.
#
# Usage: [D97_JOBS=12] [PR_JOBS=16] ./scripts/d97_on_arms.sh [ql|pr|all] [sample ...]
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
export D97_JOBS=${D97_JOBS:-12}
export PR_JOBS=${PR_JOBS:-16}
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES:-pr_display}
export LD_LIBRARY_PATH=$HOME/tmp/doc94r3b-libsnap:${LD_LIBRARY_PATH:-}
export PR_CFG_TREE=${PR_CFG_TREE:-$HOME/tmp/dbg25-cfgsnap}
LOGD=/home/xqian/tmp/d97; mkdir -p "$LOGD"
STAGE=${1:-all}; shift || true
SAMPLES=${*:-ncpi0 nuecc48 mcp1k mcp2k}
rc_all=0

for smp in $SAMPLES; do
    QL=$BASE/work-$smp-d97on
    PR=$BASE/work-$smp-d97onpr
    if [ "$STAGE" = all ] || [ "$STAGE" = ql ]; then
        if [ -e "$QL" ]; then
            echo "SKIP $QL exists (M13)"
        else
            echo "=== $smp Q/L ON  start $(date -Is)"
            QL_EXTRA="-save-pctree -sep-recarve" ROOT=$QL \
                ./scripts/d97_ql_arm.sh "$smp" > "$LOGD/on-$smp-ql.log" 2>&1
            rc=$?
            echo "=== $smp Q/L rc=$rc  ql_evt=$(find "$QL" -maxdepth 1 -type d -name 'ql_evt*' | wc -l)  fired=$(grep -h . "$QL"/.status/* 2>/dev/null | grep -vc ' fired=0$')  $(date -Is)"
            [ "$rc" -ne 0 ] && { rc_all=1; continue; }
        fi
    fi
    if [ "$STAGE" = all ] || [ "$STAGE" = pr ]; then
        if [ -e "$PR" ]; then
            echo "SKIP $PR exists (M13)"
        else
            echo "=== $smp PR  start $(date -Is)"
            ./run_pr_chain_batch.sh "$QL" "$PR" data > "$LOGD/on-$smp-pr.log" 2>&1
            rc=$?
            echo "=== $smp PR rc=$rc  pr_evt=$(find "$PR" -maxdepth 1 -type d -name 'pr_evt*' | wc -l)  $(date -Is)"
            [ "$rc" -ne 0 ] && rc_all=1
        fi
    fi
done
echo "=== D97 ON ARMS DONE rc_all=$rc_all $(date -Is)"
exit $rc_all
