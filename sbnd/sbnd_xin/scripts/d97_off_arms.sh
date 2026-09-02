#!/bin/bash
# doc 97 -- a FRESH OFF arm over the whole validation sample, both knobs off.
#
# Why this exists.  The round started by gating work-<s>-grp0825 (2026-08-25,
# the on-disk stage-A half of production) against today's binary+cfg on a
# 367-event manifest: 367/367 byte-identical, three times over.  That manifest
# was too small.  A spot check of four mcp2k events found TWO -- 18255-x-74190
# and 53793 -- where today's OFF run does NOT reproduce grp0825, and two OFF
# runs today reproduce EACH OTHER exactly, so the difference is an epoch drift
# between 2026-08-25 and now, not run-to-run noise.
#
# grp0825 is therefore not a safe baseline for attributing a knob's effect, and
# neither is work-<s>-r3entry, which was built on it.  This arm is the baseline
# the A/B actually uses; the grp0825 comparison becomes a separate measurement
# of how far production's stored stage A has drifted.
#
# Usage: [D97_JOBS=8] [PR_JOBS=12] ./scripts/d97_off_arms.sh [ql|pr|all] [sample ...]
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
export D97_JOBS=${D97_JOBS:-8}
export PR_JOBS=${PR_JOBS:-12}
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES:-pr_display}
export LIBSNAP=${LIBSNAP:-$HOME/tmp/d97b-libsnap}
export LD_LIBRARY_PATH=$LIBSNAP:${LD_LIBRARY_PATH:-}
export PR_CFG_TREE=${PR_CFG_TREE:-$HOME/tmp/dbg25-cfgsnap}
LOGD=/home/xqian/tmp/d97; mkdir -p "$LOGD"
STAGE=${1:-all}; shift || true
SAMPLES=${*:-ncpi0 nuecc48 mcp1k mcp2k}
rc_all=0
for smp in $SAMPLES; do
    QL=$BASE/work-$smp-d97off2
    PR=$BASE/work-$smp-d97off2pr
    if [ "$STAGE" = all ] || [ "$STAGE" = ql ]; then
        if [ -e "$QL" ]; then echo "SKIP $QL exists (M13)"; else
            echo "=== $smp Q/L OFF  start $(date -Is)"
            QL_EXTRA="-save-pctree" ROOT=$QL \
                ./scripts/d97_ql_arm.sh "$smp" > "$LOGD/off2-$smp-ql.log" 2>&1
            rc=$?
            echo "=== $smp Q/L rc=$rc  ql_evt=$(find "$QL" -maxdepth 1 -type d -name 'ql_evt*' | wc -l)  $(date -Is)"
            [ "$rc" -ne 0 ] && { rc_all=1; continue; }
        fi
    fi
    if [ "$STAGE" = all ] || [ "$STAGE" = pr ]; then
        if [ -e "$PR" ]; then echo "SKIP $PR exists (M13)"; else
            echo "=== $smp PR  start $(date -Is)"
            ./run_pr_chain_batch.sh "$QL" "$PR" data > "$LOGD/off2-$smp-pr.log" 2>&1
            rc=$?
            echo "=== $smp PR rc=$rc  pr_evt=$(find "$PR" -maxdepth 1 -type d -name 'pr_evt*' | wc -l)  $(date -Is)"
            [ "$rc" -ne 0 ] && rc_all=1
        fi
    fi
done
echo "=== D97 OFF ARMS DONE rc_all=$rc_all $(date -Is)"
exit $rc_all
