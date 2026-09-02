#!/bin/bash
# doc 97 -- the sep_fv_point ON arm over the whole validation sample.
#
# Separate file, not a flag on d97_on_arms.sh: that script is the record of the
# track_recarve arm and was running when this one was written (bash re-reads a
# running script at a byte offset).
#
# sep_fv_point puts the per-APA separate() pass -- and only that pass -- at the
# PDHD/PDVD separation operating point: fv_inset_yz 15 cm (the new C++ knob),
# far_point_x_cut 14 cm, far_point_mid_dis 60 cm, dec1_guard_main_angle 45 deg.
# The binary is pinned to ~/tmp/d97b-libsnap, the build that carries
# inset_scope_fv(); the OFF baseline for stage A is still work-<s>-grp0825 and
# for stage B still work-<s>-r3entry.
#
# Usage: [D97_JOBS=12] [PR_JOBS=16] ./scripts/d97_fv_arms.sh [ql|pr|all] [sample ...]
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
export D97_JOBS=${D97_JOBS:-12}
export PR_JOBS=${PR_JOBS:-16}
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES:-pr_display}
export LIBSNAP=${LIBSNAP:-$HOME/tmp/d97b-libsnap}
export LD_LIBRARY_PATH=$LIBSNAP:${LD_LIBRARY_PATH:-}
export PR_CFG_TREE=${PR_CFG_TREE:-$HOME/tmp/dbg25-cfgsnap}
LOGD=/home/xqian/tmp/d97; mkdir -p "$LOGD"
STAGE=${1:-all}; shift || true
SAMPLES=${*:-ncpi0 nuecc48 mcp1k mcp2k}
rc_all=0

for smp in $SAMPLES; do
    QL=$BASE/work-$smp-d97fv
    PR=$BASE/work-$smp-d97fvpr2
    if [ "$STAGE" = all ] || [ "$STAGE" = ql ]; then
        if [ -e "$QL" ]; then echo "SKIP $QL exists (M13)"; else
            echo "=== $smp Q/L sep_fv_point  start $(date -Is)"
            QL_EXTRA="-save-pctree -sep-fv-point" ROOT=$QL \
                ./scripts/d97_ql_arm.sh "$smp" > "$LOGD/fv-$smp-ql.log" 2>&1
            rc=$?
            echo "=== $smp Q/L rc=$rc  ql_evt=$(find "$QL" -maxdepth 1 -type d -name 'ql_evt*' | wc -l)  $(date -Is)"
            [ "$rc" -ne 0 ] && { rc_all=1; continue; }
        fi
    fi
    if [ "$STAGE" = all ] || [ "$STAGE" = pr ]; then
        if [ -e "$PR" ]; then echo "SKIP $PR exists (M13)"; else
            echo "=== $smp PR  start $(date -Is)"
            ./run_pr_chain_batch.sh "$QL" "$PR" data > "$LOGD/fv-$smp-pr.log" 2>&1
            rc=$?
            echo "=== $smp PR rc=$rc  pr_evt=$(find "$PR" -maxdepth 1 -type d -name 'pr_evt*' | wc -l)  $(date -Is)"
            [ "$rc" -ne 0 ] && rc_all=1
        fi
    fi
done
echo "=== D97 FV ARMS DONE rc_all=$rc_all $(date -Is)"
exit $rc_all
