#!/bin/bash
# doc 89 Phase 1 -- re-run SBND production at the PINNED operating point
# (ref/prod-2026-09-01b, save_in_scope ON) over all 3067 events.
#
# Why a fresh arm at all: work-*-prod0901 finished 2026-09-01 05:00; the
# save_in_scope flip (toolkit d52d818c) landed 13:36, so prod0901's
# tracking-pr.root has no T_cluster tree.  This arm IS the pinned point.
#
# BOTH the binary and the cfg tree are pinned, because a peer session is
# actively editing the shared toolkit tree (libWireCellGen.so rebuilt 14:23,
# qlport/uboone-mabc.jsonnet toggled between two consecutive gate runs).
#   ~/tmp/prod0901b-libsnap  == doc 87 lib-flip for Clus+Root (verified md5)
#   ~/tmp/prod0901b-cfgsnap  == prod_cfg_gate.py PASS 21/21
#
# FULL output on purpose (doc 87 sec 5.3): no suppression knob is set and
# PR_EXTRA_STAGES=pr_display keeps the calib dump.  A validation arm must
# never adopt the production minimal line.
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
export LD_LIBRARY_PATH=$HOME/tmp/prod0901b-libsnap:${LD_LIBRARY_PATH:-}
export PR_CFG_TREE=$HOME/tmp/prod0901b-cfgsnap
export PR_JOBS=${PR_JOBS:-32}
export PR_EXTRA_STAGES=pr_display
LOGD=$BASE/logs; mkdir -p "$LOGD"
echo "BASE=$BASE  PR_JOBS=$PR_JOBS  $(date -Is)"
echo "libsnap=$HOME/tmp/prod0901b-libsnap  cfgsnap=$PR_CFG_TREE"
rc_all=0
# smallest first, so a defect shows up in minutes rather than an hour
for s in ncpi0 nuecc48 mcp1k mcp2k; do
    out=work-$s-prod0901b
    if [ -e "$BASE/$out" ]; then
        echo "SKIP $out -- already exists (M13: never write into an existing label)"
        continue
    fi
    echo "=== $s -> $out  start $(date -Is)"
    ./run_pr_chain_batch.sh "work-$s-grp0825" "$out" data > "$LOGD/$out.log" 2>&1
    rc=$?
    n=$(find "$BASE/$out" -maxdepth 1 -type d -name 'pr_evt*' 2>/dev/null | wc -l)
    echo "=== $s  rc=$rc  pr_evt dirs=$n  end $(date -Is)"
    [ "$rc" -ne 0 ] && rc_all=1
done
echo "ALL DONE rc_all=$rc_all $(date -Is)"
exit $rc_all
