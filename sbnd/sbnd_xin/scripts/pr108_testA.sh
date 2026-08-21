#!/bin/bash
set -u; cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin
echo "=== $(date +%T) OFF gate 1 event (no env)"
PR_JOBS=4 ./run_pr_chain_batch.sh work-nuecc48-ql0819 work-pr108-off1-nuecc48 data 10550 > /home/xqian/tmp/pr108_off1.log 2>&1; echo rc=$?
python3 scripts/pr85_hash_gate.py work-pr107-off-nuecc48 work-pr108-off1-nuecc48 > /home/xqian/tmp/pr108_gate_off1.log 2>&1; echo gate_rc=$?; tail -1 /home/xqian/tmp/pr108_gate_off1.log
echo "=== $(date +%T) Test A (assoc check) 3 events"
WCT_DQDX_ASSOC_CHECK=1 SBND_FIT_EXCLUSION=true SBND_DQDX_FIT_KEEP_ALL_POINTS=true PR_JOBS=4 ./run_pr_chain_batch.sh work-nuecc48-ql0819 work-pr108-assoccheck-nuecc48 data 10550 46363 81597 > /home/xqian/tmp/pr108_assoc.log 2>&1; echo rc=$?
grep -h "dqdx_assoc_check" work-pr108-assoccheck-nuecc48/pr_evt*/wct_pr_evt*.log | awk '{print}' | sort | uniq -c | sort -rn | head -20
echo "=== $(date +%T) done"
