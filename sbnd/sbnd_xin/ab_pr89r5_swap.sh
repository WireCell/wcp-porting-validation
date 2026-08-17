#!/bin/bash
# doc pr/89 round 5 (sec 13) -- swap-guard live arm.  Candidate =
# production + SBND_DL_VTX_SWAP_GUARD=true, scored against the round-4
# work-*-pr89base baselines (same toolkit a681b3e1, same event lists).
set -u
cd "$(dirname "$0")"
export PR_JOBS=${PR_JOBS:-24}
export PR_EXTRA_STAGES=pr_display
export SBND_DL_VTX_HARVEST=true
export SBND_DL_VTX_SWAP_GUARD=true
for s in mcp2k:work-mcp2k-cb0816 nuecc48:work-nuecc48-cb0805 ncpi0:work-ncpi0-cb0805 mcp1k:work-mcp1k-cb0805; do
  n=${s%%:*}; r=${s##*:}
  evts=$(cat /home/xqian/tmp/pr89/ab-events-$n.txt)
  echo "=== $n swap -> work-$n-pr89swap ==="
  ./run_pr_chain_batch.sh "$r" "work-$n-pr89swap" data $evts
  echo "rc_swap_$n=$?"
done
echo "ALL DONE"
