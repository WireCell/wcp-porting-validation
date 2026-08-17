#!/bin/bash
# doc pr/89 phase 5 -- the live A/B.  Baseline (production knobs, toolkit
# a681b3e1 whose knob-off path is gate-proven byte-identical to c3741088)
# vs candidate = production + SBND_DL_VTX_TOPO_WEIGHT=3.0 (C2, owner-approved
# 2026-08-17).  Event set = the full six-tag label universe (1014 events:
# mcp2k 541, mcp1k 407, nuecc48 47, ncpi0 19 -- delta labels folded into
# their source samples), lists in /home/xqian/tmp/pr89/ab-events-<sample>.txt.
# Both arms run with identical envs apart from the topo weight.
set -u
cd "$(dirname "$0")"
export PR_JOBS=${PR_JOBS:-24}
export PR_EXTRA_STAGES=pr_display
export SBND_DL_VTX_HARVEST=true
for s in mcp2k:work-mcp2k-cb0816 nuecc48:work-nuecc48-cb0805 ncpi0:work-ncpi0-cb0805 mcp1k:work-mcp1k-cb0805; do
  n=${s%%:*}; r=${s##*:}
  evts=$(cat /home/xqian/tmp/pr89/ab-events-$n.txt)
  echo "=== $n baseline -> work-$n-pr89base ($(wc -l < /home/xqian/tmp/pr89/ab-events-$n.txt) evts) ==="
  ./run_pr_chain_batch.sh "$r" "work-$n-pr89base" data $evts
  echo "rc_base_$n=$?"
  echo "=== $n topo -> work-$n-pr89topo ==="
  SBND_DL_VTX_TOPO_WEIGHT=3.0 ./run_pr_chain_batch.sh "$r" "work-$n-pr89topo" data $evts
  echo "rc_topo_$n=$?"
done
echo "ALL DONE"
