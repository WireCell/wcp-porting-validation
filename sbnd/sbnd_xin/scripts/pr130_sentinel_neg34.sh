#!/bin/bash
# doc pr/130 item 3 -- negative controls for the sentinels the earlier neg arms
# could not reach.  work-sent130neg{,2}-* were run BEFORE pr/129 landed and
# disabled only the doc-84 knobs, so pr/129's three entries and pr/124's 406125
# had never been shown capable of failing.
#
# Two arms, not one, so a FAIL is attributable to the right knob:
#   neg3  SBND_KINE_GF_IMPACT=0        -> pr/129 pointing test off (393505,171572,94392)
#   neg4  SBND_SHOWER_PASS4_PRUNE_GAP2=0 -> pr/124 gap-band tier-2 prune off (406125)
# pr/123's own entries on 393505/171572 use a different knob and must still PASS
# in neg3 -- that is the specificity check.
set -u
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
export PR_EXTRA_STAGES=pr_display PR_JOBS=${PR_JOBS:-12}
SBND_KINE_GF_IMPACT=0 ./run_pr_chain_batch.sh work-mcp2k-grp0825 work-sent130neg3-mcp2k data \
    393505 171572 94392 > /home/xqian/tmp/pr130_neg3.log 2>&1
echo "neg3 rc=$?"
SBND_SHOWER_PASS4_PRUNE_GAP2=0 ./run_pr_chain_batch.sh work-mcp2k-grp0825 work-sent130neg4-mcp2k data \
    406125 > /home/xqian/tmp/pr130_neg4.log 2>&1
echo "neg4 rc=$?"
