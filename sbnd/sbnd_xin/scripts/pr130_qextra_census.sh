#!/bin/bash
# pr/130 item 1b -- absorb census on the 10 events that carry the affirmative
# q_extra pool (fork of pr130_arms.sh; that file stays byte-untouched, M10).
#
# Purpose: the label store's `absorbed_by` names the absorber for each of the
# 22 scanner-condemned segments, but those marks were made against the
# scan-time arms (work-*-prod0825 era).  This re-runs the same 10 events at
# TODAY's production point with the byte-neutral stderr census on, so the
# attribution can be confirmed against a live run before any knob is designed.
#
# Byte-neutral: WCT_SHOWER_*_DEBUG only add stderr lines.
set -u
TAG=${1:-qx1}
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
export WCT_SHOWER_ABSORB_DEBUG=1 WCT_SHOWER_CONTENT_DEBUG=1 WCT_SHOWER_PID_DEBUG=1 WCT_SHOWER_TOPO_DEBUG=1
export PR_EXTRA_STAGES=pr_display PR_JOBS=${PR_JOBS:-24}
declare -A EV
EV[mcp1k]="175896 489327 286655 278420 400504"
EV[mcp2k]="100222 499577 69232 350354 72786"
for s in mcp1k mcp2k; do
  ./run_pr_chain_batch.sh work-$s-grp0825 work-pr130-$TAG-$s data ${EV[$s]} \
      > /home/xqian/tmp/pr130_${TAG}_$s.log 2>&1
  echo "arm $TAG-$s rc=$?"
done
