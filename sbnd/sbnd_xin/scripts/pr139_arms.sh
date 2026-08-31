#!/bin/bash
# doc pr/139 phase 1 -- arm launcher.  Fork of the pr/138 round recipe; the
# pr/138 scripts stay byte-untouched (M10).
#
#   ./scripts/pr139_arms.sh <armtag>            # env already exported by caller
#
# The binary is PINNED (/home/xqian/tmp/pin-pr139b) so a peer's wcbuild cannot
# swap local/lib mid-campaign.  Every arm gets a FRESH name (M13); the runner
# itself refuses a non-empty out_root.
set -u
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
TAG=$1
export LD_LIBRARY_PATH=/home/xqian/tmp/pin-pr139b:${LD_LIBRARY_PATH:-}
for s in mcp1k mcp2k ncpi0 nuecc48; do
  EVTS=$(tr '\n' ' ' < /home/xqian/tmp/pr139-manifest-$s.lst)
  PR_JOBS=32 PR_EXTRA_STAGES=pr_display \
    ./run_pr_chain_batch.sh work-$s-grp0825 ${PR139_ARM:-work-pr139r1}-$TAG-$s data $EVTS \
      > /home/xqian/tmp/pr139-arm-${PR139_ARM:-work-pr139r1}-$TAG-$s.log 2>&1
  echo "arm $TAG $s rc=$?"
done
echo "ARM $TAG DONE"
