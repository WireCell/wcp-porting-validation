#!/bin/bash
# doc pr/139 items 1/3/4 -- arm launcher.  FORK of pr139_arms.sh; that script
# stays byte-untouched (M10).  Two things differ and both matter:
#   * arm prefix PR140_ARM (sec 3ter pre-named these work-pr140r1-*; there is
#     no doc pr/140 -- the tracker is 139_pi0-after-the-splitter.md sec 8).
#   * the production config now has shower_split_em_start ON (owner flip
#     2026-08-31), so these arms are single-knob deltas on the FLIPPED config
#     and their baseline is work-pr139r3-flipchk-*, NOT work-pr139r1-off-*.
#
#   ./scripts/pr140_arms.sh <armtag>            # env already exported by caller
#
# The binary is PINNED (/home/xqian/tmp/pin-pr139b, md5-verified equal to
# local/lib on 2026-08-31) so a peer's wcbuild cannot swap it mid-campaign.
set -u
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
TAG=$1
export LD_LIBRARY_PATH=${PR140_PIN:-/home/xqian/tmp/pin-pr139b}:${LD_LIBRARY_PATH:-}
for s in mcp1k mcp2k ncpi0 nuecc48; do
  EVTS=$(tr '\n' ' ' < /home/xqian/tmp/pr139-manifest-$s.lst)
  PR_JOBS=32 PR_EXTRA_STAGES=pr_display \
    ./run_pr_chain_batch.sh work-$s-grp0825 ${PR140_ARM:-work-pr140r1}-$TAG-$s data $EVTS \
      > /home/xqian/tmp/pr140-arm-${PR140_ARM:-work-pr140r1}-$TAG-$s.log 2>&1
  echo "arm $TAG $s rc=$?"
done
echo "ARM $TAG DONE"
