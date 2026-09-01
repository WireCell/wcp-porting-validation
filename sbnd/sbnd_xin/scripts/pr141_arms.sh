#!/bin/bash
# doc pr/141 M1 -- arm launcher.  FORK of scripts/pr140_arms.sh (M10; that
# script keeps producing doc pr/139's numbers and is untouched).  Two things
# differ and both matter:
#   * arm prefix work-pr141r1-*, and the binary is PINNED to
#     /home/xqian/tmp/pin-pr141 (md5 0c2e53f1..., verified equal to local/lib
#     right after wcbuild) so a peer's build cannot swap it mid-campaign.
#   * the `on` arm sets SBND_PI0_MU_HYP=1, the M1 knob, ALONE on the flipped
#     production config -- so its baseline is the `off` arm here, and the `off`
#     arm's own baseline is work-pr140r4-off-* (the previous binary, all knobs
#     off), which is what the knob-off hash gate compares.
#
#   ./scripts/pr141_arms.sh <armtag>      # env already exported by caller
set -u
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
TAG=$1
export LD_LIBRARY_PATH=${PR141_PIN:-/home/xqian/tmp/pin-pr141}:${LD_LIBRARY_PATH:-}
for s in mcp1k mcp2k ncpi0 nuecc48; do
  EVTS=$(tr '\n' ' ' < /home/xqian/tmp/pr139-manifest-$s.lst)
  PR_JOBS=32 PR_EXTRA_STAGES=pr_display \
    ./run_pr_chain_batch.sh work-$s-grp0825 work-pr141r1-$TAG-$s data $EVTS \
      > /home/xqian/tmp/pr141-arm-$TAG-$s.log 2>&1
  echo "arm $TAG $s rc=$?"
done
echo "ARM $TAG DONE"
