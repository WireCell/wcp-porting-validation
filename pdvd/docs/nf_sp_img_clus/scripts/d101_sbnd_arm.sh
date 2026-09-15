#!/bin/bash
# doc pdvd/101 sec 6 -- the SBND PR knob-OFF gate arm for this round's C++ (TrackFitting,
# BlobSampler Stepped, Facade/Grouping, ClusteringFlagMatchedMains): SBND binds all of them.
#
# Fork by duplication (CLAUDE.md M10) of sbnd_xin/scripts/pr146_satarm.sh (that script stays
# untouched).  Differences: the Q/L source is work-<s>-d102m (the d97fv trees pr146 read were
# retired by the cleanup rounds), the manifest is the pr146 arm-C movers plus sentinels that still
# exist there (16 events), the neutrino vertex is GEOMETRIC (SBND_NO_DL=1: the SCN net is not
# bit-stable, CLAUDE.md M4), and there is no TLA -- both tags run the same config:
#   TAG=d101sold  PIN=/home/xqian/tmp/d101/libpin    (pre-round)
#   TAG=d101snew  PIN=/home/xqian/tmp/d101/libpin_k  (this round, knobs at defaults)
# Refuses an existing output tree (M13).
#
# Usage: TAG=... PIN=... [JOBS=4] bash d101_sbnd_arm.sh > /home/xqian/tmp/d101/sbnd_<TAG>.log 2>&1
set -u
JOBS=${JOBS:-4}
PIN=${PIN:?set PIN}
TAG=${TAG:?set TAG}
SX=/home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
cd "$SX" || exit 2

export LD_LIBRARY_PATH="$PIN:${LD_LIBRARY_PATH:-}"
export PR_EXTRA_STAGES=pr_display
export SBND_NO_DL=1
unset PR_GROUP_SIZE PR_EXTRA_TLA

EV_mcp1k="408534 321371 320667 54341 395610 350935 172794 175896"
EV_mcp2k="74336 170098 392009 94392 101828 393505 47212 52693"

echo "=== d101 sbnd arm $TAG  jobs=$JOBS  pin=$PIN  $(date +%F_%H:%M:%S)"
md5sum "$PIN/libWireCellClus.so"
for s in mcp1k mcp2k; do
  OUT="work-$s-$TAG"
  [ -e "$OUT" ] && { echo "REFUSING: $OUT exists (M13)"; exit 2; }
done
for s in mcp1k mcp2k; do
  eval "EVTS=\$EV_$s"
  OUT="work-$s-$TAG"
  echo "--- $s -> $OUT  ($(echo $EVTS | wc -w) events)  $(date +%H:%M:%S)"
  PR_JOBS=$JOBS ./run_pr_chain_batch.sh "work-$s-d102m" "$OUT" data $EVTS
  echo "--- $s rc=$? events=$(ls -d $OUT/pr_evt*/ 2>/dev/null | wc -l)  $(date +%H:%M:%S) loadavg=$(cut -d' ' -f1 /proc/loadavg)"
done
md5sum "$PIN/libWireCellClus.so"
echo "=== d101 sbnd arm $TAG DONE $(date +%F_%H:%M:%S)"
