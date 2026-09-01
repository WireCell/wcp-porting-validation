#!/bin/bash
# doc pr/139 sec 24 -- the splitter-OFF arm on the CURRENT pinned binary.
# shower_split is true by default in the job now, so the only way to get one is
# the SBND_SHOWER_SPLIT_OFF hook (a TLA that sets it false; the jsonnet then
# suppresses the key and the C++ default false applies -- verified by 0 peels
# and by the key's absence from the compiled config).
# This exists because sec 24's first pass computed the rest-mass term from a PDG
# LOOKUP instead of measuring what is actually booked, and got it wrong.
set -u
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
export LD_LIBRARY_PATH=/home/xqian/tmp/pin-pr140r2:${LD_LIBRARY_PATH:-}
export SBND_SHOWER_SPLIT_OFF=1
for s in mcp1k mcp2k ncpi0 nuecc48; do
  EVTS=$(tr '\n' ' ' < /home/xqian/tmp/pr139-manifest-$s.lst)
  PR_JOBS=32 PR_EXTRA_STAGES=pr_display \
    ./run_pr_chain_batch.sh work-$s-grp0825 work-pr140r3-splitoff-$s data $EVTS \
      > /home/xqian/tmp/pr140r3-splitoff-$s.log 2>&1
  echo "arm splitoff $s rc=$?"
done
echo "SPLITOFF ARM DONE"
