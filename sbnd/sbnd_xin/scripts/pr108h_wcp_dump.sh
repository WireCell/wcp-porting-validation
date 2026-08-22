#!/bin/bash
# doc pr/108 sec 10 -- prototype dQ/dx system dump for events 6505 6532 6650 6805
set -u
ARM=$1
D=/nfs/data/1/xqian/toolkit-dev/toolkit/qlport/scripts/sweep/pr108h_wcp_$ARM; mkdir -p $D
for ev in 6505 6532 6650 6805; do
  WCP_TRAJ_DUMP=$D/traj_$ev.dump WCP_DQDX_DUMP=$D/dqdx_$ev.dump \
    /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin/scripts/run_wcp.sh $ARM $ev
done
echo "=== ALL DONE $(date +%T)"
