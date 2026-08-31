#!/bin/bash
# doc pr/108 sec 10 -- dQ/dx system dump for the 4 remaining uBooNE events (idx 1,6,16,22)
set -u; cd /nfs/data/1/xqian/toolkit-dev && exec direnv exec . bash -c '
cd toolkit/qlport/scripts
for arm in off on; do
  E=""; [ $arm = on ] && E="QL_FIT_EXCLUSION=true QL_DQDX_KEEP_ALL=true"
  for idx in 1 6 16 22; do
    D=/nfs/data/1/xqian/toolkit-dev/toolkit/qlport/scripts/sweep/pr108h_wct_$arm; mkdir -p $D
    echo "=== $(date +%T) arm=$arm idx=$idx"; env $E WCT_TRAJ_DUMP=$D/traj_$idx.dump WCT_DQDX_DUMP=$D/dqdx_$idx.dump ./run_one.sh $idx pr108h_wct_$arm
  done
done; echo "=== ALL DONE $(date +%T)"'
