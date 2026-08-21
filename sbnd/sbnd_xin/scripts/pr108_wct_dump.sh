#!/bin/bash
set -u; cd /nfs/data/1/xqian/toolkit-dev && exec direnv exec . bash -c '
cd toolkit/qlport/scripts
for arm in off on; do
  E=""; [ $arm = on ] && E="QL_FIT_EXCLUSION=true QL_DQDX_KEEP_ALL=true"
  for idx in 1 4 6 16 22 23; do
    D=/nfs/data/1/xqian/toolkit-dev/toolkit/qlport/scripts/sweep/pr108e_wct_$arm; mkdir -p $D
    echo "=== $(date +%T) arm=$arm idx=$idx"; env $E WCT_TRAJ_DUMP=$D/traj_$idx.dump ./run_one.sh $idx pr108e_wct_$arm
  done
done; echo "=== ALL DONE $(date +%T)"'
