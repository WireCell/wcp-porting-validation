#!/bin/bash
# doc pr/108 Test B: WCT uBooNE arms via qlport/scripts/run_one.sh (ASLR off, DL off).
set -u; cd /nfs/data/1/xqian/toolkit-dev && exec direnv exec . bash -c '
cd toolkit/qlport/scripts
for arm in off on onkeep; do
  case $arm in off) E="";; on) E="QL_FIT_EXCLUSION=true";; onkeep) E="QL_FIT_EXCLUSION=true QL_DQDX_KEEP_ALL=true";; esac
  for idx in 1 4 6 16 22 23; do
    echo "=== $(date +%T) arm=$arm idx=$idx"; env $E ./run_one.sh $idx pr108_wct_$arm
  done
done; echo "=== ALL DONE $(date +%T)"'
