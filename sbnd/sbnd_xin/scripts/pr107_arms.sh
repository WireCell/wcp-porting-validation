#!/bin/bash
# doc pr/107 -- dqdx_fit_keep_all_points: OFF gate + ON arm, nueCC48 + NCpi0 (owner 2026-08-21).
set -u
cd /nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin
for s in nuecc48 ncpi0; do
  evts=$(cat /home/xqian/tmp/vtx105-evts-$s.txt)
  echo "=== $(date +%T) arm=off sample=$s"
  PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-$s-ql0819 work-pr107-off-$s data $evts > /home/xqian/tmp/pr107_off_$s.log 2>&1
  echo "=== $(date +%T) arm=off sample=$s rc=$? ok=$(cat work-pr107-off-$s/pr_evt*/rc.txt 2>/dev/null | grep -c 'rc=0')"
done
for s in nuecc48 ncpi0; do
  evts=$(cat /home/xqian/tmp/vtx105-evts-$s.txt)
  echo "=== $(date +%T) arm=on sample=$s"
  SBND_DQDX_FIT_KEEP_ALL_POINTS=true SBND_DL_VTX_HARVEST=true PR_JOBS=32 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh work-$s-ql0819 work-pr107-on-$s data $evts > /home/xqian/tmp/pr107_on_$s.log 2>&1
  echo "=== $(date +%T) arm=on sample=$s rc=$? ok=$(cat work-pr107-on-$s/pr_evt*/rc.txt 2>/dev/null | grep -c 'rc=0')"
done
echo "=== ALL DONE $(date +%T)"
