#!/bin/bash
# doc pdvd/102 P4: run d102_sim_chain.sh (sim -> NF -> SP -> imaging -> clustering) for the 18 ISO muons
# per detector.  At most JOBS chains at once (each chain is one multi-threaded wire-cell at a time).
set -u
S=/home/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts
JOBS=${JOBS:-4}
L=/home/xqian/tmp/d102/sim/batch_logs; mkdir -p $L
echo "[sim] start $(date -Is) jobs=$JOBS pin md5 $(md5sum /home/xqian/tmp/d102/libpin_d102/libWireCellClus.so | cut -c1-12)"
for det in pdvd pdhd; do for k in $(seq 0 17); do echo "$det $k"; done; done | \
  xargs -P "$JOBS" -L 1 bash -c 'DET=$0 K=$1 bash '"$S"'/d102_sim_chain.sh > '"$L"'/${0}_k${1}.log 2>&1; echo "[sim] $0 k=$1 rc=$? $(date +%H:%M:%S) load $(cut -d" " -f1 /proc/loadavg)"'
ok=$(ls /home/xqian/tmp/d102/sim/*/evt*/DONE 2>/dev/null | wc -l); bad=$(ls /home/xqian/tmp/d102/sim/*/evt*/FAIL_* 2>/dev/null | wc -l)
echo "[sim] DONE=$ok FAIL=$bad pin md5 after $(md5sum /home/xqian/tmp/d102/libpin_d102/libWireCellClus.so | cut -c1-12)"
echo SIM_BATCH_DONE
