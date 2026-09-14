#!/bin/bash
SX=/home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin; G=/home/xqian/tmp/d110/gates
cd $SX || exit 1
until grep -q "ALL110_DONE" /home/xqian/tmp/d110/run110.out; do sleep 20; done
A=(--allow Trun.toolkit_git Trun.wcp_git Trun.cfg_tree)
python3 scripts/d109_gate.py d109on2   d110off   "${A[@]}" --jobs 8 > $G/gate_on2_vs_d110off_allowcfg.txt 2>&1;   echo "RERUN on2 vs d110off rc=$?"
python3 scripts/d109_gate.py d109on2   d110grp   "${A[@]}" --jobs 8 > $G/gate_on2_vs_d110grp_allowcfg.txt 2>&1;   echo "RERUN on2 vs d110grp rc=$?"
python3 scripts/d109_gate.py d109dlon2 d110dlgrp "${A[@]}" --jobs 8 > $G/gate_dlon2_vs_d110dlgrp_allowcfg.txt 2>&1; echo "RERUN dlon2 vs d110dlgrp rc=$?"
for f in $G/gate_*_allowcfg.txt; do
  echo "## $(basename $f)"; grep -v "^   (" $f | grep -v "root tree compared"
  grep "^   (" $f | sed "s/.*\[//" | tr ',' '\n' | sed "s/[]')]//g;s/calib-pr-evt[0-9]*/calib-pr-evt*/" | sort | uniq -c | sort -rn | head -5
done
echo RERUN_DONE
