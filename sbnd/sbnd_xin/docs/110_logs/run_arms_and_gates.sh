#!/bin/bash
# doc sbnd_xin/110: the four arms on the fixed binary, then every gate.
#   chain A: d110off (per-event, knobs absent)  -> d110grpoff (group, old cfg tree => knobs absent)
#   chain B: d110grp (group, knobs on)          -> d110dlgrp  (group, knobs on, DL vertex)
set -u
SX=/home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
D=/home/xqian/tmp/d110
G=$D/gates; mkdir -p "$G"
NEW=/home/xqian/tmp/d110-libsnap/new
NEWCFG=/home/xqian/tmp/d110-cfg/new/cfg
OLDCFG=/home/xqian/tmp/d109-cfg/new/cfg
A=$SX/scripts/d109_arms.sh
cd "$SX" || exit 1
for l in d110off d110grp d110grpoff d110dlgrp; do
    ls -d work-*-$l >/dev/null 2>&1 && { echo "STOP: work-*-$l exists (M13)"; exit 1; }
done

(
  $A d110off    $NEW SBND_NO_DL=1 SBND_ROOT_OUTPUT=1 PR_JOBS=16 PR_CFG_TREE=$NEWCFG > $D/arm-d110off.txt 2>&1
  $A d110grpoff $NEW SBND_NO_DL=1 SBND_ROOT_OUTPUT=1 PR_JOBS=12 PR_GROUP_SIZE=16 PR_CFG_TREE=$OLDCFG > $D/arm-d110grpoff.txt 2>&1
) &
PA=$!
(
  $A d110grp    $NEW SBND_NO_DL=1 SBND_ROOT_OUTPUT=1 PR_JOBS=12 PR_GROUP_SIZE=16 PR_CFG_TREE=$NEWCFG > $D/arm-d110grp.txt 2>&1
  $A d110dlgrp  $NEW SBND_ROOT_OUTPUT=1 PR_JOBS=14 PR_GROUP_SIZE=16 PR_CFG_TREE=$NEWCFG > $D/arm-d110dlgrp.txt 2>&1
) &
PB=$!
wait $PA $PB
grep -h "===" $D/arm-d110*.txt

ALLOW=(--allow Trun.toolkit_git Trun.wcp_git)
CC=$SX/docs/109_logs/r2/calib_classes.py
python3 scripts/d109_gate.py d109on2   d110off    "${ALLOW[@]}" --jobs 16 > $G/gate_on2_vs_d110off.txt 2>&1;    echo "gate on2 vs d110off rc=$?"
python3 scripts/d109_gate.py d109on2   d110grp    "${ALLOW[@]}" --jobs 16 > $G/gate_on2_vs_d110grp.txt 2>&1;    echo "gate on2 vs d110grp rc=$?"
python3 scripts/d109_gate.py d109grp   d110grpoff "${ALLOW[@]}" --jobs 16 > $G/gate_grp_vs_d110grpoff.txt 2>&1; echo "gate d109grp vs d110grpoff rc=$?"
python3 scripts/d109_gate.py d109dlon2 d110dlgrp  "${ALLOW[@]}" --jobs 16 > $G/gate_dlon2_vs_d110dlgrp.txt 2>&1; echo "gate dlon2 vs d110dlgrp rc=$?"
python3 $CC d109on2   d110grp   > $G/calib_classes_on2_vs_d110grp.txt 2>&1;     echo "classes on2 vs d110grp rc=$?"
python3 $CC d109dlon2 d110dlgrp > $G/calib_classes_dlon2_vs_d110dlgrp.txt 2>&1; echo "classes dlon2 vs d110dlgrp rc=$?"
python3 scripts/d109_root_checks.py d110grp   > $G/checks_d110grp.txt 2>&1;   echo "checks d110grp rc=$?"
python3 scripts/d109_root_checks.py d110dlgrp > $G/checks_d110dlgrp.txt 2>&1; echo "checks d110dlgrp rc=$?"
for f in $G/gate_*.txt; do echo "## $(basename $f)"; grep -v "^   (" $f | grep -v "root tree compared"; done
cat $G/calib_classes_*.txt
grep -h -A2 "check failures" $G/checks_*.txt
grep -c "DL vertex failed" work-*-d110dlgrp/wct_pr_g*.log 2>/dev/null | awk -F: '{s+=$2} END {print "DL vertex failed lines:", s}'
echo ALL110_DONE
