#!/bin/bash
# doc pdvd/103: blind shots of every scannable item, per detector, display arm by arm (shoot.sh --blind --hide-selection)
set -u
C=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/campaign
S=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan
D=/home/xqian/tmp/d103
det=$1; N=${2:-3}
mkdir -p $D/round_$det/shots
for sheet in $D/round_$det/set/sheet_*.tsv; do
  arm=$(basename $sheet .tsv); arm=${arm#sheet_}
  n=$(grep -v '^#' $sheet | tail -n +2 | wc -l); [ "$n" -eq 0 ] && continue
  R=$D/round_$det/shoot_$arm
  [ "$n" -lt "$N" ] && NP=$n || NP=$N
  bash $C/shoot.sh $R $det $sheet $D/round/prep_$arm $NP > $D/round_$det/logs/shoot_$arm.out 2>&1
  echo "$det $arm shoot rc=$? items=$n"
  for d in $R/shots/*/; do [ -d "$d" ] && mv "$d" $D/round_$det/shots/; done
done
python3 $C/mkzoom.py $D/round_$det/shots > $D/round_$det/logs/mkzoom.out 2>&1; echo "$det mkzoom rc=$?"
python3 $S/check_shots.py $D/round_$det/shots $D/round_$det/reshoot.txt > $D/round_$det/logs/check_shots.out 2>&1; echo "$det check_shots rc=$?"
echo "$det SHOOT_ALL_DONE"
