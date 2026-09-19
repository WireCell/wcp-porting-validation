#!/bin/bash
# doc pdvd/116 sec 5 (fork of 103_shoot_round.sh, untouched): blind shots of every scannable item of a NAMED round dir, display arm by arm
# (fork of 103_shoot_all.sh, which is untouched and hard-codes round_<det>).
#   bash 116_shoot_round.sh DET ROUND_NAME [NPROC]     e.g.  bash 116_shoot_round.sh pdvd round_pdvd2 3
set -u
C=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/campaign
S=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan
D=/home/xqian/tmp/d116
det=$1; RN=$2; N=${3:-3}
RD=$D/$RN
[ -d "$RD/set" ] || { echo "no $RD/set" >&2; exit 2; }
mkdir -p $RD/shots $RD/logs
for sheet in $RD/set/sheet_*.tsv; do
  arm=$(basename $sheet .tsv); arm=${arm#sheet_}
  n=$(grep -v '^#' $sheet | tail -n +2 | wc -l); [ "$n" -eq 0 ] && continue
  R=$RD/shoot_$arm
  [ "$n" -lt "$N" ] && NP=$n || NP=$N
  bash $C/shoot.sh $R $det $sheet $D/round/prep_$arm $NP > $RD/logs/shoot_$arm.out 2>&1
  echo "$det $arm shoot rc=$? items=$n"
  for d in $R/shots/*/; do [ -d "$d" ] && mv "$d" $RD/shots/; done
done
python3 $C/mkzoom.py $RD/shots > $RD/logs/mkzoom.out 2>&1; echo "$det mkzoom rc=$?"
python3 $S/check_shots.py $RD/shots $RD/reshoot.txt > $RD/logs/check_shots.out 2>&1; echo "$det check_shots rc=$?"
echo "$det $RN SHOOT_ROUND_DONE"
