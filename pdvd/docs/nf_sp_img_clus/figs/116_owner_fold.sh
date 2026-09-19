#!/bin/bash
# doc pdvd/116 sec 9 -- fold the owner's own116v / own116h labels into records and re-grade the rule with the owner's
# verdicts at the highest precedence.  Run after the owner has labelled on :5017 (PDVD) / :5023 (PDHD); refuses an
# existing record (M13).  DET=pdvd|pdhd|both (default both).
set -e
IMG=/home/xqian/toolkit-dev/wcp-porting-img; S=$IMG/pdvd/docs/nf_sp_img_clus/scripts; F=$IMG/pdvd/docs/nf_sp_img_clus/figs
O=/home/xqian/tmp/d116/own
for det in ${DET:-pdvd pdhd}; do
  case $det in pdvd) tag=own116v; a0=d115voff; t=d115vp3bwp05; r=v ;; pdhd) tag=own116h; a0=d115hoff; t=d115hp3bwp05; r=h ;; esac
  L=$IMG/$det/work/stm_michel_labels/$tag/labels.json; REC=$IMG/$det/docs/scan/${det}_stm_michel_${tag}_verdicts.json
  [ -f "$L" ] || { echo "$det: no labels at $L yet"; continue; }
  python3 $S/d116_owner_scan_score.py --det $det --set $O/set_$det --labels $L --shown-arm d116${r}r2 --record-out $REC > $F/116_${tag}_$det.txt; echo "$det score rc=$?"
  python3 $S/d116_grade.py --det $det --cells A0=$a0,T=$t,R1=d116${r}r1,R2=d116${r}r2,R3=d116${r}r3 \
      --extra-record $IMG/$det/docs/scan/${det}_stm_michel_smx116_verdicts.json --owner-record $REC --split-half \
      --movers-out $F/116_movers_own_$det > $F/116_grade_${det}_own116.txt; echo "$det grade rc=$?"
  grep -E "^  R2 |T -> " $F/116_grade_${det}_own116.txt
done
# then: sha256sum -c $F/116_label_shas_before_own116.txt   (no other label tag changed)
