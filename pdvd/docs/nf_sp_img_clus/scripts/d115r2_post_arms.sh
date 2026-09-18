#!/usr/bin/env bash
# doc pdvd/115 round 2 -- every clause script of figs/115r2_pred.txt for one detector, after its arms finished.  Read-only
# on the arms.  Every output carries the 115r2_ prefix so the round-1 records (figs/115_*) are untouched; the base rows
# (115_steiner_d115?off.json, 115_support_<det>_off.txt, 115_gate_off_<det>.txt) are round 1's and are reused.
# Fork of d115_post_arms.sh.
# Usage: DET=pdhd [LEVELS="p3bwp025 p3bwp05 p3bwp1"] [JOBS=8] bash d115r2_post_arms.sh   (DET=pdvd for PDVD)
set -uo pipefail
DET=${DET:?set DET}
JOBS=${JOBS:-8}
LEVELS=${LEVELS:-"p3bwp025 p3bwp05 p3bwp1"}
S=$(cd "$(dirname "$0")" && pwd); F=$S/../figs; cd "$S/.." || exit 2
case "$DET" in pdhd) X=h ;; pdvd) X=v ;; *) exit 2 ;; esac
OFF=d115${X}off; LOGD=/home/xqian/tmp/d115; PFX=115r2
[ -s $F/115_steiner_$OFF.json ] || { echo "no round-1 base census $F/115_steiner_$OFF.json"; exit 2; }
for L in $LEVELS; do
  ARM=d115${X}$L
  [ -d $LOGD/arm_$ARM ] || { echo "no arm $ARM"; continue; }
  python3 scripts/d113_steiner_census.py --det $DET --arm $ARM --logd $LOGD/arm_$ARM --jobs $JOBS --out $F/${PFX}_steiner_$ARM > $LOGD/r2/post_${ARM}_census.log 2>&1; echo "census $ARM rc=$?"
  python3 scripts/d111_eval.py --det $DET --base $OFF --arm $ARM --out $F/${PFX}_eval_${DET}_$L.txt > $LOGD/r2/post_${ARM}_eval.log 2>&1; echo "eval $ARM rc=$?"
  python3 scripts/d113_compare.py --det $DET --base $OFF --arm $ARM --logd-base $LOGD/arm_$OFF --out $F/${PFX}_compare_${DET}_$L.txt > $LOGD/r2/post_${ARM}_compare.log 2>&1; echo "compare $ARM rc=$?"
  python3 scripts/d114_support_census.py --det $DET --arm $ARM --logd $LOGD/arm_$ARM --jobs $JOBS --out $F/${PFX}_support_${DET}_$L > $LOGD/r2/post_${ARM}_support.log 2>&1; echo "support $ARM rc=$?"
  python3 scripts/d115_proj_resid.py --det $DET --base $OFF --arm $ARM --jobs $JOBS --out $F/${PFX}_resid_${DET}_$L > $LOGD/r2/post_${ARM}_resid.log 2>&1; echo "resid $ARM rc=$?"
  python3 scripts/d115_dqdx_compare.py --det $DET --base $OFF --arm $ARM --out $F/${PFX}_dqdx_${DET}_$L > $LOGD/r2/post_${ARM}_dqdx.log 2>&1; echo "dqdx $ARM rc=$?"
done
# the tag grade with the round-1 anchors as extra cells (P3, BWP1, P3BW2) so the T line of every level is in one file
CELLS="A0=$OFF,P3=d115${X}p3,BWP1=d115${X}bwp1,P3BW2=d115${X}p3bw2"
for L in $LEVELS; do [ -d $LOGD/arm_d115${X}$L ] && CELLS="$CELLS,$(echo $L | tr a-z A-Z)=d115${X}$L"; done
python3 scripts/d113_grade.py --det $DET --cells $CELLS > $F/${PFX}_grade_$DET.txt 2>&1; echo "grade rc=$?"
