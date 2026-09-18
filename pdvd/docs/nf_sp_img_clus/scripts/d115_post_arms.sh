#!/usr/bin/env bash
# doc pdvd/115 -- every clause script of figs/115_pred.txt for one detector, after its arms finished.  Read-only on
# the arms.  Usage: DET=pdhd [LEVELS="p3 bw2 bwp1 p3bw2"] [JOBS=8] bash d115_post_arms.sh   (DET=pdvd for PDVD)
set -uo pipefail
DET=${DET:?set DET}
JOBS=${JOBS:-8}
LEVELS=${LEVELS:-"p3 bw2 bwp1 p3bw2"}
S=$(cd "$(dirname "$0")" && pwd); F=$S/../figs; cd "$S/.." || exit 2
case "$DET" in pdhd) X=h ;; pdvd) X=v ;; *) exit 2 ;; esac
OFF=d115${X}off; BASE114=d114${X}base; LOGD=/home/xqian/tmp/d115
# G2: this round's OFF arm is byte-identical to the doc-114 base (pre-knob pin)
[ -s $F/115_gate_off_$DET.txt ] || { python3 scripts/d111_identity_gate.py --det $DET --a $BASE114 --b $OFF > $F/115_gate_off_$DET.txt 2>&1; echo "gate off rc=$?"; }
for L in $LEVELS; do
  ARM=d115${X}$L
  [ -d $LOGD/arm_$ARM ] || { echo "no arm $ARM"; continue; }
  if [ "$L" = p3 ]; then   # G3: the prior knob's path is untouched by this build
    python3 scripts/d111_identity_gate.py --det $DET --a d114${X}p3 --b $ARM > $F/115_gate_p3_$DET.txt 2>&1; echo "gate p3 rc=$?"
  fi
  python3 scripts/d113_steiner_census.py --det $DET --arm $ARM --logd $LOGD/arm_$ARM --jobs $JOBS --out $F/115_steiner_$ARM > $LOGD/post_${ARM}_census.log 2>&1; echo "census $ARM rc=$?"
  python3 scripts/d111_eval.py --det $DET --base $OFF --arm $ARM --out $F/115_eval_${DET}_$L.txt > $LOGD/post_${ARM}_eval.log 2>&1; echo "eval $ARM rc=$?"
  python3 scripts/d113_compare.py --det $DET --base $OFF --arm $ARM --logd-base $LOGD/arm_$OFF --out $F/115_compare_${DET}_$L.txt > $LOGD/post_${ARM}_compare.log 2>&1; echo "compare $ARM rc=$?"
  python3 scripts/d114_support_census.py --det $DET --arm $ARM --logd $LOGD/arm_$ARM --jobs $JOBS --out $F/115_support_${DET}_$L > $LOGD/post_${ARM}_support.log 2>&1; echo "support $ARM rc=$?"
  python3 scripts/d115_proj_resid.py --det $DET --base $OFF --arm $ARM --jobs $JOBS --out $F/115_resid_${DET}_$L > $LOGD/post_${ARM}_resid.log 2>&1; echo "resid $ARM rc=$?"
  python3 scripts/d115_dqdx_compare.py --det $DET --base $OFF --arm $ARM --out $F/115_dqdx_${DET}_$L > $LOGD/post_${ARM}_dqdx.log 2>&1; echo "dqdx $ARM rc=$?"
done
CELLS="A0=$OFF"; for L in $LEVELS; do [ -d $LOGD/arm_d115${X}$L ] && CELLS="$CELLS,$(echo $L | tr a-z A-Z)=d115${X}$L"; done
python3 scripts/d113_grade.py --det $DET --cells $CELLS > $F/115_grade_$DET.txt 2>&1; echo "grade rc=$?"
