#!/usr/bin/env bash
# doc pdvd/114 -- every clause script of figs/114_pred.txt for one detector, after its three arms (off, p3, p3n10)
# finished.  Read-only on the arms.  Usage: DET=pdhd bash d114_post_arms.sh   (DET=pdvd for PDVD)
set -uo pipefail
DET=${DET:?set DET}
JOBS=${JOBS:-8}
S=$(cd "$(dirname "$0")" && pwd); F=$S/../figs; cd "$S/.." || exit 2
case "$DET" in pdhd) X=h ;; pdvd) X=v ;; *) exit 2 ;; esac
BASE=d114${X}base; OFF=d114${X}off; LOGD=/home/xqian/tmp/d114
# G2: the knob build's OFF path is byte-identical to the base (pre-knob pin)
python3 scripts/d111_identity_gate.py --det $DET --a $BASE --b $OFF > $F/114_gate_off_$DET.txt 2>&1; echo "gate rc=$?"
for L in p3 p3n10; do
  ARM=d114${X}$L
  python3 scripts/d113_steiner_census.py --det $DET --arm $ARM --logd $LOGD/arm_$ARM --jobs $JOBS --out $F/114_steiner_$ARM > $LOGD/post_${ARM}_census.log 2>&1; echo "census $ARM rc=$?"
  python3 scripts/d111_eval.py --det $DET --base $BASE --arm $ARM --out $F/114_eval_${DET}_$L.txt > $LOGD/post_${ARM}_eval.log 2>&1; echo "eval $ARM rc=$?"
  python3 scripts/d111_eval.py --det $DET --base $OFF --arm $ARM --out $F/114_evalR_${DET}_$L.txt > $LOGD/post_${ARM}_evalR.log 2>&1; echo "evalR $ARM rc=$?"
  python3 scripts/d113_compare.py --det $DET --base $BASE --arm $ARM --logd-base $LOGD/arm_$BASE --out $F/114_compare_${DET}_$L.txt > $LOGD/post_${ARM}_compare.log 2>&1; echo "compare $ARM rc=$?"
  python3 scripts/d114_support_census.py --det $DET --arm $ARM --logd $LOGD/arm_$ARM --jobs $JOBS --out $F/114_support_${DET}_$L > $LOGD/post_${ARM}_support.log 2>&1; echo "support $ARM rc=$?"
done
python3 scripts/d113_grade.py --det $DET --cells A0=$BASE,P3N=d114${X}p3n10,P3=d114${X}p3 > $F/114_grade_$DET.txt 2>&1; echo "grade rc=$?"
