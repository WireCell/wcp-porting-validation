#!/usr/bin/env bash
# doc pdvd/113 -- launch the retile_mode arms of ONE detector concurrently (base2 + none + footprint + no_paint), all on
# the pinned library, all with the Steiner dump on, all at the same JOBS, so their walls share one machine load
# (figs/113_pred.txt clause R).  Refuses to start unless the frozen rule's sha256 checks (the rule is frozen before any
# knob-on arm).  Each arm goes through d111_run_arms.sh (untouched), which refuses an existing arm tag (M13).
#
# Usage: DET=pdhd bash d113_run_levels.sh      (DET=pdvd for PDVD)
set -uo pipefail
DET=${DET:?set DET to pdhd or pdvd}
JOBS=${JOBS:-4}
S=$(cd "$(dirname "$0")" && pwd)
F=$S/../figs
P=/home/xqian/tmp/d113/libpin_d113
TR="WCT_STM_PATH_DEBUG=1 WCT_STEINER_GRAPH_DUMP=1"
(cd "$F" && sha256sum -c 113_pred.sha256) || { echo "REFUSING: figs/113_pred.txt does not match its frozen sha256" >&2; exit 2; }
case "$DET" in
  pdhd) SRC=d108hflip; X=h ;;
  pdvd) SRC=d103vflip; X=v ;;
  *) echo "DET must be pdhd or pdvd" >&2; exit 2 ;;
esac
cd /home/xqian/tmp/d113 || exit 2
declare -A TLA=([d113${X}base2]="" [d113${X}none]="-S retile_mode='none'" [d113${X}foot]="-S retile_mode='footprint'"
                [d113${X}nopaint]="-S retile_mode='no_paint'")
for ARM in d113${X}base2 d113${X}none d113${X}foot d113${X}nopaint; do
    nohup env ARM=$ARM DET=$DET SRC=$SRC JOBS=$JOBS PIN=$P LOGD=/home/xqian/tmp/d113/arm_$ARM TRACE_ENV="$TR" \
        PR_TLA="${TLA[$ARM]}" bash "$S/d111_run_arms.sh" > arm_$ARM.out 2>&1 &
    echo "launched $ARM tla=${TLA[$ARM]:-<none>} pid $!"
done
