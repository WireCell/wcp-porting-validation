#!/usr/bin/env bash
# doc pdvd/114 -- launch the terminal_blank_plane_mode arms of ONE detector concurrently: the knob-OFF control (same
# pin, no TLA; must equal the d114 base arm and is the R denominator) plus one arm per level, all on the pinned
# library, all with the Steiner dump on, all at the same JOBS (figs/114_pred.txt clause R).  Refuses to start unless
# the frozen rule's sha256 checks.  Each arm goes through d111_run_arms.sh (untouched), which refuses an existing
# arm tag (M13).  Fork of d113_run_levels.sh (doc 113).
#
# Usage: DET=pdhd [JOBS=8] [LEVELS="p3 p3n10"] bash d114_run_levels.sh      (DET=pdvd for PDVD)
set -uo pipefail
DET=${DET:?set DET to pdhd or pdvd}
JOBS=${JOBS:-8}
LEVELS=${LEVELS:-"p3 p3n10"}
S=$(cd "$(dirname "$0")" && pwd)
F=$S/../figs
P=${PIN:-/home/xqian/tmp/d114/libpin_d114k}
TR="WCT_STM_PATH_DEBUG=1 WCT_STEINER_GRAPH_DUMP=1"
(cd "$F" && sha256sum -c 114_pred.sha256) || { echo "REFUSING: figs/114_pred.txt does not match its frozen sha256" >&2; exit 2; }
case "$DET" in
  pdhd) SRC=d108hflip; X=h ;;
  pdvd) SRC=d103vflip; X=v ;;
  *) echo "DET must be pdhd or pdvd" >&2; exit 2 ;;
esac
cd /home/xqian/tmp/d114 || exit 2
declare -A TLA=([off]="" [p3]="-S steiner_blank_plane_mode='prefer3'"
                [p3n10]="-S steiner_blank_plane_mode='prefer3+nearby' -S steiner_blank_plane_radius_cm=1.0"
                [p3n15]="-S steiner_blank_plane_mode='prefer3+nearby' -S steiner_blank_plane_radius_cm=1.5"
                [n20]="-S steiner_blank_plane_mode='nearby' -S steiner_blank_plane_radius_cm=2.0")
for L in off $LEVELS; do
    ARM=d114${X}$L
    nohup env ARM=$ARM DET=$DET SRC=$SRC JOBS=$JOBS PIN=$P LOGD=/home/xqian/tmp/d114/arm_$ARM TRACE_ENV="$TR" \
        PR_TLA="${TLA[$L]}" bash "$S/d111_run_arms.sh" > arm_$ARM.out 2>&1 &
    echo "launched $ARM tla=${TLA[$L]:-<none>} pid $!"
done
