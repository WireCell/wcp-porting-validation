#!/usr/bin/env bash
# doc pdvd/115 round 2 -- launch the tree+path pricing arms WITH prefer3 of ONE detector concurrently on the doc-115 pin
# (unchanged binary, clus md5 2efa7fa09325), all with the Steiner dump on, all at the same JOBS.  The knob-OFF control
# stays d115?off (round 1).  Refuses to start unless the frozen round-2 rule's sha256 checks AND the pin's clus library
# still has the round-1 md5.  Each arm goes through d111_run_arms.sh (untouched), which refuses an existing arm tag (M13).
# Fork of d115_run_levels.sh (round 1); the level table holds real values (no placeholders: grep -n '__' must be empty).
#
# Usage: DET=pdhd [JOBS=5] LEVELS="p3bwp025 p3bwp05 p3bwp1" bash d115r2_run_levels.sh      (DET=pdvd for PDVD)
set -uo pipefail
DET=${DET:?set DET to pdhd or pdvd}
JOBS=${JOBS:-5}
LEVELS=${LEVELS:?set LEVELS (space-separated keys of the TLA table)}
S=$(cd "$(dirname "$0")" && pwd)
F=$S/../figs
P=${PIN:-/home/xqian/tmp/d115/libpin_d115}
PIN_MD5=2efa7fa09325
TR="WCT_STM_PATH_DEBUG=1 WCT_STEINER_GRAPH_DUMP=1"
(cd "$F" && sha256sum -c 115r2_pred.sha256) || { echo "REFUSING: figs/115r2_pred.txt does not match its frozen sha256" >&2; exit 2; }
M=$(md5sum "$P/libWireCellClus.so" | cut -c1-12)
[ "$M" = "$PIN_MD5" ] || { echo "REFUSING: pin clus md5 $M != $PIN_MD5" >&2; exit 2; }
case "$DET" in
  pdhd) SRC=d108hflip; X=h ;;
  pdvd) SRC=d103vflip; X=v ;;
  *) echo "DET must be pdhd or pdvd" >&2; exit 2 ;;
esac
cd /home/xqian/tmp/d115 || exit 2
# the levels (figs/115r2_pred.txt): prefer3 + tree+path pricing at alpha 0.25 / 0.5 / 1.0
declare -A TLA=([p3bwp025]="-S steiner_blank_plane_mode='prefer3' -S steiner_base_weight_blank_alpha=0.25 -S steiner_base_weight_scope='tree+path'"
                [p3bwp05]="-S steiner_blank_plane_mode='prefer3' -S steiner_base_weight_blank_alpha=0.5 -S steiner_base_weight_scope='tree+path'"
                [p3bwp1]="-S steiner_blank_plane_mode='prefer3' -S steiner_base_weight_blank_alpha=1.0 -S steiner_base_weight_scope='tree+path'")
for L in $LEVELS; do
    ARM=d115${X}$L
    [ -n "${TLA[$L]+x}" ] || { echo "unknown level $L" >&2; exit 2; }
    nohup env ARM=$ARM DET=$DET SRC=$SRC JOBS=$JOBS PIN=$P LOGD=/home/xqian/tmp/d115/arm_$ARM TRACE_ENV="$TR" \
        PR_TLA="${TLA[$L]}" bash "$S/d111_run_arms.sh" > arm_$ARM.out 2>&1 &
    echo "launched $ARM tla=${TLA[$L]} pid $!"
done
