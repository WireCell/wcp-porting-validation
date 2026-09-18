#!/usr/bin/env bash
# doc pdvd/115 -- launch the base_weight_blank_alpha arms of ONE detector concurrently on the doc-115 pin, all with
# the Steiner dump on, all at the same JOBS.  The knob-OFF control of this round is d115?off (run BEFORE the rule was
# frozen, as the dump-extension gate; it is the base of every metric and the R denominator).  Refuses to start unless
# the frozen rule's sha256 checks.  Each arm goes through d111_run_arms.sh (untouched), which refuses an existing arm
# tag (M13).  Fork of d114_run_levels.sh (doc 114).
#
# Usage: DET=pdhd [JOBS=5] [LEVELS="bw2 bwp1 p3 p3bw2"] bash d115_run_levels.sh      (DET=pdvd for PDVD)
set -uo pipefail
DET=${DET:?set DET to pdhd or pdvd}
JOBS=${JOBS:-5}
LEVELS=${LEVELS:?set LEVELS (space-separated keys of the TLA table)}
S=$(cd "$(dirname "$0")" && pwd)
F=$S/../figs
P=${PIN:-/home/xqian/tmp/d115/libpin_d115}
TR="WCT_STM_PATH_DEBUG=1 WCT_STEINER_GRAPH_DUMP=1"
(cd "$F" && sha256sum -c 115_pred.sha256) || { echo "REFUSING: figs/115_pred.txt does not match its frozen sha256" >&2; exit 2; }
case "$DET" in
  pdhd) SRC=d108hflip; X=h ;;
  pdvd) SRC=d103vflip; X=v ;;
  *) echo "DET must be pdhd or pdvd" >&2; exit 2 ;;
esac
cd /home/xqian/tmp/d115 || exit 2
# the levels (figs/115_sizing_verdict.txt selection; figs/115_pred_amend1.txt: the tags bwa/bwb/p3bwa of the first launch
# carried placeholder TLAs and produced no output, so the levels are tagged bw2 / bwp1 / p3bw2)
declare -A TLA=([p3]="-S steiner_blank_plane_mode='prefer3'"
                [bw2]="-S steiner_base_weight_blank_alpha=2.0"
                [bwp1]="-S steiner_base_weight_blank_alpha=1.0 -S steiner_base_weight_scope='tree+path'"
                [p3bw2]="-S steiner_blank_plane_mode='prefer3' -S steiner_base_weight_blank_alpha=2.0")
for L in $LEVELS; do
    ARM=d115${X}$L
    [ -n "${TLA[$L]+x}" ] || { echo "unknown level $L" >&2; exit 2; }
    nohup env ARM=$ARM DET=$DET SRC=$SRC JOBS=$JOBS PIN=$P LOGD=/home/xqian/tmp/d115/arm_$ARM TRACE_ENV="$TR" \
        PR_TLA="${TLA[$L]}" bash "$S/d111_run_arms.sh" > arm_$ARM.out 2>&1 &
    echo "launched $ARM tla=${TLA[$L]:-<none>} pid $!"
done
