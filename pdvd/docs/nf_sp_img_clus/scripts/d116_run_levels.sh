#!/usr/bin/env bash
# doc pdvd/116 -- launch the arms of ONE detector concurrently on the doc-115 pin (unchanged binary, clus md5
# 2efa7fa09325), Steiner dump on, all at the same JOBS.  Fork of d115r2_run_levels.sh (untouched).  Two families:
#   SERIES=d115  bwp05            the un-run PDVD arm of doc 115 round 2 (figs/115r2_pred_amend1.txt), tag d115vbwp05
#   SERIES=d116  <tagger levels>  the tagger-retune sweep of doc 116 (figs/116_pred.txt), tags d116{h,v}<level>
# Refuses to start unless the frozen rule named by PRED verifies (sha256sum -c) AND the pin's clus library still has
# the doc-115 md5.  Each arm goes through d111_run_arms.sh (untouched), which refuses an existing arm tag (M13).
# The level table holds real values only: the launcher greps its own TLA block for '__' and refuses on a hit.
#
# Usage: DET=pdvd SERIES=d115 PRED=115r2_pred.sha256 LEVELS="bwp05" [JOBS=8] bash d116_run_levels.sh
#        DET=pdhd SERIES=d116 PRED=116_pred.sha256 TRAJ=p3bwp05 LEVELS="r1 r2 r3" [JOBS=5] bash d116_run_levels.sh
set -uo pipefail
DET=${DET:?set DET to pdhd or pdvd}
SERIES=${SERIES:?set SERIES (d115 or d116)}
PRED=${PRED:?set PRED (sha256 file under figs/)}
JOBS=${JOBS:-5}
LEVELS=${LEVELS:?set LEVELS (space-separated keys of the TLA table)}
S=$(cd "$(dirname "$0")" && pwd)
SELF=$S/$(basename "$0")
F=$S/../figs
P=${PIN:-/home/xqian/tmp/d115/libpin_d115}
PIN_MD5=2efa7fa09325
TR="WCT_STM_PATH_DEBUG=1 WCT_STEINER_GRAPH_DUMP=1"
LOGROOT=${LOGROOT:-/home/xqian/tmp/$SERIES}
(cd "$F" && sha256sum -c "$PRED") || { echo "REFUSING: figs/$PRED does not verify" >&2; exit 2; }
M=$(md5sum "$P/libWireCellClus.so" | cut -c1-12)
[ "$M" = "$PIN_MD5" ] || { echo "REFUSING: pin clus md5 $M != $PIN_MD5" >&2; exit 2; }
case "$DET" in
  pdhd) SRC=d108hflip; X=h ;;
  pdvd) SRC=d103vflip; X=v ;;
  *) echo "DET must be pdhd or pdvd" >&2; exit 2 ;;
esac
mkdir -p "$LOGROOT" && cd "$LOGROOT" || exit 2
# ---- TLA table (BEGIN_TLA / END_TLA bracket the placeholder check) ----
# BEGIN_TLA
declare -A TLA=(
  [bwp05]="-S steiner_base_weight_blank_alpha=0.5 -S steiner_base_weight_scope='tree+path'"
  [p3bwp05]="-S steiner_blank_plane_mode='prefer3' -S steiner_base_weight_blank_alpha=0.5 -S steiner_base_weight_scope='tree+path'"
)
# the tagger ladder of figs/116_pred.txt, appended to the trajectory TLAs of TRAJ (bwp05 | p3bwp05) when SERIES=d116
declare -A LEVEL=(
  [r1]="-S stm_proton_muon_guard=true"
  [r2]="-S stm_proton_muon_guard=true -S stm_michel_extra={michel_min_kink_deg:20.0,michel_max_len_cm:30.0}"
  [r3]="-S stm_proton_muon_guard=true -S stm_michel_extra={michel_min_kink_deg:20.0,michel_max_len_cm:30.0,michel_range_energy_ke_min:5.0}"
)
# END_TLA
if [ "$SERIES" = d116 ]; then
    TRAJ=${TRAJ:?set TRAJ (bwp05 or p3bwp05) for the d116 tagger levels}
    [ -n "${TLA[$TRAJ]+x}" ] || { echo "unknown TRAJ $TRAJ" >&2; exit 2; }
    for L in $LEVELS; do
        [ -n "${LEVEL[$L]+x}" ] || { echo "unknown tagger level $L" >&2; exit 2; }
        TLA[$L]="${TLA[$TRAJ]} ${LEVEL[$L]}"
    done
fi
if sed -n '/^# BEGIN_TLA/,/^# END_TLA/p' "$SELF" | grep -q '__'; then echo "REFUSING: placeholder in the TLA table" >&2; exit 2; fi
for L in $LEVELS; do
    ARM=${SERIES}${X}$L
    [ -n "${TLA[$L]+x}" ] || { echo "unknown level $L" >&2; exit 2; }
    nohup env ARM=$ARM DET=$DET SRC=$SRC JOBS=$JOBS PIN=$P LOGD=$LOGROOT/arm_$ARM TRACE_ENV="$TR" \
        PR_TLA="${TLA[$L]}" bash "$S/d111_run_arms.sh" > arm_$ARM.out 2>&1 &
    echo "launched $ARM tla=${TLA[$L]} pid $! (logs $LOGROOT/arm_$ARM, stdout $LOGROOT/arm_$ARM.out)"
done
