#!/usr/bin/env bash
# doc pdvd/109 -- stage and run a PR arm on a SUBSET of a source arm's events, on a pinned library.
#
# Fork by duplication of d102_run_arms.sh (untouched).  One difference: EVENTS, a
# space-separated list of <run6>_<evt> prefixes; only those event dirs are staged, so a
# 5-event display/gate arm can reuse a 61-event source arm's pctrees without re-running it.
# Everything else -- the pin md5 recorded before and after, the refusal to reuse an
# existing arm tag (CLAUDE.md M13), the per-event pctree symlink -- is unchanged.
#
# Usage:
#   ARM=d109hstm DET=pdhd SRC=d108hflip JOBS=5 PIN=/home/xqian/tmp/d102/libpin_d102 \
#     EVENTS="028084_3 028084_12 029107_5 029107_16 029107_18" d109_arms.sh
set -uo pipefail
IMG=/home/xqian/toolkit-dev/wcp-porting-img
DET=${DET:?set DET to pdvd or pdhd}
ARM=${ARM:?set ARM}
SRC=${SRC:?set SRC (the source arm whose pctree symlinks are reused)}
EVENTS=${EVENTS:?set EVENTS (space-separated <run6>_<evt> prefixes)}
JOBS=${JOBS:-3}
PIN=${PIN:?set PIN (a private copy of local/lib)}
PR_TLA=${PR_TLA:-}
LOGD=${LOGD:-/home/xqian/tmp/d109/arm_$ARM}; mkdir -p "$LOGD"
D=$IMG/$DET
case "$DET" in
  pdvd) JOBVAR=PDVD_MAX_JOBS; TLAVAR=PDVD_PR_TLA; EXTRA_ENV="PDVD_LIGHT_SUFFIX=_keep" ;;
  pdhd) JOBVAR=PDHD_MAX_JOBS; TLAVAR=PDHD_PR_TLA; EXTRA_ENV="" ;;
  *) echo "DET must be pdvd or pdhd" >&2; exit 2 ;;
esac
[ -s "$PIN/libWireCellClus.so" ] || { echo "REFUSING: no pin at $PIN" >&2; exit 2; }
if ls -d "$D"/work/*_"$ARM" >/dev/null 2>&1; then
    echo "REFUSING: $D/work/*_$ARM already exists (M13: new run, new tag)" >&2; exit 2
fi
md5_before=$(md5sum "$PIN/libWireCellClus.so" | cut -c1-12)
echo "[$ARM] det=$DET src=$SRC jobs=$JOBS pin=$PIN clus md5 $md5_before tla=${PR_TLA:-<none>} $(date -Is)"
n=0
for pre in $EVENTS; do
    s="$D/work/${pre}_${SRC}"
    [ -d "$s" ] || { echo "[$ARM] $pre NO SOURCE DIR $s"; continue; }
    ls "$s"/pctree-evt*.tar.gz >/dev/null 2>&1 || { echo "[$ARM] $pre NO PCTREE"; continue; }
    dst="$D/work/${pre}_${ARM}"; mkdir -p "$dst"
    for f in "$s"/pctree-evt*.tar.gz "$s"/pctree-evt*.tlas; do ln -sfn "$(readlink -f "$f")" "$dst/$(basename "$f")"; done
    n=$((n+1))
done
echo "[$ARM] staged $n event dir(s)"
rc_all=0
for run6 in $(ls -d "$D"/work/*_"$ARM" | sed -E 's#.*/([0-9]+)_[0-9]+_.*#\1#' | sort -u); do
    run=$((10#$run6))
    echo "[$ARM] run $run6 ($run) ... $(date +%H:%M:%S)"
    ( cd "$D" && env LD_LIBRARY_PATH="$PIN" WCT_PYLIB=off $EXTRA_ENV $JOBVAR="$JOBS" $TLAVAR="$PR_TLA" \
        ./run_pr_evt.sh -nu -stm-fit -s "$ARM" "$run" all ) > "$LOGD/run_${run6}.log" 2>&1
    rc=$?; echo "[$ARM] run $run6 rc=$rc loadavg $(cut -d' ' -f1 /proc/loadavg)"
    [ $rc -ne 0 ] && rc_all=$rc
done
ok=0; bad=0
for d in "$D"/work/*_"$ARM"; do
    if [ -s "$d/tracking-pr.root" ] && [ -s "$d/tracking-stm.root" ] && grep -q "CheckSTM_Michel: .* candidate(s)" "$d"/wct_pr_*.log 2>/dev/null
    then ok=$((ok+1)); else bad=$((bad+1)); echo "[$ARM] INCOMPLETE $(basename "$d")"; fi
done
md5_after=$(md5sum "$PIN/libWireCellClus.so" | cut -c1-12)
echo "[$ARM] complete $ok, incomplete $bad, run rc=$rc_all, pin md5 after $md5_after"
[ "$md5_before" = "$md5_after" ] || echo "[$ARM] *** PIN CHANGED MID-ARM ***"
echo "ARM_DONE $ARM"
