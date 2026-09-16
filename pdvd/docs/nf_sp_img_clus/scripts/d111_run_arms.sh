#!/usr/bin/env bash
# doc pdvd/111 -- stage and run a PR arm (production -nu -stm-fit chain) on a pinned library, optionally with the
# log-only trajectory traces on, keeping each event's STDOUT.
#
# Fork by duplication of d109_arms.sh (untouched).  Differences:
#   * EVENTS is optional: unset = every event dir of the SOURCE arm (d102_run_arms.sh behaviour), set = a subset;
#   * TRACE_ENV (e.g. "WCT_STM_PATH_DEBUG=1 WCT_TRAJ_ASSOC_DEBUG=1") is passed to wire-cell.  The traces print to
#     stdout, which run_pr_evt.sh does not fold into wct_pr_*.log (a spdlog file sink); in 'all' mode the runner
#     writes each event's stdout+stderr to work/.batch_pr_<run6>_<evt>_<arm>.log.  After the run each such log is
#     gzipped into $LOGD/evt_<run6>_<evt>.log.gz and the per-event STMPATHN / TRAJASSOC line counts are listed, so
#     an arm whose trace silently vanished is visible;
#   * the pin md5 before/after, the refusal to reuse an arm tag (M13) and the per-event pctree symlink are unchanged.
#
# Usage:
#   ARM=d111htr DET=pdhd SRC=d108hflip JOBS=6 PIN=/home/xqian/tmp/d111/libpin_d111 \
#     TRACE_ENV="WCT_STM_PATH_DEBUG=1 WCT_TRAJ_ASSOC_DEBUG=1" [EVENTS="029107_16 ..."] [PR_TLA="..."] d111_run_arms.sh
set -uo pipefail
IMG=/home/xqian/toolkit-dev/wcp-porting-img
DET=${DET:?set DET to pdvd or pdhd}
ARM=${ARM:?set ARM}
SRC=${SRC:?set SRC (the source arm whose pctree symlinks are reused)}
EVENTS=${EVENTS:-}
JOBS=${JOBS:-3}
PIN=${PIN:?set PIN (a private copy of local/lib)}
PR_TLA=${PR_TLA:-}
TRACE_ENV=${TRACE_ENV:-}
LOGD=${LOGD:-/home/xqian/tmp/d111/arm_$ARM}; mkdir -p "$LOGD"
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
echo "[$ARM] det=$DET src=$SRC jobs=$JOBS pin=$PIN clus md5 $md5_before tla=${PR_TLA:-<none>} trace=${TRACE_ENV:-<none>} $(date -Is)"
if [ -z "$EVENTS" ]; then
    EVENTS=$(ls -d "$D"/work/*_"$SRC" 2>/dev/null | sed -E "s#.*/([0-9]+_[0-9]+)_$SRC\$#\1#" | sort)
fi
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
    ( cd "$D" && env LD_LIBRARY_PATH="$PIN" WCT_PYLIB=off $EXTRA_ENV $TRACE_ENV $JOBVAR="$JOBS" $TLAVAR="$PR_TLA" \
        ./run_pr_evt.sh -nu -stm-fit -s "$ARM" "$run" all ) > "$LOGD/run_${run6}.log" 2>&1
    rc=$?; echo "[$ARM] run $run6 rc=$rc loadavg $(cut -d' ' -f1 /proc/loadavg)"
    [ $rc -ne 0 ] && rc_all=$rc
done
ok=0; bad=0
for d in "$D"/work/*_"$ARM"; do
    b=$(basename "$d"); pre=${b%_$ARM}
    bl="$D/work/.batch_pr_${pre}_${ARM}.log"
    if [ -f "$bl" ]; then
        np=$(grep -c '^STMPATHN ' "$bl" || true); na=$(grep -c '^TRAJASSOC ' "$bl" || true)
        gzip -c "$bl" > "$LOGD/evt_${pre}.log.gz" && rm -f "$bl"
        echo "[$ARM] trace $pre STMPATHN=$np TRAJASSOC=$na"
    else
        echo "[$ARM] trace $pre NO BATCH LOG"
    fi
    if [ -s "$d/tracking-pr.root" ] && [ -s "$d/tracking-stm.root" ] && grep -q "CheckSTM_Michel: .* candidate(s)" "$d"/wct_pr_*.log 2>/dev/null
    then ok=$((ok+1)); else bad=$((bad+1)); echo "[$ARM] INCOMPLETE $b"; fi
done
md5_after=$(md5sum "$PIN/libWireCellClus.so" | cut -c1-12)
echo "[$ARM] complete $ok, incomplete $bad, run rc=$rc_all, pin md5 after $md5_after"
[ "$md5_before" = "$md5_after" ] || echo "[$ARM] *** PIN CHANGED MID-ARM ***"
echo "ARM_DONE $ARM"
