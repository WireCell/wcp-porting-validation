#!/usr/bin/env bash
# doc pdvd/102 -- stage and run a PR data arm on the production -nu -stm-fit chain on a PINNED library.
# Fork by duplication of pdhd/docs/scripts/d16_run_arms.sh (untouched).  Differences: LD_LIBRARY_PATH=$PIN
# (a peer's wcbuild must not swap local/lib mid-arm) with the pin md5 recorded before and after, and a
# refusal to reuse an existing arm tag (CLAUDE.md M13).  Each event dir symlinks the SOURCE arm's pctree,
# so every arm of doc 102 reads byte-identical input.
#
# Usage:
#   ARM=d102hcs DET=pdhd SRC=d101hkf JOBS=3 PIN=/home/xqian/tmp/d102/libpin_d102 \
#     PR_TLA="-A trackfitting_config=<kf json> -S retile_sampler_strategy='charge_stepped'" d102_run_arms.sh
set -uo pipefail
IMG=/home/xqian/toolkit-dev/wcp-porting-img
DET=${DET:?set DET to pdvd or pdhd}
ARM=${ARM:?set ARM}
SRC=${SRC:?set SRC (the source arm whose pctree symlinks are reused)}
JOBS=${JOBS:-3}
PIN=${PIN:?set PIN (a private copy of local/lib)}
PR_TLA=${PR_TLA:-}
LOGD=${LOGD:-/home/xqian/tmp/d102/arm_$ARM}; mkdir -p "$LOGD"
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
for s in "$D"/work/*_"$SRC"; do
    [ -d "$s" ] || continue
    b=$(basename "$s"); pre=${b%_$SRC}
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
