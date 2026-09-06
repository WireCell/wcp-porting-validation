#!/bin/bash
# doc pdhd/04 sec 10 -- END TO END (clustering + Q/L + PR) on the 6-event manifest.
#
# Fork BY DUPLICATION of docs/scripts/run_d03_arms.sh, which runs the PR job only
# on a pre-existing pctree; that script is untouched.  This one re-runs the
# CLUSTERING job too, because the knob under test (the clustering job's
# wrapped_channel_charge) is what writes the pctree.
#
# BOTH arms must run on ONE binary: the existing stm0/d04bee arms are from
# 2026-09-05 and several unrelated commits have landed since, so comparing
# against them would confound the knob with everything else.  Run this twice
# with the same PIN, once with CLUS_TLA empty (the 2026-09-06 defaults, fix ON)
# and once with CLUS_TLA="-S wrapped_channel_charge=false" (pre-fix).
#
# The PR job is left at its production settings in both arms: its own
# same-named knob has been true since 2026-09-05, so it is not the variable.
#
# Usage:
#   ARM=d05mON  PIN=<libpin dir> ./docs/scripts/run_d04_manifest.sh
#   ARM=d05mOFF PIN=<libpin dir> CLUS_TLA="-S wrapped_channel_charge=false" ./docs/scripts/run_d04_manifest.sh
#
#   EVENTS="0 1 12 16 20 22"   the manifest (default)
#   SRC=stm0                   arm whose imaging-input symlinks are reused
#   JOBS=6                     concurrent wire-cell processes per stage
set -u
ARM=${ARM:?ARM=<tag>}
PIN=${PIN:?PIN=<libpin dir>}
CLUS_TLA=${CLUS_TLA:-}
EVENTS=${EVENTS:-"0 1 12 16 20 22"}
SRC=${SRC:-stm0}
JOBS=${JOBS:-6}
cd "$(dirname "$0")/../.." || exit 9      # pdhd/
[ -d "$PIN" ] || { echo "no pin $PIN" >&2; exit 2; }
export LD_LIBRARY_PATH=$PIN
WC=/home/xqian/toolkit-dev/local/bin/wire-cell
if ldd $WC | grep -i wirecell | grep -qv "$PIN"; then
    echo "REFUSING: wire-cell libs not resolved from $PIN" >&2; exit 2
fi

# M13: a new run gets a new tag, never an existing label dir.
for e in $EVENTS; do
    n=work/029107_${e}_$ARM
    if [ -s "$n/tracking-stm.root" ] || ls "$n"/pctree-evt*.tar.gz >/dev/null 2>&1; then
        echo "REFUSING $ARM: $n already has outputs (M13: new run => new tag)" >&2; exit 3
    fi
done

# Stage the imaging inputs: copy the SOURCE arm's symlinks (they point into
# wcp-porting-img/pdhd/work/029107_<evt>/), never the products.
for e in $EVENTS; do
    d=work/029107_${e}_$SRC
    [ -d "$d" ] || { echo "no source arm $d" >&2; exit 4; }
    n=work/029107_${e}_$ARM
    mkdir -p "$n"
    for f in "$d"/*; do
        [ -L "$f" ] || continue
        ln -sfn "$(readlink -f "$f")" "$n/$(basename "$f")"
    done
done

{
  echo "arm=$ARM pin=$PIN clus_tla='$CLUS_TLA' events='$EVENTS' src=$SRC date=$(date -Is)"
  md5sum "$PIN"/libWireCellClus.so "$PIN"/libWireCellRoot.so
  echo "toolkit=$(git -C /home/xqian/toolkit-dev/toolkit rev-parse --short HEAD) wcp=$(git -C /home/xqian/toolkit-dev/wcp-porting-img rev-parse --short HEAD)"
} > "work/.d04m_$ARM.info"

# Concurrency throttle: launch, count, and block on `wait -n` once JOBS are in
# flight.  Verified empirically (12 jobs at JOBS=3 -> max 3 concurrent), not
# assumed -- `wait -n` returns on the FIRST job to finish, which is what makes
# the running-- bookkeeping correct.
run_stage () {                     # $1 = label, rest = a function to call with the event id
    local label="$1"; shift
    local rc=0 running=0
    for e in $EVENTS; do
        ( "$@" "$e" ) &
        running=$((running+1))
        if [ "$running" -ge "$JOBS" ]; then wait -n 2>/dev/null || rc=$?; running=$((running-1)); fi
    done
    wait || rc=$?
    echo "stage=$label rc=$rc"
    return $rc
}

do_clus () { PDHD_KEEP_CFG=1 PDHD_CLUS_TLA="$CLUS_TLA" ./run_clus_evt.sh -s "$ARM" -save-pctree 029107 "$1"; }
do_pr   () { PDHD_KEEP_CFG=1 ./run_pr_evt.sh -s "$ARM" -stm-fit 029107 "$1"; }

run_stage clustering do_clus
crc=$?
# Completion marker is the product, never the log (feedback_completion_marker_not_the_log).
missing=""
for e in $EVENTS; do
    ls work/029107_${e}_$ARM/pctree-evt*.tar.gz >/dev/null 2>&1 || missing="$missing $e"
done
[ -z "$missing" ] || { echo "MISSING pctree for events:$missing" >&2; echo "$crc" > "work/.d04m_$ARM.rc"; exit 5; }

run_stage pr do_pr
prc=$?
missing=""
for e in $EVENTS; do
    [ -s "work/029107_${e}_$ARM/mabc-pr.zip" ] || missing="$missing $e"
done
[ -z "$missing" ] && echo "all PR outputs present" || echo "MISSING mabc-pr.zip for events:$missing" >&2

echo "arm=$ARM clus_rc=$crc pr_rc=$prc pctrees=$(ls work/029107_*_$ARM/pctree-evt*.tar.gz 2>/dev/null | wc -l) prbee=$(ls work/029107_*_$ARM/mabc-pr.zip 2>/dev/null | wc -l)" | tee -a "work/.d04m_$ARM.info"
echo "$prc" > "work/.d04m_$ARM.rc"
exit $prc
