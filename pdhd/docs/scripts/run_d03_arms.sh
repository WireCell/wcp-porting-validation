#!/bin/bash
# doc pdhd/03 -- PDHD CheckSTM_Michel arms (fork BY DUPLICATION of docs/scripts/run_d02_arms.sh;
# that script is untouched).  One binary per pin, fresh tags only (M13).  Every event reads its
# pctree from work/029107_<evt>_<SRC> (default stm0; symlinked into the new tag dir).
#
#   MODE=-nu         (default) the doc-03 chain: ... check_stm_michel, tracking_visitor, pr_display
#   MODE=-nu-legacy  the pre-doc-03 neutrino tail (gate partner)
#   MODE=-stm        production cosmic-tagger chain (gate partner)
# -stm-fit is always appended (tracking-stm.root; save_stm_fit is PDHD production anyway).
#
# Usage:
#   ARM=d03nu1 PIN=/home/xqian/tmp/d47_libpin/new5 [MODE=-nu] [JOBS=10] [EVENTS="0 6"] [EXTRA="-S ..."] \
#       ./docs/scripts/run_d03_arms.sh
set -u
ARM=${ARM:?ARM=<tag>}
PIN=${PIN:?PIN=<libpin dir>}
MODE=${MODE:--nu}
JOBS=${JOBS:-10}
EXTRA=${EXTRA:-}
EVENTS=${EVENTS:-all}
SRC=${SRC:-stm0}
cd "$(dirname "$0")/../.." || exit 9      # pdhd/
[ -d "$PIN" ] || { echo "no pin $PIN" >&2; exit 2; }
export LD_LIBRARY_PATH=$PIN
WC=/home/xqian/toolkit-dev/local/bin/wire-cell
if ldd $WC | grep -i wirecell | grep -qv "$PIN"; then echo "REFUSING: wire-cell libs not resolved from $PIN" >&2; exit 2; fi
case " $EXTRA" in *"trackfitting_config="|*"trackfitting_config= "*)
    echo "REFUSING $ARM: empty trackfitting_config would drop the PDHD fitting parameters" >&2; exit 2;; esac
for d in work/029107_*_$SRC; do
    e=${d#work/029107_}; e=${e%_$SRC}
    if [ "$EVENTS" != all ]; then case " $EVENTS " in *" $e "*) ;; *) continue ;; esac; fi
    n=work/029107_${e}_$ARM
    if [ -s "$n/tracking-stm.root" ] || [ -s "$n/tracking-pr.root" ]; then echo "REFUSING $ARM: $n already has outputs (M13: new run => new tag)" >&2; exit 3; fi
    mkdir -p "$n"; ln -sfn "$PWD/$d"/pctree-evt*.tar.gz "$n/"; ln -sfn "$PWD/$d"/pctree-evt*.tlas "$n/"
done
{
  echo "arm=$ARM pin=$PIN mode=$MODE extra='$EXTRA' events=$EVENTS date=$(date -Is)"
  md5sum "$PIN"/libWireCellClus.so "$PIN"/libWireCellRoot.so
  echo "toolkit=$(git -C /home/xqian/toolkit-dev/toolkit rev-parse --short HEAD) wcp=$(git -C /home/xqian/toolkit-dev/wcp-porting-img rev-parse --short HEAD)"
} > "work/.d03_$ARM.info"
if [ "$EVENTS" = all ]; then
    PDHD_MAX_JOBS=$JOBS PDHD_KEEP_CFG=1 PDHD_PR_TLA="$EXTRA" ./run_pr_evt.sh -s "$ARM" $MODE -stm-fit 029107 all
    rc=$?
else
    rc=0
    for e in $EVENTS; do
        PDHD_KEEP_CFG=1 PDHD_PR_TLA="$EXTRA" ./run_pr_evt.sh -s "$ARM" $MODE -stm-fit 029107 "$e" || rc=$?
    done
fi
echo "arm=$ARM rc=$rc markers=$(ls work/029107_*_$ARM/pr_resource_029107_*.txt 2>/dev/null | wc -l)" | tee -a "work/.d03_$ARM.info"
echo "$rc" > "work/.d03_$ARM.rc"
exit $rc
