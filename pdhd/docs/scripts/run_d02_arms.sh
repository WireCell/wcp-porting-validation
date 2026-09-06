#!/bin/bash
# doc pdhd/02 -- PDHD STM-fit arms with alternative TrackFitting constants (fork BY
# DUPLICATION of pdvd/docs/nf_sp_img_clus/scripts/run_d44_arms.sh; the PDVD script is
# untouched).  One binary per pin, fresh tags only (M13).  Every event reads its pctree
# from work/029107_<evt>_stm0 (symlinked into the new tag dir).
#
#   d02ref   canonical JSON, no TLA           -- the same-epoch reference (gate partner of stmwc)
#   d02sig   TFJSON=stm/pdhd_track_fitting_d02.json     share-matched joint line (the graded arm)
#   d02sigb  TFJSON=stm/pdhd_track_fitting_d02b.json    rms-matched joint line
#   d02fix   TFJSON=stm/pdhd_track_fitting_d02fix.json  DT fixed at 8.2017, c only
#
# TFJSON (optional) is the RUNTIME TrackFitting parameter file, passed as
# -A trackfitting_config=<abs path>.  It never enters the compiled jsonnet, so an arm
# carrying one is graded on outputs only.  An EMPTY value would silently run the uBooNE
# C++ presets (doc pdvd/38, tag d38g2: 111/111 events changed), so the guard below
# refuses it; the file's md5 goes into work/.d02_<ARM>.info.
#
# Usage:
#   ARM=d02sig PIN=/home/xqian/tmp/pdhd02_libpin [TFJSON=stm/x.json] [JOBS=8] [EVENTS="0 6"] [EXTRA="-S ..."] \
#       ./docs/scripts/run_d02_arms.sh
set -u
ARM=${ARM:?ARM=<tag>}
PIN=${PIN:?PIN=<libpin dir>}
JOBS=${JOBS:-8}
EXTRA=${EXTRA:-}
TFJSON=${TFJSON:-}
EVENTS=${EVENTS:-all}
SRC=${SRC:-stm0}
cd "$(dirname "$0")/../.." || exit 9      # pdhd/
[ -d "$PIN" ] || { echo "no pin $PIN" >&2; exit 2; }
export LD_LIBRARY_PATH=$PIN
WC=/home/xqian/toolkit-dev/local/bin/wire-cell
if ldd $WC | grep -i wirecell | grep -qv "$PIN"; then echo "REFUSING: wire-cell libs not resolved from $PIN" >&2; exit 2; fi
TFNOTE="canonical"
if [ -n "$TFJSON" ]; then
    [ -f "$TFJSON" ] || { echo "no TFJSON $TFJSON" >&2; exit 2; }
    TFABS=$(readlink -f "$TFJSON")
    python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$TFABS" || { echo "TFJSON is not valid JSON" >&2; exit 2; }
    EXTRA="$EXTRA -A trackfitting_config=$TFABS"
    TFNOTE="$TFABS md5 $(md5sum "$TFABS" | cut -c1-16)"
fi
case " $EXTRA" in *"trackfitting_config="|*"trackfitting_config= "*)
    echo "REFUSING $ARM: empty trackfitting_config would drop the PDHD fitting parameters" >&2; exit 2;; esac
# stage: every stm0 event (or the listed ones) gets a fresh tag dir with the pctree symlinked
for d in work/029107_*_$SRC; do
    e=${d#work/029107_}; e=${e%_$SRC}
    if [ "$EVENTS" != all ]; then case " $EVENTS " in *" $e "*) ;; *) continue ;; esac; fi
    n=work/029107_${e}_$ARM
    if [ -s "$n/tracking-stm.root" ]; then echo "REFUSING $ARM: $n already has outputs (M13: new run => new tag)" >&2; exit 3; fi
    mkdir -p "$n"; ln -sfn "$PWD/$d"/pctree-evt*.tar.gz "$n/"; ln -sfn "$PWD/$d"/pctree-evt*.tlas "$n/"
done
{
  echo "arm=$ARM pin=$PIN tf=$TFNOTE extra='$EXTRA' events=$EVENTS date=$(date -Is)"
  md5sum "$PIN"/libWireCellClus.so "$PIN"/libWireCellRoot.so
  echo "toolkit=$(git -C /home/xqian/toolkit-dev/toolkit rev-parse --short HEAD) wcp=$(git -C /home/xqian/toolkit-dev/wcp-porting-img rev-parse --short HEAD)"
} > "work/.d02_$ARM.info"
if [ "$EVENTS" = all ]; then
    PDHD_MAX_JOBS=$JOBS PDHD_KEEP_CFG=1 PDHD_PR_TLA="$EXTRA" ./run_pr_evt.sh -s "$ARM" -stm -stm-fit 029107 all
    rc=$?
else
    rc=0
    for e in $EVENTS; do
        PDHD_KEEP_CFG=1 PDHD_PR_TLA="$EXTRA" ./run_pr_evt.sh -s "$ARM" -stm -stm-fit 029107 "$e" || rc=$?
    done
fi
echo "arm=$ARM rc=$rc markers=$(ls work/029107_*_$ARM/pr_resource_029107_*.txt 2>/dev/null | wc -l)" | tee -a "work/.d02_$ARM.info"
exit $rc
