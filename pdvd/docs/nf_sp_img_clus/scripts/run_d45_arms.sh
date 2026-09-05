#!/bin/bash
# doc pdvd/45 -- PDVD PR-chain arms (fork of run_d44_arms.sh, which stays doc 44's record):
# the SAME staging/pin/info conventions, plus
#   MODE=nu|stm   (default nu)  the full PR tail (-nu: track_fit/vertices/shower_track/mc Bee
#                               layers + tracking-pr.root) or the cosmic-tagger chain (-stm)
#   NOSTMFIT=1                  drop -stm-fit (default: stm_fit dump ON so track_fit can be
#                               graded against it)
#   WCT_EXCL_DUMP / WCT_TRAJ_DUMP / WCT_DQDX_DROP_DEBUG are inherited by wire-cell
#                               (getenv in TrackFitting) and echoed into $ARM.info
# Every event reads its provenance pctree: d39r2prov for the 21 doc-39 round-2 events,
# d41prov for the other 99 (scripts/stage_pr_tag.sh).  Fresh tags only (M13).
#
#   d45nu0    PIN=new, 120 events, MODE=nu                                   -- baseline (today's production knobs)
#   d45on     PIN=new, 120 events, MODE=nu, EXTRA="-S excl_t0_frame=true"    -- the fix
#
# Usage:
#   ARM=d45nu0 PIN=new [MODE=nu] [EVENTS=stm/events.txt|<file>] [JOBS=16] [NOSTMFIT=1] [EXTRA="-S ..."] \
#       ./docs/nf_sp_img_clus/scripts/run_d45_arms.sh
# JOBS is PER RUN; runs are launched sequentially here (one run's events in parallel).
set -u
ARM=${ARM:?ARM=<tag>}
PIN=${PIN:?PIN=ref|new}
JOBS=${JOBS:-16}
EXTRA=${EXTRA:-}
MODE=${MODE:-nu}
PINROOT=${PINROOT:-/home/xqian/tmp/d44_libpin}
cd "$(dirname "$0")/../../.." || exit 9      # pdvd/
case "$MODE" in nu) MODEFLAG=-nu ;; stm) MODEFLAG=-stm ;; *) echo "MODE must be nu|stm" >&2; exit 2 ;; esac
case " $EXTRA" in *"trackfitting_config="|*"trackfitting_config= "*)
    echo "REFUSING $ARM: empty trackfitting_config would drop the PDVD fitting parameters" >&2; exit 2;; esac
EVENTS=${EVENTS:-stm/events.txt}
PINDIR=$PINROOT/$PIN
[ -f "$PINDIR/libWireCellClus.so" ] || { echo "no pin $PINDIR" >&2; exit 2; }
export LD_LIBRARY_PATH="$PINDIR:${LD_LIBRARY_PATH:-}"
OUT=/home/xqian/tmp/d45_arms; mkdir -p "$OUT"
FITFLAG=-stm-fit; [ "${NOSTMFIT:-0}" = 1 ] && FITFLAG=""

n=0; runs=""
while read -r run idx rest; do
    [ -z "$run" ] && continue; [ "${run:0:1}" = "#" ] && continue
    e=$(printf '%06d' "$((10#$run))")_$idx
    if [ -d "work/${e}_d39r2prov" ]; then src=d39r2prov; else src=d41prov; fi
    ./scripts/stage_pr_tag.sh "$run" "$idx" "$ARM" "$src" >/dev/null || { echo "stage failed $e" >&2; exit 3; }
    n=$((n+1)); case " $runs " in *" $run "*) ;; *) runs="$runs $run" ;; esac
done < "$EVENTS"
echo "staged $n events for $ARM (pin $PINDIR, md5 $(md5sum "$PINDIR/libWireCellClus.so" | cut -c1-16) / root $(md5sum "$PINDIR/libWireCellRoot.so" | cut -c1-16)); runs:$runs; flags: $MODEFLAG $FITFLAG $EXTRA; env: WCT_EXCL_DUMP=${WCT_EXCL_DUMP:-} WCT_TRAJ_DUMP=${WCT_TRAJ_DUMP:-} WCT_DQDX_DROP_DEBUG=${WCT_DQDX_DROP_DEBUG:-}" | tee "$OUT/$ARM.info"
echo "toolkit $(git -C /nfs/data/1/xqian/toolkit-dev/toolkit rev-parse --short HEAD) wcp-porting-img $(git -C /nfs/data/1/xqian/toolkit-dev/wcp-porting-img rev-parse --short HEAD) $(date -Is)" | tee -a "$OUT/$ARM.info"

for run in $runs; do
    PDVD_PR_TLA="$EXTRA" PDVD_KEEP_CFG=1 PDVD_MAX_JOBS=$JOBS \
        ./run_pr_evt.sh -s "$ARM" $MODEFLAG $FITFLAG "$run" all > "$OUT/$ARM.$run.log" 2>&1
    echo "run $run rc=$? ($(date +%T)) loadavg $(cut -d' ' -f1 /proc/loadavg)" | tee -a "$OUT/$ARM.info"
done
r6=$(for run in $runs; do printf '%06d ' "$((10#$run))"; done)
zips=$(for p in $r6; do ls work/${p}_*_$ARM/mabc-pr.zip 2>/dev/null; done | wc -l)
roots=$(for p in $r6; do ls work/${p}_*_$ARM/tracking-pr.root work/${p}_*_$ARM/tracking-stm.root 2>/dev/null; done | wc -l)
echo "$ARM done: $zips mabc-pr.zip, $roots tracking*.root of $n staged" | tee -a "$OUT/$ARM.info"
