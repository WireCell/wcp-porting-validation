#!/bin/bash
# doc pdvd/42 -- PDVD STM-fit arms: the cosmic-tagger chain (-stm, the production
# default since doc 39) plus the Magnify-tracking dump (-stm-fit), one binary per
# pin, fresh tags only (M13).  Every event reads its provenance pctree: d39r2prov
# for the 21 doc-39 round-2 events, d41prov for the other 99 (scripts/stage_pr_tag.sh).
#
#   d42fit      PIN=new, 120 events            -- the graded arm (T_proj_data per pass)
#   d42fitold   PIN=ref, 2 events              -- pre-fix dump, the tree-hash gate partner
#   d42fitnew   PIN=new, 2 events              -- same events, fixed dump
#   d42stmchk   PIN=new, 2 events, NOSTMFIT=1  -- production chain, vs d41stmon (identity gate)
#
# Usage:
#   ARM=d42fit PIN=new [EVENTS=stm/events.txt|<file>] [JOBS=16] [NOSTMFIT=1] [EXTRA="-S ..."] \
#       ./docs/nf_sp_img_clus/scripts/run_d42_arms.sh
# JOBS is PER RUN; runs are launched sequentially here (one run's events in parallel).
set -u
ARM=${ARM:?ARM=<tag>}
PIN=${PIN:?PIN=ref|new}
JOBS=${JOBS:-16}
EXTRA=${EXTRA:-}
cd "$(dirname "$0")/../../.." || exit 9      # pdvd/
EVENTS=${EVENTS:-stm/events.txt}
PINDIR=/home/xqian/tmp/d42_libpin/$PIN
[ -f "$PINDIR/libWireCellClus.so" ] || { echo "no pin $PINDIR" >&2; exit 2; }
export LD_LIBRARY_PATH="$PINDIR:${LD_LIBRARY_PATH:-}"
OUT=/home/xqian/tmp/d42_arms; mkdir -p "$OUT"
FITFLAG=-stm-fit; [ "${NOSTMFIT:-0}" = 1 ] && FITFLAG=""

n=0; runs=""
while read -r run idx rest; do
    [ -z "$run" ] && continue; [ "${run:0:1}" = "#" ] && continue
    e=$(printf '%06d' "$((10#$run))")_$idx
    if [ -d "work/${e}_d39r2prov" ]; then src=d39r2prov; else src=d41prov; fi
    ./scripts/stage_pr_tag.sh "$run" "$idx" "$ARM" "$src" >/dev/null || { echo "stage failed $e" >&2; exit 3; }
    n=$((n+1)); case " $runs " in *" $run "*) ;; *) runs="$runs $run" ;; esac
done < "$EVENTS"
echo "staged $n events for $ARM (pin $PIN, md5 $(md5sum "$PINDIR/libWireCellClus.so" | cut -c1-16) / root $(md5sum "$PINDIR/libWireCellRoot.so" | cut -c1-16)); runs:$runs; flags: -stm $FITFLAG $EXTRA" | tee "$OUT/$ARM.info"
echo "toolkit $(git -C /nfs/data/1/xqian/toolkit-dev/toolkit rev-parse --short HEAD) wcp-porting-img $(git -C /nfs/data/1/xqian/toolkit-dev/wcp-porting-img rev-parse --short HEAD) $(date -Is)" | tee -a "$OUT/$ARM.info"

for run in $runs; do
    # only the staged events of this run: run_pr_evt.sh 'all' takes every work/<run6>_*_<ARM> dir
    PDVD_PR_TLA="$EXTRA" PDVD_KEEP_CFG=1 PDVD_MAX_JOBS=$JOBS \
        ./run_pr_evt.sh -s "$ARM" -stm $FITFLAG "$run" all > "$OUT/$ARM.$run.log" 2>&1
    echo "run $run rc=$? ($(date +%T)) loadavg $(cut -d' ' -f1 /proc/loadavg)" | tee -a "$OUT/$ARM.info"
done
r6=$(for run in $runs; do printf '%06d ' "$((10#$run))"; done)
zips=$(for p in $r6; do ls work/${p}_*_$ARM/mabc-pr.zip 2>/dev/null; done | wc -l)
roots=$(for p in $r6; do ls work/${p}_*_$ARM/tracking-stm.root 2>/dev/null; done | wc -l)
echo "$ARM done: $zips mabc-pr.zip, $roots tracking-stm.root of $n staged" | tee -a "$OUT/$ARM.info"
