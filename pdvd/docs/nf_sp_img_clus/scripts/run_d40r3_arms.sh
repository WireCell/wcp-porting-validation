#!/bin/bash
# doc pdvd/40 round 3 -- the 120-event PDVD PR manifest (stm/events.txt) under the
# retiler's bad-blob run bound (ImproveCluster_1 knobs bad_blob_max_run /
# bad_blob_report), one binary per pin, fresh tags only (M13).
#
# Every arm reads the SAME provenance pctree per event: d39r2prov for the 21
# doc-39 round-2 events, d41prov (this round) for the other 99, symlinked into
# work/<run6>_<idx>_<arm>/ by scripts/stage_pr_tag.sh.  All arms run the -nu
# chain (only it writes the calib dump whose 'steiner' block carries every
# cluster with a steiner_pc) with dl_weights='' so the neutrino tail is
# deterministic (the DL vertex is not bit-stable and sits downstream of
# everything graded).
#
#   d41ref     PIN=ref (pre-round libs), no knobs        -- the knob-OFF reference
#   d41base    PIN=new, bad_blob_report=true             -- census + baseline; vs d41ref
#                                                          proves OFF-path identity AND
#                                                          report neutrality in one gate
#   d41rep     PIN=new, as d41base                       -- repeat, the noise floor
#   d41fix<N>  PIN=new, report + bad_blob_max_run=<N> cm
#
# Usage:
#   ARM=d41base PIN=new [EVENTS=stm/events.txt|<file>] [JOBS=8] [EXTRA="-S ..."] \
#       ./docs/nf_sp_img_clus/scripts/run_d40r3_arms.sh
#   ARM=d41fix20 PIN=new EXTRA="-S retile_bad_blob_max_run=20 -S retile_bad_blob_report=true" ...
# JOBS is PER RUN (3 runs launched together => 3*JOBS concurrent wire-cell jobs).
set -u
ARM=${ARM:?ARM=<tag>}
PIN=${PIN:?PIN=ref|new}
JOBS=${JOBS:-8}
EXTRA=${EXTRA:-}
cd "$(dirname "$0")/../../.." || exit 9      # pdvd/
EVENTS=${EVENTS:-stm/events.txt}
PINDIR=/home/xqian/tmp/d41_libpin/$PIN
[ -f "$PINDIR/libWireCellClus.so" ] || { echo "no pin $PINDIR" >&2; exit 2; }
export LD_LIBRARY_PATH="$PINDIR:${LD_LIBRARY_PATH:-}"
OUT=/home/xqian/tmp/d41_arms; mkdir -p "$OUT"

# stage: one dir per event, symlinked to its provenance pctree
n=0
while read -r run idx rest; do
    e=$(printf '%06d_%s' "$run" "$idx")
    [ -d "work/${e}_${ARM}" ] && continue
    if [ -d "work/${e}_d39r2prov" ]; then src=d39r2prov; else src=d41prov; fi
    ./scripts/stage_pr_tag.sh "$run" "$idx" "$ARM" "$src" >/dev/null || { echo "stage failed $e" >&2; exit 3; }
    n=$((n+1))
done < <(awk 'NR>2 && $1 ~ /^[0-9]+$/ {print $1, $2}' "$EVENTS")
echo "staged $n dirs for $ARM (pin=$PIN extra='$EXTRA')"
md5sum "$PINDIR/libWireCellClus.so" | tee "$OUT/${ARM}.pin.md5"

for run in $(awk 'NR>2 && $1 ~ /^[0-9]+$/ {print $1}' "$EVENTS" | sort -u); do
    PDVD_PR_TLA="-S dl_weights='' $EXTRA" PDVD_KEEP_CFG=1 PDVD_MAX_JOBS=$JOBS \
        ./run_pr_evt.sh -nu -s "$ARM" "$run" all > "$OUT/${ARM}_${run}.log" 2>&1 &
done
wait
md5sum "$PINDIR/libWireCellClus.so" | tee -a "$OUT/${ARM}.pin.md5"
for run in $(awk 'NR>2 && $1 ~ /^[0-9]+$/ {print $1}' "$EVENTS" | sort -u); do
    echo "== $ARM $run"; grep -A3 "batch summary" "$OUT/${ARM}_${run}.log"
done
echo "$ARM ALL_DONE"
