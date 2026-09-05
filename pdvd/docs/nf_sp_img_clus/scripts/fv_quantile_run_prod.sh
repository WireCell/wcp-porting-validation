#!/bin/bash
# doc pdvd/43: the exit-gap quantile FV arms, PRODUCTION chain, staged from d41prov (the
# 99 events with -save-assoc provenance, the same set as d41fvoff / d41fvon).
#   d43p80c3 : curved_fv + profile p80, cushion 3 cm (fv_tolerance y/z)
#   d43p90c3 : curved_fv + profile p90, cushion 3 cm
#   d43p90c5 : curved_fv + profile p90, cushion 5 cm
set -u
PDVD=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
LIBS="/home/xqian/toolkit-dev/local/lib/libWireCellClus.so /home/xqian/toolkit-dev/local/lib/libWireCellAux.so /home/xqian/toolkit-dev/local/lib/libWireCellMatch.so /nfs/data/1/xqian/toolkit-dev/toolkit/build/apps/wire-cell"
fingerprint () { for f in $LIBS; do stat -c '%n %s %Y' "$f"; done; }
declare -A TLA
TLA[d43prod]=""
echo "=== binary fingerprint BEFORE ==="; fingerprint
n=0
while read -r run idx; do
    for t in "${!TLA[@]}"; do
        d=$PDVD/work/${run}_${idx}_${t}
        [ -d "$d" ] || $PDVD/scripts/stage_pr_tag.sh "$run" "$idx" "$t" d41prov > /dev/null || exit 1
    done
    n=$((n+1))
done < /home/xqian/tmp/doc41/prov.txt
echo "staged $n events x 1 tag (PRODUCTION defaults, no TLA)"
cd "$PDVD" || exit 1
export PDVD_MAX_JOBS=16
arm () {
    local t=$1
    echo "=== ARM $t  (PDVD_PR_TLA='${TLA[$t]}')  start $(date +%H:%M:%S) ==="
    for r in 039252 039253 039349; do
        PDVD_PR_TLA="${TLA[$t]}" ./run_pr_evt.sh -s "$t" "$r" all > /home/xqian/tmp/doc43/arm_${t}_${r}.log 2>&1
        echo "  $t run $r rc=$?  $(grep -A3 'batch summary' /home/xqian/tmp/doc43/arm_${t}_${r}.log | grep -E 'ok:|failed:' | tr '\n' ' ')  $(date +%H:%M:%S)"
    done
    echo "  $t events with mabc-pr.zip: $(ls -d work/*_${t}/mabc-pr.zip 2>/dev/null | wc -l)"
}
arm d43prod &
wait
echo "=== binary fingerprint AFTER ==="; fingerprint
echo "=== done $(date +%H:%M:%S) ==="
