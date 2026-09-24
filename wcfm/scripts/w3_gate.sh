#!/bin/bash
# doc 02 sec 3.1 determinism gate: two full img+clus runs of the wcfm manifest under
# `setarch x86_64 -R` through abtest/run_events.sh (snapshots abtest/snap/<label>/), plus the
# truth tiers and the pctree copied into the snapshots, then abtest/ab_compare.sh.
# Usage: scripts/w3_gate.sh [labelA] [labelB]     (refuses to overwrite an existing label, M13)
set -u
WCFM_DIR=$(cd "$(dirname "$0")/.." && pwd)
AB=$(cd "$WCFM_DIR/../abtest" && pwd)
LA=${1:-wcfm-w3-a}; LB=${2:-wcfm-w3-b}
for L in "$LA" "$LB"; do
    [ -d "$AB/snap/$L" ] && { echo "REFUSING: snapshot $L exists" >&2; exit 2; }
done
extra() {
    local L=$1 det run evt r6
    while read -r det run evt; do
        case "$det" in ''|\#*) continue ;; esac
        r6=$(printf '%06d' "$((10#$run))")
        cp -f "$WCFM_DIR/work/${r6}_${evt}"/clusters-tru*-*.tar.gz "$WCFM_DIR/work/${r6}_${evt}"/pctree-evt*.tar.gz \
              "$AB/snap/$L/${det}_${r6}_${evt}/" 2>/dev/null
    done < "$WCFM_DIR/abtest_events.txt"
}
cd "$AB" || exit 1
for L in "$LA" "$LB"; do
    WCFM_SETARCH=1 ./run_events.sh "$L" both "$WCFM_DIR/abtest_events.txt" > "$WCFM_DIR/work/.gate_$L.log" 2>&1
    rc=$?
    echo "run_events $L rc=$rc"
    # a failed run leaves stale work-dir files that ab_compare would "PASS" on: abort instead
    [ $rc -eq 0 ] || { echo "ABORT: run_events $L failed (see work/.gate_$L.log)" >&2; exit 1; }
    for m in "$AB/snap/$L"/*/img_meta.txt "$AB/snap/$L"/*/clus_meta.txt; do
        grep -q '^rc=0$' "$m" || { echo "ABORT: $m is not rc=0" >&2; exit 1; }
    done
    extra "$L"
done
./ab_compare.sh "$LA" "$LB"
