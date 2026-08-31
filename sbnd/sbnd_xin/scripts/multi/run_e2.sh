#!/bin/bash
# E2 -- the control: the OTHER SIX flagged events, each ALONE and IN-PROCESS,
# exactly the configuration doc 81 used when it re-ran them as single-event
# groups.  If they flip too, 99438 is not distinct.
set -u
SX=$(cd "$(dirname "$0")/../.." && pwd -P)
OUT=${1:-/home/xqian/tmp/d82r2}
cd "$SX" || exit 1
run_one() {
    local sample=$1 evt=$2
    DRAWS=3 PRECOMPILE=0 ./scripts/multi/repro_ql_nondet.sh \
        "work-${sample}-grp0825" "$OUT/e2-$evt" "$evt" \
        > "$OUT/e2-$evt.log" 2>&1
}
N=0
for pair in mcp1k:286191 mcp1k:292643 mcp2k:53793 mcp2k:161043 mcp2k:321101 mcp2k:350816; do
    run_one "${pair%%:*}" "${pair##*:}" &
    N=$((N+1))
    [ $((N % 3)) -eq 0 ] && wait
done
wait
echo "E2 done"
