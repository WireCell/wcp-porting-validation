#!/bin/bash
# usage: cmp_consumers.sh <dirA> <dirB>
#
# doc 77 round 2.  Exact cmp of every artifact compile_consumers.sh produced.
# Exit 0 only when every one is byte-identical.
set -u
A=${1:?}; B=${2:?}; fail=0; n=0
for fa in "$A"/*.json "$A"/prod.wcls "$A"/prod.standalone; do
    [ -s "$fa" ] || continue
    tag=$(basename "$fa"); fb="$B/$tag"; n=$((n+1))
    if [ ! -s "$fb" ]; then echo "MISSING $tag"; fail=$((fail+1)); continue; fi
    cmp -s "$fa" "$fb" || { echo "DIFFER  $tag"; fail=$((fail+1)); }
done
echo "=== $((n-fail))/$n identical; $fail differ ==="
[ $fail -eq 0 ]
