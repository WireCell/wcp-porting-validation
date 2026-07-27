#!/bin/bash
# Compare two compile_all_cfg.sh output dirs.  Master's pgraph.jsonnet adds a
# `_pnode` key (sibling of `data`, never read by ConfigManager) to every
# pgraph-wrapped component, so the gate is: identical after del(._pnode), same
# element count, same component order, and identical Pgrapher `edges` array.
#
# Usage: ./cmp_cfg.sh <dirA> <dirB>
set -u
A=${1:?usage: cmp_cfg.sh <dirA> <dirB>}
B=${2:?usage: cmp_cfg.sh <dirA> <dirB>}
norm='walk(if type=="object" then del(._pnode) else . end)'
overall=PASS
printf '%-16s %9s %-10s %-10s %-10s %s\n' JOB ELEMENTS ORDER EDGES NORMDIFF PNODE
for fa in "$A"/*.json; do
    tag=$(basename "$fa" .json); fb="$B/$tag.json"
    if [ ! -s "$fb" ]; then printf '%-16s  MISSING in %s\n' "$tag" "$B"; overall=FAIL; continue; fi
    na=$(jq 'length' "$fa"); nb=$(jq 'length' "$fb")
    cnt="$na->$nb"; [ "$na" = "$nb" ] || { cnt="$na->$nb!!"; overall=FAIL; }
    if diff -q <(jq -r '.[]|(.type+":"+(.name//""))' "$fa") \
                <(jq -r '.[]|(.type+":"+(.name//""))' "$fb") >/dev/null
    then ord=same; else ord=DIFFERS; overall=FAIL; fi
    ea=$(jq -c '[.[]|select(.type=="Pgrapher")|.data.edges]' "$fa")
    eb=$(jq -c '[.[]|select(.type=="Pgrapher")|.data.edges]' "$fb")
    if [ "$ea" = "$eb" ]; then edg=same; else edg=DIFFERS; overall=FAIL; fi
    nd=$(diff <(jq -S "$norm" "$fa") <(jq -S "$norm" "$fb") | wc -l)
    [ "$nd" -eq 0 ] || overall=FAIL
    pn=$(jq '[.[]|select(has("_pnode"))]|length' "$fb")
    ind=$(jq '[.[]|.data|..|objects|select(has("_pnode"))]|length' "$fb")
    [ "$ind" -eq 0 ] || { pn="$pn/IN-DATA:$ind"; overall=FAIL; }
    printf '%-16s %9s %-10s %-10s %-10s %s\n' "$tag" "$cnt" "$ord" "$edg" "$nd" "$pn"
done
echo "=== OVERALL: $overall ==="
[ "$overall" = PASS ]
