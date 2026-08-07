#!/bin/bash
# doc pr/40 G1: knob-off byte-identical proof.
# Usage: g1_compare.sh <arm_A_dir> <arm_B_dir>
set -u
AB=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest
A="$1"; B="$2"
n_ok=0; n_bad=0; n_missing=0
for pra in "$A"/pr_evt*; do
    evt=$(basename "$pra" | sed 's/pr_evt//')
    prb="$B/pr_evt${evt}"
    if [ ! -d "$prb" ]; then
        echo "MISSING evt=$evt in $B"
        n_missing=$((n_missing+1))
        continue
    fi
    for member in mabc-pr.zip pctree-pr-evt${evt}.tar.gz; do
        fa="$pra/$member"; fb="$prb/$member"
        if [ ! -f "$fa" ] || [ ! -f "$fb" ]; then
            echo "MISSING FILE evt=$evt member=$member (a=$([ -f "$fa" ] && echo y || echo n) b=$([ -f "$fb" ] && echo y || echo n))"
            n_missing=$((n_missing+1))
            continue
        fi
        ha=$(python3 "$AB/hash_archive.py" "$fa" | awk '{print $1}')
        hb=$(python3 "$AB/hash_archive.py" "$fb" | awk '{print $1}')
        if [ "$ha" = "$hb" ]; then
            n_ok=$((n_ok+1))
        else
            echo "DIFF evt=$evt member=$member  a=$ha  b=$hb"
            n_bad=$((n_bad+1))
        fi
    done
done
echo "=== summary: ok=$n_ok diff=$n_bad missing=$n_missing ==="
