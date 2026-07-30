#!/bin/bash
# Summarize the 48-event two-arm run: per-event Bee-zip member-hash equality,
# the cosmic_tagger part-D line, and whether the z-prior comparators ran.
set -u
ROOT=${ROOT:-/home/xqian/tmp/geomab}
HA=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest/hash_archive.py
printf 'evt\tsame\trc_on\trc_off\thighest_y\tflag_on\tflag_off\tncmp\tncmpg\n'
for d in $(ls -d $ROOT/*/ | sort -t/ -k5 -n); do
    evt=$(basename "$d")
    ron=$(sed 's/rc=//' "$d/on/rc.txt" 2>/dev/null); rof=$(sed 's/rc=//' "$d/off/rc.txt" 2>/dev/null)
    hon=$(python3 $HA "$d/on/mabc-pr.zip" 2>/dev/null | awk '{print $1}')
    hof=$(python3 $HA "$d/off/mabc-pr.zip" 2>/dev/null | awk '{print $1}')
    same=$([ -n "$hon" ] && [ "$hon" = "$hof" ] && echo SAME || echo DIFF)
    lon="$d/on/wct_nupr_evt$evt.log"; lof="$d/off/wct_nupr_evt$evt.log"
    # grep -o, not sed: WCT log lines can tear mid-word under threading, and a
    # torn line would otherwise smuggle a whole log record into the field.
    hy=$(grep -m1 -oE "highest_y=[-0-9.]+ cm" "$lon" 2>/dev/null | grep -oE "[-0-9.]+")
    fon=$(grep -m1 -oE "flagp_cosmic=(true|false)" "$lon" 2>/dev/null | cut -d= -f2)
    fof=$(grep -m1 -oE "flagp_cosmic=(true|false)" "$lof" 2>/dev/null | cut -d= -f2)
    ncmp=$(grep -c "multiple candidates, calling compare_main_vertices" "$lon" 2>/dev/null)
    ncmpg=$(grep -c "compare_main_vertices_global: cluster" "$lon" 2>/dev/null)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$evt" "$same" "${ron:-?}" "${rof:-?}" "${hy:-na}" "${fon:-na}" "${fof:-na}" "$ncmp" "$ncmpg"
done
