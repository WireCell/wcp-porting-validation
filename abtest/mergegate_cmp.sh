#!/bin/bash
# Compare two mergegate arms stage-by-stage using member-content hashes
# (hash_archive.py -- tar/zip embed mtimes so raw cmp/md5sum is meaningless, M2).
#
# Usage: ./mergegate_cmp.sh <armA> <armB>
set -u
A=/home/xqian/tmp/mergegate/${1:?usage: mergegate_cmp.sh <armA> <armB>}
B=/home/xqian/tmp/mergegate/${2:?usage: mergegate_cmp.sh <armA> <armB>}
HA=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest/hash_archive.py

overall=PASS
declare -A st_pass st_fail
for da in "$A"/*/; do
    ev=$(basename "$da"); db="$B/$ev"
    if [ ! -d "$db" ]; then echo "== $ev  MISSING in $2"; overall=FAIL; continue; fi
    echo "== $ev"
    for stage in nfsp img clus; do
        ma="$da/${stage}_meta.txt"; mb="$db/${stage}_meta.txt"
        if [ -f "$ma" ] && [ -f "$mb" ]; then
            wa=$(awk -F= '/^wall_s/{print $2}' "$ma"); wb=$(awk -F= '/^wall_s/{print $2}' "$mb")
            ra=$(awk -F= '/^maxrss_kb/{printf "%d", $2/1024}' "$ma"); rb=$(awk -F= '/^maxrss_kb/{printf "%d", $2/1024}' "$mb")
            printf '   %-5s wall %ss -> %ss   rss %sMB -> %sMB\n' "$stage" "$wa" "$wb" "$ra" "$rb"
        fi
    done
    for fa in "$da"*.tar.bz2 "$da"*.tar.gz "$da"*.zip; do
        [ -f "$fa" ] || continue
        f=$(basename "$fa"); fb="$db/$f"
        # which stage does this artifact belong to?
        case "$f" in *sp-frames*) s=nfsp ;; clusters-apa-*) s=img ;; mabc-*) s=clus ;; *) s=other ;; esac
        if [ ! -f "$fb" ]; then echo "   FAIL  [$s] $f  MISSING"; overall=FAIL; st_fail[$s]=$((${st_fail[$s]:-0}+1)); continue; fi
        ha=$(python3 "$HA" "$fa" | awk '{print $1, $2}')
        hb=$(python3 "$HA" "$fb" | awk '{print $1, $2}')
        if [ "$ha" = "$hb" ]; then
            st_pass[$s]=$((${st_pass[$s]:-0}+1))
        else
            echo "   FAIL  [$s] $f   ($ha vs $hb)"; overall=FAIL; st_fail[$s]=$((${st_fail[$s]:-0}+1))
        fi
    done
done
echo
echo "=== per-stage archive identity ==="
for s in nfsp img clus other; do
    p=${st_pass[$s]:-0}; f=${st_fail[$s]:-0}
    [ $((p+f)) -eq 0 ] && continue
    printf '  %-6s %d/%d identical%s\n' "$s" "$p" "$((p+f))" "$([ $f -gt 0 ] && echo '   <-- FAIL')"
done
echo "=== OVERALL: $overall ==="
[ "$overall" = PASS ]
