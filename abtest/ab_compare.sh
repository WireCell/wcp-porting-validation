#!/bin/bash
# Compare two snapshots made by run_events.sh: per-archive content hashes
# (timestamp-insensitive, via hash_archive.py) + wall/RSS deltas.
#
# Usage: ./ab_compare.sh <labelA> <labelB>

set -u
AB_DIR=$(cd "$(dirname "$0")" && pwd)
A="$AB_DIR/snap/${1:?usage: ab_compare.sh <labelA> <labelB>}"
B="$AB_DIR/snap/${2:?usage: ab_compare.sh <labelA> <labelB>}"

[ -d "$A" ] || { echo "missing snapshot dir $A" >&2; exit 2; }
[ -d "$B" ] || { echo "missing snapshot dir $B" >&2; exit 2; }

overall=PASS
for evdirA in "$A"/*/; do
    ev=$(basename "$evdirA")
    evdirB="$B/$ev"
    if [ ! -d "$evdirB" ]; then
        echo "[$ev] MISSING in $2"
        overall=FAIL
        continue
    fi
    echo "== $ev"
    # metrics
    for stage in img clus; do
        mA="$evdirA/${stage}_meta.txt"; mB="$evdirB/${stage}_meta.txt"
        if [ -f "$mA" ] && [ -f "$mB" ]; then
            wA=$(awk -F= '/^wall_s/{print $2}' "$mA"); wB=$(awk -F= '/^wall_s/{print $2}' "$mB")
            rA=$(awk -F= '/^maxrss_kb/{print int($2/1024)}' "$mA"); rB=$(awk -F= '/^maxrss_kb/{print int($2/1024)}' "$mB")
            echo "   $stage: wall ${wA}s -> ${wB}s   rss ${rA}MB -> ${rB}MB"
        fi
    done
    # archives
    archA=$(ls "$evdirA"/*.tar.gz "$evdirA"/*.zip 2>/dev/null | xargs -r -n1 basename | sort)
    archB=$(ls "$evdirB"/*.tar.gz "$evdirB"/*.zip 2>/dev/null | xargs -r -n1 basename | sort)
    if [ "$archA" != "$archB" ]; then
        echo "   FILELIST DIFFERS:"
        diff <(echo "$archA") <(echo "$archB") | sed 's/^/     /'
        overall=FAIL
    fi
    for f in $archA; do
        [ -f "$evdirB/$f" ] || continue
        hA=$(python3 "$AB_DIR/hash_archive.py" "$evdirA/$f" | awk '{print $1, $2}')
        hB=$(python3 "$AB_DIR/hash_archive.py" "$evdirB/$f" | awk '{print $1, $2}')
        if [ "$hA" = "$hB" ]; then
            echo "   PASS  $f"
        else
            echo "   FAIL  $f   ($hA vs $hB)"
            overall=FAIL
        fi
    done
done

echo "=== OVERALL: $overall ==="
[ "$overall" = "PASS" ]
