#!/bin/bash
# Retirement round 2026-08-02 -- delete the pr/11..pr/22-era sbnd_xin work-*
# arms whose record layer is in archive/records/pr-era-20260802/ (see
# ../../docs/work-tags.md section "RETIREMENT ROUND 2026-08-02").
#
#   ./retire_20260802.sh 1            # dry run, tier 1 (the full removal set)
#   CONFIRM=yes ./retire_20260802.sh 1        # actually delete
#
# Same guards as retire_20260730.sh: refuses any arm whose <tag>.tar.gz is not
# in archive/records/, refuses to delete while a Bokeh viewer is running,
# dry-run by default, and runs relink + broken-link + du checks afterwards.
# Pre-flights for THIS round (must both have passed before CONFIRM=yes):
#   scripts/retire/materialize_20260802.sh   (pr/22 exhibit chain self-contained)
#   scripts/retire/lightcheck_20260802.py    (SP+light coverage / exceptions)
set -u
cd "$(dirname "$0")/../.." || exit 1        # -> sbnd_xin
BASE=$PWD
REC=$BASE/archive/records
TIERS=${1:-1}
CONFIRM=${CONFIRM:-no}

viewers=$(pgrep -a -f 'bokeh serve' 2>/dev/null)
if [ -n "$viewers" ]; then
    echo "!! a Bokeh viewer is running -- its tags must not be deleted:"
    echo "$viewers" | sed 's/^/     /'
    echo "!! stop it (or confirm none of its tags are in the list) before CONFIRM=yes"
    [ "${CONFIRM:-no}" = yes ] && { echo "refusing to delete while a viewer is up"; exit 2; }
fi

list=""
for t in ${TIERS//,/ }; do
    f=$BASE/scripts/retire/tier${t}_20260802.txt
    [ -f "$f" ] || { echo "no such tier list: $f"; exit 1; }
    list="$list $(cat "$f")"
done

miss=0; n=0; bytes=0
for d in $list; do
    [ -d "$BASE/$d" ] || { echo "SKIP (already gone): $d"; continue; }
    tgz=$(find "$REC" -name "$d.tar.gz" -print -quit)
    if [ -z "$tgz" ]; then echo "REFUSE (no archive): $d"; miss=$((miss+1)); continue; fi
    sz=$(du -sm "$BASE/$d" | cut -f1)
    n=$((n+1)); bytes=$((bytes+sz))
    if [ "$CONFIRM" = yes ]; then
        rm -rf "$BASE/$d" && echo "removed  $d  (${sz} MB, archive $(basename "$tgz"))"
    else
        echo "would remove  $d  (${sz} MB, archive $(basename "$tgz"))"
    fi
done

echo
echo "tiers=$TIERS  dirs=$n  bytes=$((bytes/1024)) GiB  refused=$miss  CONFIRM=$CONFIRM"
[ "$CONFIRM" = yes ] || { echo "dry run only -- re-run with CONFIRM=yes to delete"; exit 0; }

echo
echo "== post-deletion checks =="
python3 "$BASE/relink_tags.py"
echo "broken symlinks: $(find "$BASE" -xtype l | wc -l)   (MUST be 0)"
echo "git-deleted tracked files (MUST be empty):"
git -C "$BASE" status --short -- . 2>/dev/null | grep '^ D' || echo "    none"
du -sh "$BASE"
