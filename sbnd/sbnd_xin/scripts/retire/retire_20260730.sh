#!/bin/bash
# Retirement round 2026-07-30 -- delete the retiring sbnd_xin work-* arms whose
# record layer is already in archive/records/ (see ../../docs/work-tags.md and
# ../../archive/records/README.md).
#
#   ./retire_20260730.sh 1            # dry run, tier 1 (125 dirs, 25.1 GiB)
#   ./retire_20260730.sh 1,2          # dry run, tiers 1+2 (134 dirs, 35.9 GiB)
#   CONFIRM=yes ./retire_20260730.sh 1        # actually delete
#
# Refuses to delete an arm whose <tag>.tar.gz is missing from archive/records/,
# so an un-archived directory can never be lost by running this.
set -u
cd "$(dirname "$0")/../.." || exit 1        # -> sbnd_xin
BASE=$PWD
REC=$BASE/archive/records
TIERS=${1:-1}
CONFIRM=${CONFIRM:-no}

# A running scan viewer pins its tag and its --prev baselines on the command
# line; deleting one of those blanks the live scan.  The check that mattered on
# 2026-07-30 (nothing running) is a point-in-time fact, so re-do it here.
viewers=$(pgrep -a -f 'bokeh serve' 2>/dev/null)
if [ -n "$viewers" ]; then
    echo "!! a Bokeh viewer is running -- its tags must not be deleted:"
    echo "$viewers" | sed 's/^/     /'
    echo "!! stop it (or confirm none of its tags are in the list) before CONFIRM=yes"
    [ "${CONFIRM:-no}" = yes ] && { echo "refusing to delete while a viewer is up"; exit 2; }
fi

list=""
for t in ${TIERS//,/ }; do
    f=$BASE/scripts/retire/tier$t.txt
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
du -sh "$BASE"
