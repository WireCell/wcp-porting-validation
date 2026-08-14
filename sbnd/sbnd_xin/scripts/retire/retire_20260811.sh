#!/bin/bash
# Retirement round 2026-08-11 -- the pr/38-65 campaign sweep, 103 G -> ~24-26 G.
#
# Fork of retire_20260805cs.sh. Two defects fixed, both found live during
# exploration for this round (see docs/work-tags.md and plan_20260811.py
# docstring for the full account):
#
#   1. SURVIVOR CENSUS BUG. Both retire_20260805.sh:188 and
#      retire_20260805cs.sh:195 count `ls -d work* | wc -l`, which in THIS tree
#      also counts 118 orphan `work-*.driver.log` FILES (records of arms
#      deleted by earlier rounds) alongside directories -- the count can never
#      match the expected survivor count. Fixed below with `find -maxdepth 1
#      -type d`.
#   2. BRACE-EXPANSION BUG. retire_20260805cs.sh:85 iterates
#      `tier{A}_20260805cs.txt` -- bash leaves a single-element brace literal,
#      so the `cat` silently matches nothing, `$vhit` stays empty, and the
#      Bokeh interlock unconditionally prints "safe to proceed" even when a
#      live viewer names a removal-set directory. Fixed below by iterating the
#      actual $TIERS-derived file list, same as the removal loop does.
#
# THREE EXTRA REMOVAL CLASSES this round, invisible to the `work*`-glob
# machinery every prior round used (handled as separate blocks below, not by
# the tier-file loop):
#   - VOID-pr32-round1/       included in plan_20260811.py's universe directly
#                              (a real dir, not `work`-prefixed) -- flows
#                              through the normal tier-A archive-then-delete
#                              path, tag name "VOID-pr32-round1".
#   - work-*.driver.log       118 orphan files, 0.12 GiB. Tarred as one bundle
#                              (scripts/retire/state-20260811/driver-logs.tar.gz)
#                              then removed -- not directories, so plan.json's
#                              per-dir walk never sees them.
#   - scan-d59k/bee/          694 MB, 9 Bee zips. Dropped WITHOUT archiving:
#                              all 9 already have a saved `.url` (checked --
#                              unlike the 08-03 round, which found only 12/44),
#                              so the sets are still live on the BNL twister
#                              server. Its record layer (.url/.index.txt/
#                              .stmid-map.txt/.build.log) and the hand-scan
#                              .tsv/.txt tables (already copied to
#                              archive/records/labels/scan-d59k/ in Phase 2)
#                              are untouched.
#
#   ./retire_20260811.sh A              # dry run (default action)
#   CONFIRM=yes ./retire_20260811.sh A  # actually delete
#
# Tier dispositions: tier A archives (already done via
# archive_records_20260811.py) then deletes -- all 454 dirs in this round.
#
# Pre-flights (all must have passed before CONFIRM=yes):
#   scripts/retire/plan_20260811.py             (6 asserts, "OVERALL: PASS")
#   scripts/retire/archive_records_20260811.py  (integrity gate PASS 454/454)
#   Phase 2 (this round's doc): overclustering_labels archived + git-committed
set -u
cd "$(dirname "$0")/../.." || exit 1        # -> sbnd_xin
BASE=$PWD
STATE=$BASE/scripts/retire/state-20260811
REC=$BASE/archive/records/pr38-65-era-20260811
TIERS=${1:-A}
CONFIRM=${CONFIRM:-no}

PROTECTED=$(python3 -c "
import json;print(' '.join(json.load(open('$STATE/plan.json'))['PROTECTED']))")
KEEP=$(python3 -c "
import json;print(' '.join(json.load(open('$STATE/plan.json'))['KEEP']))")

# ---- interlock 0: broken symlinks BEFORE the round --------------------------
pre_broken=$(find "$BASE" -xtype l | wc -l)
echo "broken symlinks before the round: $pre_broken   (MUST be 0)"
if [ "$pre_broken" -ne 0 ]; then
    echo "!! the tree already has dangling links -- fix them before deleting anything."
    find "$BASE" -xtype l | head -20 | sed 's/^/     /'
    [ "$CONFIRM" = yes ] && exit 2
fi

# ---- build the tier-list file set (used by both interlock 1 and the loop) --
tierfiles=""
for t in ${TIERS//,/ }; do
    f=$BASE/scripts/retire/tier${t}_20260811.txt
    [ -f "$f" ] || { echo "no such tier list: $f"; exit 1; }
    tierfiles="$tierfiles $f"
done
list=$(cat $tierfiles)

# ---- interlock 1: Bokeh viewers -- FIXED (defect 2): iterate real tier files, not a
#      literal brace pattern -----------------------------------------------
viewers=$(pgrep -a -f 'bokeh serve' 2>/dev/null)
if [ -n "$viewers" ]; then
    echo "!! a Bokeh viewer is running:"
    echo "$viewers" | cut -c1-160 | sed 's/^/     /'
    vhit=""
    for d in $list; do
        case "$viewers" in *"$d"*) vhit="$vhit $d";; esac
    done
    if [ -n "$vhit" ]; then
        echo "   viewer references REMOVAL CANDIDATES:$vhit"
        [ "$CONFIRM" = yes ] && { echo "refusing -- stop the viewer first"; exit 2; }
    else
        echo "   viewer references no dir in the removal set -- safe to proceed."
    fi
fi

# ---- interlock 2: a live wire-cell / runner batch (M5) ---------------------
jobs=$(pgrep -a -f 'wire-cell |run_(ql|pr|nusel)_evt' 2>/dev/null \
       | grep -F 'sbnd_xin' | grep -v 'retire_2026')
if [ -n "$jobs" ]; then
    echo "!! an sbnd_xin wire-cell / runner batch is live ($(echo "$jobs" | wc -l) processes):"
    echo "$jobs" | head -5 | cut -c1-120 | sed 's/^/     /'
    echo "     loadavg $(cut -d' ' -f1-3 /proc/loadavg)  ncores $(nproc)"
    if [ "$CONFIRM" = yes ] && [ "${ALLOW_LIVE_JOBS:-no}" != yes ]; then
        echo "refusing to delete while an sbnd_xin batch is running (M5)."
        exit 2
    fi
fi

# ---- interlock 3: no KEEP / PROTECTED dir in any tier list ------------------
for p in $KEEP $PROTECTED; do
    for d in $list; do
        [ "$d" = "$p" ] && { echo "!! survivor $p is in the tier list -- refusing"; exit 2; }
    done
done

need_archive=$(python3 -c "
import json;print(' '.join(json.load(open('$STATE/plan.json'))['ARCHIVE']))")

MAN=$STATE/removed.tsv
if [ "$CONFIRM" = yes ]; then
    {
      echo "# retire_20260811.sh  tiers=$TIERS"
      echo "# started              $(date -Is)"
      echo "# wcp-porting-img HEAD $(git -C "$BASE" rev-parse --short HEAD)"
      echo "# toolkit HEAD         $(git -C /nfs/data/1/xqian/toolkit-dev/toolkit rev-parse --short HEAD)"
      echo "# broken symlinks pre  $pre_broken"
      echo "# du -sh sbnd_xin pre  $(du -sh "$BASE" | cut -f1)"
      echo "# df /nfs/data/1 pre   $(df -h /nfs/data/1 | tail -1 | awk '{print $4" avail"}')"
      printf 'iso_ts\tdir\ttier\tMB\tarchive_tarball\tdir_mtime\tcitations\n'
    } > "$MAN"
fi

miss=0; n=0; bytes=0
for d in $list; do
    [ -d "$BASE/$d" ] || { echo "SKIP (already gone): $d"; continue; }
    tier=D; note="tier-D drop (no archive)"; tgzname="-"
    case " $need_archive " in
        *" $d "*)
            tier=A
            tgz=$(find "$REC" -name "$d.tar.gz" -print -quit 2>/dev/null)
            if [ -z "$tgz" ]; then echo "REFUSE (no archive): $d"; miss=$((miss+1)); continue; fi
            tgzname=$(basename "$tgz"); note="archive $tgzname" ;;
    esac
    sz=$(du -sm "$BASE/$d" | cut -f1)
    n=$((n+1)); bytes=$((bytes+sz))
    if [ "$CONFIRM" = yes ]; then
        mt=$(date -Is -r "$BASE/$d")
        if rm -rf "$BASE/$d"; then
            echo "removed  $d  (${sz} MB, $note)"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                   "$(date -Is)" "$d" "$tier" "$sz" "$tgzname" "$mt" "0" >> "$MAN"
        else
            echo "!! rm FAILED: $d"; miss=$((miss+1))
        fi
    else
        echo "would remove  $d  (${sz} MB, $note)"
    fi
done

echo
echo "tiers=$TIERS  dirs=$n  bytes=$((bytes/1024)) GiB  refused=$miss  CONFIRM=$CONFIRM"

# ---- EXTRA: driver-log bundle + scan-d59k/bee (dry-run reported always) ----
dlogs=$(find "$BASE" -maxdepth 1 -name 'work-*.driver.log' -type f)
ndlogs=$(echo "$dlogs" | grep -c . || true)
dlog_mb=$(du -cm $dlogs 2>/dev/null | tail -1 | cut -f1)
echo
echo "EXTRA: $ndlogs work-*.driver.log files (${dlog_mb:-0} MB) -> "
echo "       tar to $STATE/driver-logs.tar.gz then remove"
if [ "$CONFIRM" = yes ] && [ -n "$dlogs" ]; then
    tar czf "$STATE/driver-logs.tar.gz" -C "$BASE" $(echo "$dlogs" | xargs -n1 basename)
    rm -f $dlogs
    echo "  done: $STATE/driver-logs.tar.gz ($(du -sh "$STATE/driver-logs.tar.gz" | cut -f1))"
fi

nzips=$(ls "$BASE/scan-d59k/bee"/*.zip 2>/dev/null | wc -l)
nurls=$(ls "$BASE/scan-d59k/bee"/*.url 2>/dev/null | wc -l)
zip_mb=$(du -sm "$BASE/scan-d59k/bee" 2>/dev/null | cut -f1)
echo
echo "EXTRA: scan-d59k/bee/ -- $nzips zips, $nurls saved .url (${zip_mb:-0} MB) -> drop, no archive"
if [ "$nzips" -ne "$nurls" ]; then
    echo "  !! zip/url count mismatch -- refusing to drop, check by hand"
elif [ "$CONFIRM" = yes ]; then
    rm -f "$BASE"/scan-d59k/bee/*.zip
    echo "  done: $nzips zip(s) removed, record files (.url/.index.txt/.stmid-map.txt/.build.log) kept"
fi

[ "$CONFIRM" = yes ] || { echo; echo "dry run only -- re-run with CONFIRM=yes to delete"; exit 0; }

echo
echo "== post-deletion checks =="
python3 "$BASE/relink_tags.py"
post_broken=$(find "$BASE" -xtype l | wc -l)
echo "broken symlinks: $post_broken   (MUST be 0; was $pre_broken before)"
echo "git-deleted tracked files (MUST be empty):"
git -C "$BASE" status --short -- . 2>/dev/null | grep '^ D' || echo "    none"
exp=$(python3 -c "
import json;p=json.load(open('$STATE/plan.json'))
print(len(p['KEEP']))")
# FIXED (defect 1): -type d, not a bare glob that also matches the driver-log files.
survivors=$(find "$BASE" -maxdepth 1 -name 'work*' -type d | wc -l)
echo "work* DIRS remaining: $survivors   (expect $exp = KEEP)"
echo "removal manifest rows: $(grep -vc '^#\|^iso_ts' "$MAN")   (expect $n)"
{
  echo "# finished             $(date -Is)"
  echo "# broken symlinks post $post_broken"
  echo "# du -sh sbnd_xin post $(du -sh "$BASE" | cut -f1)"
  echo "# df /nfs/data/1 post  $(df -h /nfs/data/1 | tail -1 | awk '{print $4" avail"}')"
} >> "$MAN"
du -sh "$BASE"
df -h /nfs/data/1 | tail -1
echo "manifest: $MAN"
