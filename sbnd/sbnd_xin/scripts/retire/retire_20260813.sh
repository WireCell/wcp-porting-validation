#!/bin/bash
# Retirement round 2026-08-13 -- the pr/66-75 campaign sweep, 74 G -> ~20 G.
#
# Fork of retire_20260811.sh. Both of that round's fixes are carried forward
# (the `-type d` survivor census and the real-tier-file Bokeh interlock). Three
# further changes, each a live defect or dead code found during exploration for
# this round -- see docs/work-tags.md "RETIREMENT ROUND 2026-08-13":
#
#   1. THE DRIVER-LOG BLOCK IS DELETED, not carried forward. The 08-11 round
#      tarred and removed all 118 `work-*.driver.log` orphans and none has been
#      recreated, so the block matches nothing -- but it does not fail quietly.
#      With $dlogs empty, `du -cm $dlogs` runs as bare `du -cm .` and reports
#      the size of the ENTIRE 74 GB tree as the driver-log footprint. Verified
#      by reproduction. A block that cannot do useful work but can print a
#      wildly wrong number is worse than no block.
#   2. THE scan-d59k/bee BLOCK IS DELETED. That dir was stripped to 2 MB by the
#      08-11 round; its guard now compares 0 zips to 0 urls and removes nothing.
#   3. INTERLOCK 2 SELF-TRIPS. `pgrep -f 'wire-cell |run_(ql|pr|nusel)_evt'`
#      matches any shell whose command line merely CONTAINS the pattern --
#      including the exploration shells an agent runs while preparing the round,
#      and including pgrep's own process. Reproduced: 2 phantom matches with no
#      wire-cell job running anywhere. The old `grep -v 'retire_2026'` was too
#      narrow. Widened below. This matters because the documented workaround
#      (ALLOW_LIVE_JOBS=yes) defeats the real M5 check entirely -- a false
#      positive here trains the operator to disable the interlock.
#
# NO EXTRA REMOVAL CLASSES this round: verified that no non-`work`-prefixed
# removal-candidate dir exists (the full non-work top-level list is archive bee
# docs dqdx_rr_sample input_files_reco1 nusel_display overclustering_display
# overclustering_labels pics pmt_nonlin_out pr_display products ql_scan
# sbnd_geometry scan-* scripts showcase-* stm_campaign valfast vertex_labels,
# all legitimate) and that 0 orphan work-*.driver.log files remain.
#
# NO PHASE 4 HUB THINNING. thin_hubs_20260811.py must NOT be re-run: the five
# work-*-cb0805 hubs are the prod0813 campaign's INPUT this round, not a record
# layer. See plan_20260813.py's docstring.
#
#   ./retire_20260813.sh A              # dry run (default action)
#   CONFIRM=yes ./retire_20260813.sh A  # actually delete
#
# Tier dispositions: tier A archives (already done via
# archive_records_20260813.py) then deletes -- all 388 dirs in this round.
#
# Pre-flights (all must have passed before CONFIRM=yes):
#   scripts/retire/plan_20260813.py             (6 asserts, "OVERALL: PASS")
#   scripts/retire/archive_records_20260813.py  (integrity gate PASS 388/388)
#   4. $BASE WAS THE SYMLINK PATH, MAKING THREE CHECKS VACUOUS. Every prior
#      round did `cd "$(dirname "$0")/../.." ; BASE=$PWD`. Invoked through
#      toolkit/sbnd_xin -- which is a SYMLINK to wcp-porting-img/sbnd/sbnd_xin,
#      the normal way to reach this tree -- $PWD is the logical path, so $BASE
#      names a symlink. `find "$BASE" ...` does not descend a symlink argument
#      and `du -sh "$BASE"` measures the link itself. Consequences, all observed
#      live in this round's own execution log:
#        - interlock 0 (`find "$BASE" -xtype l`) reported 0 broken symlinks
#          because it scanned NOTHING. It would have reported 0 with the tree on
#          fire. This is the round's only pre-deletion safety check.
#        - the post-run survivor census printed "work* DIRS remaining: 0
#          (expect 13)" after a completely successful round.
#        - `du -sh` wrote "0" into removed.tsv's footer as the post-round size.
#      `rm -rf "$BASE/$d"` was unaffected -- only the final path component
#      matters there -- so the deletion itself was correct; it was the
#      verification that was blind. Fixed with `cd -P`.
set -u
cd -P "$(dirname "$0")/../.." || exit 1     # -P: resolve the symlink, see defect 4
BASE=$PWD
STATE=$BASE/scripts/retire/state-20260813
REC=$BASE/archive/records/pr66-75-era-20260813
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
    f=$BASE/scripts/retire/tier${t}_20260813.txt
    [ -f "$f" ] || { echo "no such tier list: $f"; exit 1; }
    tierfiles="$tierfiles $f"
done
list=$(cat $tierfiles)

# ---- interlock 1: Bokeh viewers --------------------------------------------
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
# FIXED (defect 3): drop the agent/exploration shells and pgrep itself, which
# match the pattern by quoting it rather than by running a job.
jobs=$(pgrep -a -f 'wire-cell |run_(ql|pr|nusel)_evt' 2>/dev/null \
       | grep -F 'sbnd_xin' \
       | grep -v 'retire_2026' \
       | grep -vE 'snapshot-bash|/claude|[[:space:]]claude([[:space:]]|$)|pgrep|grep -')
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
      echo "# retire_20260813.sh  tiers=$TIERS"
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
