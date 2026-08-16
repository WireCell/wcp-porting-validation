#!/bin/bash
# Retirement round 2026-08-16 -- the pr/79-86 campaign sweep, 158 G -> ~28 G.
#
# Fork of retire_20260813.sh. All of that round's fixes are carried forward
# (the `-type d` survivor census, the real-tier-file Bokeh interlock, the
# driver-log and scan-d59k/bee block deletions, the widened interlock-2 grep,
# and the `cd -P` symlink fix). One change this round, requested by the
# approved plan:
#
#   INTERLOCKS 0/1/2 NOW EXIT IN DRY RUN TOO, not only under CONFIRM=yes.
#   08-13's script only made these interlocks blocking when CONFIRM=yes was
#   set -- a dry run would print the warning and continue regardless, so
#   "read the dry run" could not by itself catch a live-viewer or live-job
#   conflict. This round's dry run is the ONLY exercise `cd -P` has had before
#   a real CONFIRM=yes deletion (08-13's fix was never actually run under
#   CONFIRM=yes -- see plan_20260816.py's docstring), so the dry run must be
#   trustworthy on its own.
#
# Interlock 3 (no KEEP/PROTECTED dir in the tier list) was already
# unconditional in 08-13 and is unchanged.
#
#   ./retire_20260816.sh A              # dry run (default action)
#   CONFIRM=yes ./retire_20260816.sh A  # actually delete
#
# Tier dispositions: tier A archives (already done via
# archive_records_20260816.py) then deletes -- all removal-set dirs this round.
#
# Pre-flights (all must have passed before CONFIRM=yes):
#   scripts/retire/plan_20260816.py             (7 asserts, "OVERALL: PASS")
#   scripts/retire/archive_records_20260816.py  (integrity gate PASS n/n)
#
# THE cd -P FIX (inherited from 08-13, carried verbatim): every round before
# 08-13 did `cd "$(dirname "$0")/../.." ; BASE=$PWD`. Invoked through
# toolkit/sbnd_xin -- a SYMLINK to wcp-porting-img/sbnd/sbnd_xin, the normal
# way to reach this tree -- $PWD was the logical path, so $BASE named a
# symlink. `find "$BASE" ...` does not descend a symlink argument and
# `du -sh "$BASE"` measures the link itself, which made interlock 0 and the
# post-round survivor census vacuous in the 08-13 round. `rm -rf "$BASE/$d"`
# was unaffected -- only the final path component matters there. Fixed with
# `cd -P`. This round adds an explicit BASE echo right after it so the fix is
# never trusted silently again.
set -u
cd -P "$(dirname "$0")/../.." || exit 1     # -P: resolve the symlink
BASE=$PWD
echo "BASE=$BASE"
case "$BASE" in
    */toolkit/sbnd_xin*) echo "!! BASE is still the symlink path -- cd -P failed"; exit 1 ;;
esac
STATE=$BASE/scripts/retire/state-20260816
REC=$BASE/archive/records/pr79-86-era-20260816
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
    exit 2
fi

# ---- build the tier-list file set (used by both interlock 1 and the loop) --
tierfiles=""
for t in ${TIERS//,/ }; do
    f=$BASE/scripts/retire/tier${t}_20260816.txt
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
        echo "refusing -- stop the viewer first"; exit 2
    else
        echo "   viewer references no dir in the removal set -- safe to proceed."
    fi
fi

# ---- interlock 2: a live wire-cell / runner batch (M5) ---------------------
jobs=$(pgrep -a -f 'wire-cell |run_(ql|pr|nusel)_evt' 2>/dev/null \
       | grep -F 'sbnd_xin' \
       | grep -v 'retire_2026' \
       | grep -vE 'snapshot-bash|/claude|[[:space:]]claude([[:space:]]|$)|pgrep|grep -')
if [ -n "$jobs" ]; then
    echo "!! an sbnd_xin wire-cell / runner batch is live ($(echo "$jobs" | wc -l) processes):"
    echo "$jobs" | head -5 | cut -c1-120 | sed 's/^/     /'
    echo "     loadavg $(cut -d' ' -f1-3 /proc/loadavg)  ncores $(nproc)"
    if [ "${ALLOW_LIVE_JOBS:-no}" != yes ]; then
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
      echo "# retire_20260816.sh  tiers=$TIERS"
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
