#!/bin/bash
# Retirement round 2026-08-03 -- delete the valfast (tier V), doc pr/23-25
# (tier 1) and master-merge/test-round (tier 2) sbnd_xin work-* arms.
# See ../../docs/work-tags.md section "RETIREMENT ROUND 2026-08-03".
#
#   ./retire_20260803.sh V,1,2              # dry run (default action)
#   CONFIRM=yes ./retire_20260803.sh V,1,2  # actually delete
#
# Tier dispositions differ, so the archive guard is tier-aware:
#   tier V  DROP whole, no archive -- except the three arms in
#           state-20260803/plan.json ARCHIVE (no citation anywhere), which are
#           held to the same archive-present rule as tiers 1+2.
#   tier 1  archive record layer first, then delete.
#   tier 2  same as tier 1.
#
# Pre-flights (must both have passed before CONFIRM=yes):
#   scripts/retire/plan_20260803.py         (four asserts, "OVERALL: PASS")
#   scripts/retire/lightcheck_20260803.py   (light/SP coverage, "ALL COVERED")
#   scripts/retire/archive_records_20260803.py  (integrity gate PASS)
set -u
cd "$(dirname "$0")/../.." || exit 1        # -> sbnd_xin
BASE=$PWD
STATE=$BASE/scripts/retire/state-20260803
REC=$BASE/archive/records/pr23-25-era-20260803
TIERS=${1:-V,1,2}
CONFIRM=${CONFIRM:-no}

# The owner's live campaign.  Named here as well as in plan_20260803.py so a
# hand-edited tier file cannot slip one through.
PROTECTED="work-nuecc48-prod0803 work-vfnuecc48-prod0803"

# ---- interlock 1: Bokeh viewers (a deleted tag blanks a live scan) ----------
viewers=$(pgrep -a -f 'bokeh serve' 2>/dev/null)
if [ -n "$viewers" ]; then
    echo "!! a Bokeh viewer is running -- its tags must not be deleted:"
    echo "$viewers" | sed 's/^/     /'
    [ "$CONFIRM" = yes ] && { echo "refusing to delete while a viewer is up"; exit 2; }
fi

# ---- interlock 2 (NEW this round): a live wire-cell / runner batch ----------
# 26 GB of NFS deletes while a campaign competes for IO is exactly M5.  The
# 2026-08-03 round was written while 8 such processes were writing into
# work-nuecc48-prod0803, and a fresh arm (work-pr116962-nocosmicveto) appeared
# mid-round.
#
# Scoped to sbnd_xin: this box is shared, and an unrelated user's `wire-cell`
# must not be able to block our housekeeping (one was running during
# development, out of /nfs/data/1/xning).
#
# ALLOW_LIVE_JOBS=yes overrides.  Legitimate when the live jobs only write into
# PROTECTED/KEEP arms and loadavg is comfortably below ncores -- otherwise an
# owner who works continuously can never run this script at all.
jobs=$(pgrep -a -f 'wire-cell |run_(ql|pr|nusel)_evt' 2>/dev/null \
       | grep -F 'sbnd_xin' | grep -v 'retire_2026')
if [ -n "$jobs" ]; then
    echo "!! an sbnd_xin wire-cell / runner batch is live ($(echo "$jobs" | wc -l) processes):"
    echo "$jobs" | head -5 | cut -c1-120 | sed 's/^/     /'
    echo "     loadavg $(cut -d' ' -f1-3 /proc/loadavg)  ncores $(nproc)"
    if [ "$CONFIRM" = yes ] && [ "${ALLOW_LIVE_JOBS:-no}" != yes ]; then
        echo "refusing to delete while an sbnd_xin batch is running (M5)."
        echo "re-run with ALLOW_LIVE_JOBS=yes if those jobs touch only PROTECTED/KEEP arms."
        exit 2
    fi
fi

# ---- build the list --------------------------------------------------------
list=""
for t in ${TIERS//,/ }; do
    f=$BASE/scripts/retire/tier${t}_20260803.txt
    [ -f "$f" ] || { echo "no such tier list: $f"; exit 1; }
    list="$list $(cat "$f")"
done

# ---- interlock 3: the PROTECTED pair must not appear in any tier file -------
for p in $PROTECTED; do
    for d in $list; do
        [ "$d" = "$p" ] && { echo "!! PROTECTED dir $p is in the tier list -- refusing"; exit 2; }
    done
done

# arms that require an archive before deletion: tiers 1+2, plus the uncited
# tier-V arms.  Read from plan.json so the two scripts cannot drift apart.
need_archive=$(python3 -c "import json;print(' '.join(json.load(open('$STATE/plan.json'))['ARCHIVE']))")

miss=0; n=0; bytes=0
for d in $list; do
    [ -d "$BASE/$d" ] || { echo "SKIP (already gone): $d"; continue; }
    note="tier-V drop (valfast disposal rule)"
    case " $need_archive " in
        *" $d "*)
            tgz=$(find "$REC" -name "$d.tar.gz" -print -quit)
            if [ -z "$tgz" ]; then echo "REFUSE (no archive): $d"; miss=$((miss+1)); continue; fi
            note="archive $(basename "$tgz")" ;;
    esac
    sz=$(du -sm "$BASE/$d" | cut -f1)
    n=$((n+1)); bytes=$((bytes+sz))
    if [ "$CONFIRM" = yes ]; then
        rm -rf "$BASE/$d" && echo "removed  $d  (${sz} MB, $note)"
    else
        echo "would remove  $d  (${sz} MB, $note)"
    fi
done

echo
echo "tiers=$TIERS  dirs=$n  bytes=$((bytes/1024)) GiB  refused=$miss  CONFIRM=$CONFIRM"
[ "$CONFIRM" = yes ] || { echo "dry run only -- re-run with CONFIRM=yes to delete"; exit 0; }

echo
echo "== post-deletion checks =="
python3 "$BASE/relink_tags.py"
# 284 broken links existed before this round, ALL inside three tier-V roots, so
# the target here is exactly 0 -- a nonzero result means something outside the
# plan's model broke.
echo "broken symlinks: $(find "$BASE" -xtype l | wc -l)   (MUST be 0)"
echo "git-deleted tracked files (MUST be empty):"
git -C "$BASE" status --short -- . 2>/dev/null | grep '^ D' || echo "    none"
# expected = KEEP + PROTECTED from plan.json, so this stays right even when work
# in flight adds an auto-protected arm mid-round (it added one on 2026-08-03).
exp=$(python3 -c "import json;p=json.load(open('$STATE/plan.json'));print(len(p['KEEP'])+len(p['PROTECTED']))")
echo "work* dirs remaining: $(ls -d "$BASE"/work* 2>/dev/null | wc -l)   (expect $exp = KEEP + PROTECTED)"
du -sh "$BASE"
df -h /nfs/data/1 | tail -1
