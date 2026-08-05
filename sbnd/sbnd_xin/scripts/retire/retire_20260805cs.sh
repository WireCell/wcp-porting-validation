#!/bin/bash
# Retirement round 2026-08-05 CLEAN SLATE -- clear the deck for reprocessing.
#
# PREREQUISITE, NOT OPTIONAL: scripts/retire/preserve_20260805cs.sh must have
# run with CONFIRM=yes and printed "PRESERVE OK".  It discharges the two asserts
# that legitimately blocked this round -- 3 hand-scan label dirs copied verbatim
# to archive/records/labels/, and 551 archive/ symlinks (351 dir + 200 file)
# into work-mcp10/work-mcp1000 materialized as hardlinks so archive/ becomes
# self-contained.  Without it plan_20260805cs.py FAILS asserts 2 and 4.
#
# The KEEP set is 5 explicit names and is NOT derived from the dependency graph:
# with imaging regenerated, run_img_evt.sh reads input_files_reco1/ and NOT ONE
# work-* dir is an input to the campaign.  See plan_20260805cs.py's docstring.
# See ../../docs/work-tags.md section "RETIREMENT ROUND 2026-08-05 CLEAN SLATE" and
# ../../docs/pr/37_cross-audit-residual-round.md sec.13.
#
#   ./retire_20260805cs.sh A,D              # dry run (default action)
#   CONFIRM=yes ./retire_20260805cs.sh A,D  # actually delete
#
# Tier dispositions:
#   tier A  archive record layer first, then delete   (27 dirs, ALL of them)
#   tier D  unused this round
#
# Pre-flights (all three must have passed before CONFIRM=yes):
#   scripts/retire/plan_20260805.py            (seven asserts, "OVERALL: PASS")
#   scripts/retire/lightcheck_20260805.py      (light/SP coverage, "ALL COVERED")
#   scripts/retire/archive_records_20260805.py (integrity gate PASS)
#
# FORKED FROM retire_20260803.sh.  Three differences, each fixing a real defect:
#
#   a. PROTECTED IS READ FROM PROTECTED.txt, not hardcoded.  The 08-03 driver
#      hardcoded two names while PROTECTED.txt's header claimed to be the
#      carry-forward registry that "a new round's driver must union" -- no
#      script had ever read it.  Field 1 is word-split because three of its
#      lines pack four arm names into that field.
#   b. NEW INTERLOCK 0: the broken-symlink count must be 0 BEFORE the round.
#      The 08-03 script only checks after (:114), and by then relink_tags.py
#      cannot repair a link whose target is gone -- it is a tripwire, not an
#      interlock.  docs/work-tags.md:14-16 says measure it before ("it was 284
#      going into 2026-08-03 and nobody had noticed").
#   c. NEW: a post-hoc removal manifest, state-20260805cs/removed.tsv.  Three
#      rounds have deleted 484 directories and the only surviving record is the
#      PRE-execution tier*.txt intent files.  This is the first round to record
#      what actually happened.
set -u
cd "$(dirname "$0")/../.." || exit 1        # -> sbnd_xin
BASE=$PWD
STATE=$BASE/scripts/retire/state-20260805cs
REC=$BASE/archive/records/clean-slate-20260805
TIERS=${1:-A}
CONFIRM=${CONFIRM:-no}

# ---- (a) PROTECTED = inline union PROTECTED.txt field 1, word-split ---------
# The 12 vf37 arms were released by the owner for this round and are in tier D;
# plan_20260805.py subtracts them from PROTECTED, and interlock 3 below reads
# the same effective set out of plan.json so the two cannot drift.
PROTECTED=$(python3 -c "
import json;print(' '.join(json.load(open('$STATE/plan.json'))['PROTECTED']))")
HUBS=$(python3 -c "
import json;print(' '.join(json.load(open('$STATE/plan.json'))['HUB']))")
POSTBUILD=$(python3 -c "
import json;print(' '.join(json.load(open('$STATE/plan.json'))['POSTBUILD']))")

# ---- interlock 0 (NEW): broken symlinks BEFORE the round --------------------
pre_broken=$(find "$BASE" -xtype l | wc -l)
echo "broken symlinks before the round: $pre_broken   (MUST be 0)"
if [ "$pre_broken" -ne 0 ]; then
    echo "!! the tree already has dangling links -- fix them before deleting anything,"
    echo "   or the post-round check cannot distinguish new breakage from old."
    find "$BASE" -xtype l | head -20 | sed 's/^/     /'
    [ "$CONFIRM" = yes ] && exit 2
fi

# ---- interlock 1: Bokeh viewers (a deleted tag blanks a live scan) ----------
# The 08-03 version refused unconditionally whenever any viewer was up.  That is
# both too strict and too loose: an owner who keeps a viewer open can never run
# the script, while a viewer on an unrelated tag was never the hazard.  What
# matters is whether the viewer's command line names a dir in THIS removal set.
# Checked explicitly; the blanket refusal is kept for the case where it does.
viewers=$(pgrep -a -f 'bokeh serve' 2>/dev/null)
if [ -n "$viewers" ]; then
    echo "!! a Bokeh viewer is running:"
    echo "$viewers" | cut -c1-160 | sed 's/^/     /'
    vhit=""
    for d in $(cat "$BASE"/scripts/retire/tier{A}_20260805cs.txt); do
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
        echo "re-run with ALLOW_LIVE_JOBS=yes if those jobs touch only PROTECTED/KEEP arms."
        exit 2
    fi
fi

# ---- build the list --------------------------------------------------------
list=""
for t in ${TIERS//,/ }; do
    f=$BASE/scripts/retire/tier${t}_20260805cs.txt
    [ -f "$f" ] || { echo "no such tier list: $f"; exit 1; }
    list="$list $(cat "$f")"
done

# ---- interlock 3: no PROTECTED / HUB / POSTBUILD dir in any tier file -------
# Widened from the 08-03 version, which only checked the two PROTECTED names.
# POSTBUILD is the clean-source reference family (doc pr/33 sec.11.2) -- losing
# it would leave the next campaign with nothing to be gated against, and it is
# exactly the class a citation-driven rule would have swept.
for p in $PROTECTED $HUBS $POSTBUILD; do
    for d in $list; do
        [ "$d" = "$p" ] && { echo "!! survivor $p is in the tier list -- refusing"; exit 2; }
    done
done

# arms that require an archive before deletion: tier A.  Read from plan.json so
# the plan script and this driver cannot drift apart.
need_archive=$(python3 -c "
import json;print(' '.join(json.load(open('$STATE/plan.json'))['ARCHIVE']))")

# ---- (c) the removal manifest ----------------------------------------------
MAN=$STATE/removed.tsv
if [ "$CONFIRM" = yes ]; then
    {
      echo "# retire_20260805cs.sh  tiers=$TIERS"
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
        cites=$(python3 -c "
import json;print(json.load(open('$STATE/plan.json'))['cites'].get('$d',0))")
        if rm -rf "$BASE/$d"; then
            echo "removed  $d  (${sz} MB, $note)"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                   "$(date -Is)" "$d" "$tier" "$sz" "$tgzname" "$mt" "$cites" >> "$MAN"
        else
            echo "!! rm FAILED: $d"; miss=$((miss+1))
        fi
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
post_broken=$(find "$BASE" -xtype l | wc -l)
echo "broken symlinks: $post_broken   (MUST be 0; was $pre_broken before)"
echo "git-deleted tracked files (MUST be empty):"
git -C "$BASE" status --short -- . 2>/dev/null | grep '^ D' || echo "    none"
# survivors = HUB + POSTBUILD + PROTECTED.  The 08-03 script summed KEEP +
# PROTECTED; this round has three survivor classes and plan.json's KEEP is
# HUB+POSTBUILD, so spell the sum out rather than reusing that key blindly.
exp=$(python3 -c "
import json;p=json.load(open('$STATE/plan.json'))
print(len(p['HUB'])+len(p['POSTBUILD'])+len(p['PROTECTED']))")
echo "work* dirs remaining: $(ls -d "$BASE"/work* 2>/dev/null | wc -l)   (expect $exp = HUB + POSTBUILD + PROTECTED)"
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
