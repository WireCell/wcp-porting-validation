#!/bin/bash
# Retirement round 2026-08-29 -- eleven CLOSED doc families released, the pi0
# epoch and the open pr/130 round kept.  432 arms removed (~81 G), 120 kept.
#
# Fork of retire_20260825b.sh.  Every earlier fix is carried verbatim: the
# `cd -P` symlink fix, the `-type d` survivor census, the real-tier-file Bokeh
# interlock, the widened interlock-2 grep, interlocks 0/1/2 exiting in dry run
# too, and 08-17's interlock-0 refinement.
#
# Owner scope, given directly (2026-08-29): "the sbnd_xin directory now grow to
# 152 G, we should plan for a clean up round again before going to the next
# step.  We can keep the latest PR results. ... our next major move is to
# improve the pi0 reconstruction.  We need to use the hand scan results for pi0
# to do this. ... we also do not want to delete the files related to the other
# running session."  Asked about the one discretionary block, the doc-84
# long-muon/MCS product layer (work-d84r3-cens-* 14.2 G + work-mcp1k-mcs80on
# 3.0 G), the owner chose "Release all three".
#
# TWO INTERLOCKS CHANGE THIS ROUND, BOTH BECAUSE THE TREE IS NOT QUIET.
#
# INTERLOCK 2 IS REFINED, NOT BYPASSED.  Every prior round ran against a quiet
# tree and could afford a blanket refusal on any live sbnd_xin wire-cell
# process (M5).  Today a peer session owns an OPEN round -- doc pr/130, a live
# 239-event knob-off gate -- and its jobs come and go for hours.  A blanket
# refusal here does not make the round safer; it makes it impossible, and the
# predictable next step is ALLOW_LIVE_JOBS=yes, which disarms the interlock
# COMPLETELY and would have let a genuinely dangerous round through.  So the
# refusal is narrowed to the thing that is actually unsafe: a live process that
# names a dir in the REMOVAL set.  That is the same shape interlock 1 has
# always used for Bokeh viewers.  A live job on a KEEP dir now proceeds, and is
# printed so the operator sees it.
#
# INTERLOCK 6 IS NEW: PLAN-TIME EVIDENCE EXPIRES.  plan_20260829.py ASSERT 12
# checked the live-process and mtime conditions when the plan ran; between then
# and CONFIRM=yes the peer can start a job or touch an arm.  So both are
# RE-DERIVED here, immediately before the first rm.  The mtime half is the
# stronger of the two -- a process can finish writing and exit between the two
# checks, but the mtime it left behind cannot un-happen.
#
# WHY A LATE-CREATED ARM CANNOT BE SWEPT.  The loop iterates the TIER LIST, not
# the directory.  An arm the peer creates after the plan is not in
# tierA_20260829.txt and is therefore invisible to this script by construction.
# That is not incidental: work-pr130r1-gs1on-* and work-pr130r1-g1off141-* were
# created between this round's first census and its plan.
#
#   ./retire_20260829.sh A              # dry run (default action)
#   CONFIRM=yes ./retire_20260829.sh A  # actually delete
#
# Pre-flights (all must have passed before CONFIRM=yes):
#   scripts/retire/verify_group_dupes_20260829.py  (188/188, 1231656 members)
#   scripts/retire/plan_20260829.py                (14 asserts, "OVERALL: PASS")
#   scripts/retire/archive_records_20260829.py     (integrity gate PASS n/n)
set -u
cd -P "$(dirname "$0")/../.." || exit 1     # -P: resolve the symlink
BASE=$PWD
echo "BASE=$BASE"
case "$BASE" in
    */toolkit/sbnd_xin*) echo "!! BASE is still the symlink path -- cd -P failed"; exit 1 ;;
esac
STATE=$BASE/scripts/retire/state-20260829
REC=$BASE/archive/records/em-pr-era-20260829
TIERS=${1:-A}
CONFIRM=${CONFIRM:-no}

PROTECTED=$(python3 -c "
import json;print(' '.join(json.load(open('$STATE/plan.json'))['PROTECTED']))")
KEEP=$(python3 -c "
import json;print(' '.join(json.load(open('$STATE/plan.json'))['KEEP']))")

# ---- interlock 0: broken symlinks BEFORE the round --------------------------
tierfiles=""
for t in ${TIERS//,/ }; do
    f=$BASE/scripts/retire/tier${t}_20260829.txt
    [ -f "$f" ] || { echo "no such tier list: $f"; exit 1; }
    tierfiles="$tierfiles $f"
done
list=$(cat $tierfiles)

pre_broken=$(find "$BASE" -xtype l | wc -l)
echo "broken symlinks before the round: $pre_broken"
if [ "$pre_broken" -ne 0 ]; then
    outside=0
    while IFS= read -r l; do
        [ -z "$l" ] && continue
        rel=${l#"$BASE"/}
        top=${rel%%/*}
        hit=no
        for d in $list; do [ "$d" = "$top" ] && hit=yes && break; done
        if [ "$hit" = no ]; then
            outside=$((outside+1))
            [ "$outside" -le 20 ] && echo "     $l -> ($top not in this round's removal set)"
        fi
    done < <(find "$BASE" -xtype l)
    if [ "$outside" -ne 0 ]; then
        echo "!! $outside broken symlink(s) OUTSIDE the removal set -- fix them before deleting anything."
        exit 2
    fi
    echo "   all $pre_broken are self-contained inside dirs already in the removal set"
    echo "   -- WARNING, not a refusal."
fi

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

# ---- interlock 2 (REFINED): a live batch that touches the REMOVAL set ------
# See the header.  A live job is no longer disqualifying on its own; a live job
# that names a removal-set dir is.  ALLOW_LIVE_JOBS is deliberately NOT honoured
# here any more -- there is nothing left for it to unlock that would be safe.
jobs=$(pgrep -a -f 'wire-cell |run_(ql|pr|nusel)_evt' 2>/dev/null \
       | grep -F 'sbnd_xin' \
       | grep -v 'retire_2026' \
       | grep -vE 'snapshot-bash|/claude|[[:space:]]claude([[:space:]]|$)|pgrep|grep -')
if [ -n "$jobs" ]; then
    njob=$(echo "$jobs" | wc -l)
    echo "note: $njob live sbnd_xin process(es) (a peer session owns doc pr/130)"
    echo "     loadavg $(cut -d' ' -f1-3 /proc/loadavg)  ncores $(nproc)"
    jhit=""
    for d in $list; do
        case "$jobs" in *"$d"*) jhit="$jhit $d";; esac
    done
    if [ -n "$jhit" ]; then
        echo "!! a LIVE JOB references REMOVAL CANDIDATES:$jhit"
        echo "refusing -- that arm is being written right now (M5)."; exit 2
    fi
    echo "   none of them names a dir in the removal set -- safe to proceed."
fi

# ---- interlock 3: no KEEP / PROTECTED dir in any tier list ------------------
for p in $KEEP $PROTECTED; do
    for d in $list; do
        [ "$d" = "$p" ] && { echo "!! survivor $p is in the tier list -- refusing"; exit 2; }
    done
done

# ---- interlock 4: the group-duplicate proof must be COMPLETE, by ROW COUNT --
# NOT `[ -s ]`.  archive_records_20260829.py DROPS 4.96 GiB of
# .groups/g<N>.tar.gz on the strength of verify_group_dupes_20260829.py's
# member-by-member proof that each is a copy of a surviving grp0825 Q/L root.
# A header-only or short manifest is non-empty and would pass an existence test
# while proving nothing -- the 08-25b lesson, applied to this round's class.
GD=$STATE/group-dupes.tsv
if [ ! -f "$GD" ]; then
    echo "!! $GD MISSING -- run verify_group_dupes_20260829.py before deleting"; exit 2
fi
gd_rows=$(grep -vc '^arm\b' "$GD")
gd_ok=$(awk -F'\t' 'NR>1 && $7=="OK"' "$GD" | wc -l)
gd_want=$(find $list -maxdepth 2 -path '*/.groups/g*.tar.gz' 2>/dev/null | wc -l)
if [ "$gd_rows" -ne "$gd_want" ] || [ "$gd_ok" -ne "$gd_want" ] || [ "$gd_want" -eq 0 ]; then
    echo "!! group-dupes proof incomplete: $gd_rows rows, $gd_ok OK, $gd_want archives on disk"
    exit 2
fi
echo "  group-dupes proof  $gd_ok/$gd_want archives verified duplicates of a KEEP Q/L root"

# ---- interlock 5: the live pi0 / hand-scan manifests must resolve into KEEP -
# The owner's stated next move reads these.  plan_20260829.py ASSERT 11 checked
# them; re-checked here because the plan may be minutes or hours old.
python3 - "$STATE" <<'PYEOF' || exit 2
import csv, json, os, sys
state = sys.argv[1]
plan = json.load(open(os.path.join(state, "plan.json")))
R = set(plan["R"])
bad = 0
for m, want in sorted(plan["LIVE_MANIFESTS"].items()):
    if not os.path.exists(m):
        print(f"!! {m}: MISSING"); bad += 1; continue
    n, arms = 0, set()
    for r in csv.DictReader(open(m), delimiter='\t'):
        p = (r.get('dump') or '').strip()
        if p and '/' in p:
            n += 1; arms.add(p.split('/')[0])
    hit = sorted(arms & R)
    if n != want:
        print(f"!! {m}: {n} rows, expected {want}"); bad += 1
    elif hit:
        print(f"!! {m}: dump arms in the REMOVAL set: {hit}"); bad += 1
print(f"  live manifests    {len(plan['LIVE_MANIFESTS'])-bad}/{len(plan['LIVE_MANIFESTS'])} resolve into KEEP")
sys.exit(1 if bad else 0)
PYEOF

# ---- interlock 6 (NEW): concurrent-writer safety, RE-DERIVED at delete time -
# plan-time evidence expires.  A removal-set dir written since the plan ran, or
# named by a process alive right now, refuses -- regardless of what ASSERT 12
# saw earlier.  The mtime half is the stronger of the two: a writer can exit
# between the two checks, but the mtime it left cannot un-happen.
planned_at=$(python3 -c "
import json;print(int(json.load(open('$STATE/plan.json'))['planned_at']))")
now=$(date +%s)
echo "  plan is $(( (now - planned_at) / 60 )) min old"
fresh=$(python3 - "$STATE" "$planned_at" <<'PYEOF'
import json, os, sys
state, t0 = sys.argv[1], float(sys.argv[2])
plan = json.load(open(os.path.join(state, "plan.json")))
out = [d for d in plan["R"] if os.path.isdir(d) and os.path.getmtime(d) > t0]
print(" ".join(out))
PYEOF
)
if [ -n "$fresh" ]; then
    echo "!! removal-set dir(s) written SINCE the plan ran: $fresh"
    echo "refusing -- re-run plan_20260829.py (RETIRE_REPLAN=1) and re-read the list."
    exit 2
fi
echo "  no removal-set dir has been written since the plan ran"

need_archive=$(python3 -c "
import json;print(' '.join(json.load(open('$STATE/plan.json'))['ARCHIVE']))")

MAN=$STATE/removed.tsv
if [ "$CONFIRM" = yes ]; then
    {
      echo "# retire_20260829.sh  tiers=$TIERS"
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
echo "work* DIRS remaining: $survivors   (expect >= $exp = KEEP; more only if the peer"
echo "                                    created arms after the plan -- they are not"
echo "                                    in the tier list and cannot have been swept)"
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
