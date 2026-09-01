#!/bin/bash
# Retirement round 2026-09-01 (doc 89) -- the CLOSED pr/136-142 campaigns and
# doc 77 r3/r4 released, and production REBASED onto the pinned operating point.
#
# Fork of retire_20260831b.sh (whose own header said "Fork of
# retire_20260831b.sh" -- of itself; that is the fork hazard this family keeps
# reproducing, so it is fixed here rather than carried).  Every interlock is
# carried verbatim; interlock 7 is new.
#
# Owner scope, verbatim (2026-09-01): "the sbnd_xin directory is large now with
# a lots of work* directory, it is time to do a clean up and retire some work*
# directory. ... We basically just need to keep the latest production results,
# note, we have been testing the results with minimal outputs. if this is the
# case, we should save the other outputs locally so that we have the full
# validation sets. Other than this we want to minimize the sbnd_xin directory."
#
# THE PREMISE WAS MEASURED AND DOES NOT HOLD.  No production arm was ever run
# with minimal outputs: work-*-prod0901 and work-*-empre0901 carry mabc-pr.zip,
# the pctree, tracking-pr.root and nusel-evt<N>.tsv on 3067/3067 events, plus
# 1433 calib dumps (pr142_arms.sh sets PR_EXTRA_STAGES=pr_display, which ADDS
# an output).  Nothing had been dropped, so there was nothing to save back.
#
# WHAT WAS ACTUALLY WRONG, AND WHAT WAS DONE.  prod0901 predates the
# save_in_scope flip by 8 h, so its tracking-pr.root has no T_cluster tree.
# The owner chose to re-run all 3067 events at the pinned operating point
# (work-*-prod0901b) and gate the new arm against the old before releasing it.
# That gate is ASSERT 14 / interlock 7 below, and it is the FIRST full-scale
# check of the five-link chain doc77r3 -> doc77r4 -> master merge -> doc 87
# knobs -> save_in_scope, each previously gated only on 308 events.
#
# WHAT THIS ROUND REFUSES TO RELEASE, AND WHY THAT IS THE POINT.  The planner
# wanted the four doc-87 three-sample gate arms (4.2 G).  They stay: doc 87
# shipped hours ago and those arms are the acceptance evidence for a knob that
# moved the production operating point.  Interlock 7's gate is a DIFFERENT
# comparison and does not supersede them.  Same shape as 08-31's prod0825
# refusal -- a block the planner wanted and the evidence kept.
#
#   ./retire_20260901.sh A              # dry run (default action)
#   CONFIRM=yes ./retire_20260901.sh A  # actually delete
#
# Pre-flights (all must have passed before CONFIRM=yes):
#   scripts/retire/plan_20260901.py                (13 asserts, "OVERALL: PASS")
#   scripts/retire/archive_records_20260901.py     (integrity gate PASS n/n)
set -u
cd -P "$(dirname "$0")/../.." || exit 1     # -P: resolve the symlink
BASE=$PWD
echo "BASE=$BASE"
case "$BASE" in
    */toolkit/sbnd_xin*) echo "!! BASE is still the symlink path -- cd -P failed"; exit 1 ;;
esac
STATE=$BASE/scripts/retire/state-20260901
REC=$BASE/archive/records/campaign-close-20260901
TIERS=${1:-A}
CONFIRM=${CONFIRM:-no}

PROTECTED=$(python3 -c "
import json;print(' '.join(json.load(open('$STATE/plan.json'))['PROTECTED']))")
KEEP=$(python3 -c "
import json;print(' '.join(json.load(open('$STATE/plan.json'))['KEEP']))")

# ---- interlock 0: broken symlinks BEFORE the round --------------------------
tierfiles=""
for t in ${TIERS//,/ }; do
    f=$BASE/scripts/retire/tier${t}_20260901.txt
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
# NOTE, doc 89: the match is a SUBSTRING test, and this round has an arm pair
# where one name contains the other -- work-mcp2k-prod0901 is a substring of
# work-mcp2k-prod0901b.  While Phase 1 was writing the successor arm this
# interlock refused on the arm being RETIRED.  That is the safe direction and
# it clears when Phase 1 ends, but do not "fix" it into an exact match without
# thinking about what a partially-written arm named like another one means.
# See the header.  A live job is no longer disqualifying on its own; a live job
# that names a removal-set dir is.  ALLOW_LIVE_JOBS is deliberately NOT honoured
# here any more -- there is nothing left for it to unlock that would be safe.
jobs=$(pgrep -a -f 'wire-cell |run_(ql|pr|nusel)_evt' 2>/dev/null \
       | grep -F 'sbnd_xin' \
       | grep -v 'retire_2026' \
       | grep -vE 'snapshot-bash|/claude|[[:space:]]claude([[:space:]]|$)|pgrep|grep -')
if [ -n "$jobs" ]; then
    njob=$(echo "$jobs" | wc -l)
    echo "note: $njob live sbnd_xin process(es) -- see the header (a peer owns doc 88)"
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
# NOT `[ -s ]`.  archive_records_20260901.py DROPS 4.96 GiB of
# .groups/g<N>.tar.gz on the strength of verify_group_dupes_20260901.py's
# member-by-member proof that each is a copy of a surviving grp0825 Q/L root.
# A header-only or short manifest is non-empty and would pass an existence test
# while proving nothing -- the 08-25b lesson, applied to this round's class.
GD=$STATE/group-dupes.tsv
if [ ! -f "$GD" ]; then
    echo "!! $GD MISSING -- run verify_group_dupes_20260901.py (RETIRE_STATE=...state-20260901) before deleting"; exit 2
fi
# 2026-08-31: the 08-29 form ended `|| [ "$gd_want" -eq 0 ]`, i.e. it treated
# ZERO group archives as proof of a broken census.  That was right for the
# round it was written for -- that round's removal set was full of group-mode
# arms and a zero could only mean the find had failed.  It is wrong here: this
# round's removal set legitimately contains none (archive_records reports
# "groupin class dropped 0.00 GiB", and the class is re-counted from disk
# below).  So the empty case is handled EXPLICITLY rather than by relaxing the
# non-empty case, which keeps its full strength: gd_want is still re-derived by
# a fresh find at DELETION time, the proof file must still agree with it, and a
# header-only file still cannot pass when any archive exists.
gd_rows=$(grep -v '^#' "$GD" | grep -vc '^arm\b')
gd_ok=$(awk -F'\t' '!/^#/ && NR>1 && $7=="OK"' "$GD" | wc -l)
gd_want=$(find $list -maxdepth 2 -path '*/.groups/g*.tar.gz' 2>/dev/null | wc -l)
if [ "$gd_want" -eq 0 ]; then
    if [ "$gd_rows" -ne 0 ]; then
        echo "!! group class is empty on disk but the proof carries $gd_rows row(s) -- stale proof"
        exit 2
    fi
    if ! grep -q 'group-input class is EMPTY' "$GD"; then
        echo "!! group class is empty on disk but $GD does not record it -- rerun the verifier"
        exit 2
    fi
    echo "  group-dupes proof  EMPTY class, recorded by the verifier and re-confirmed on disk"
elif [ "$gd_rows" -ne "$gd_want" ] || [ "$gd_ok" -ne "$gd_want" ]; then
    echo "!! group-dupes proof incomplete: $gd_rows rows, $gd_ok OK, $gd_want archives on disk"
    exit 2
else
    echo "  group-dupes proof  $gd_ok/$gd_want archives verified duplicates of a KEEP Q/L root"
fi

# ---- interlock 5: the live pi0 / hand-scan manifests must resolve into KEEP -
# The owner's stated next move reads these.  plan_20260901.py ASSERT 11 checked
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
    echo "refusing -- re-run plan_20260901.py (RETIRE_REPLAN=1) and re-read the list."
    exit 2
fi
echo "  no removal-set dir has been written since the plan ran"

# ---- interlock 7 (NEW): the successor gate, RE-DERIVED at delete time -------
# prod0901 is in the removal set only because prod0901b reproduces it on every
# SHARED product.  ASSERT 14 checked that at plan time; re-checked here by ROW
# COUNT because a proof written before the last event finished is not a proof
# about the arm.  Deliberately NOT `[ -s ]`: a header-only file is non-empty.
SG=$STATE/successor-gate.tsv
sg_need=$(python3 -c "
import json;p=json.load(open('$STATE/plan.json'))
S=p.get('SUCCESSOR',{});R=set(p['R'])
print(1 if [o for o in S if o in R] else 0)")
if [ "$sg_need" = 1 ]; then
    if [ ! -f "$SG" ]; then
        echo "!! $SG MISSING -- run scripts/doc89_successor_gate.py before deleting"; exit 2
    fi
    sg_want=$(python3 -c "
import json;print(json.load(open('$STATE/plan.json')).get('SUCCESSOR_ROWS',0))")
    sg_rows=$(grep -v '^#' "$SG" | grep -vc '^sample')
    sg_ok=$(awk -F'\t' '!/^#/ && $1!="sample" && $NF=="OK"' "$SG" | wc -l)
    if [ "$sg_rows" -ne "$sg_want" ] || [ "$sg_ok" -ne "$sg_want" ]; then
        echo "!! successor gate incomplete: $sg_rows rows, $sg_ok OK, want $sg_want"
        echo "refusing -- prod0901 may not be released without it."; exit 2
    fi
    echo "  successor gate    $sg_ok/$sg_want events reproduce on every shared product"
    for s_ in nuecc48 ncpi0 mcp1k mcp2k; do
        [ -d "$BASE/work-$s_-prod0901b" ] || { echo "!! successor arm work-$s_-prod0901b missing"; exit 2; }
    done
    echo "  successor arms    all four prod0901b arms present on disk"
else
    echo "  successor gate    not required (no superseded arm in the removal set)"
fi

need_archive=$(python3 -c "
import json;print(' '.join(json.load(open('$STATE/plan.json'))['ARCHIVE']))")

MAN=$STATE/removed.tsv
if [ "$CONFIRM" = yes ]; then
    {
      echo "# retire_20260901.sh  tiers=$TIERS"
      echo "# started              $(date -Is)"
      echo "# wcp-porting-img HEAD $(git -C "$BASE" rev-parse --short HEAD)"
      echo "# toolkit HEAD         $(git -C /home/xqian/toolkit-dev/toolkit rev-parse --short HEAD)"
      echo "# broken symlinks pre  $pre_broken"
      echo "# du -sh sbnd_xin pre  $(du -sh "$BASE" | cut -f1)"
      echo "# df filesystem pre   $(df -h "$BASE" | tail -1 | awk '{print $4" avail"}')"
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
  echo "# df filesystem post  $(df -h "$BASE" | tail -1 | awk '{print $4" avail"}')"
} >> "$MAN"
du -sh "$BASE"
df -h "$BASE" | tail -1
echo "manifest: $MAN"
