#!/bin/bash
# doc 91 -- the COUNT-driven retire round, 2026-09-01b.
#
#   ./retire_20260901b.sh A              # dry run (default action)
#   CONFIRM=yes ./retire_20260901b.sh A  # actually delete
#
# Order (non-negotiable, from the 08-31 round):
#   scripts/retire/verify_group_dupes_20260901b.py  (writes group-dupes.tsv)
#   scripts/retire/plan_20260901b.py                (16 asserts, "OVERALL: PASS")
#   scripts/retire/archive_records_20260901b.py     (integrity gate PASS n/n)
#   ./retire_20260901b.sh A                         (DRY RUN, check dirs=/bytes=)
#   RETIRE_REPLAN=1 python3 .../plan_20260901b.py   (re-stamp planned_at)
#   CONFIRM=yes ./retire_20260901b.sh A
#
# OWNER SCOPE, verbatim (2026-09-01): "The sbnd_xin directory still have a lot
# of work* directory, do we need all of them?  Can we retire some and go
# minimum?  I understand that they do not take much disk space, but it is just
# difficult to look at them."  Then, on being shown the classification: "Peer
# is done, we can remove or retire them.  We want to keep the latest production
# though."
#
# So the metric is DIR COUNT: 101 -> 52.  Bytes (~7.0 GiB) are a side effect and
# are not the reason for any decision in this round.  work-*-prod0901b and
# work-*-grp0825 are untouchable by that instruction; doc 90's nine arms, which
# the 09-01 round protected by prefix while that round was live, are released by
# it.
#
# WHAT THIS ROUND KEEPS THAT A BYTE-DRIVEN ROUND WOULD NOT, and why.
# Running scripts/pr127_sentinels.py against work-*-prod0901b gives 27 PASS,
# 6 FAIL.  FIVE of those six still PASS in some arm on disk -- 47212 (pr/120
# backward-stem guard), 137238 (pr/93 r4 sccc), 173819 (pr/125 pass3_cone
# guard), 292643 (pr/130 B) and 406125 (pr/124 gap-band prune).  For 406125 the
# knob is still ON in wct-pr-perevt.jsonnet and the C++ log line still exists;
# it simply no longer fires.  That is the doc pr/127 failure mode -- a shipped
# fix dying silently -- recurring, and it is NOT diagnosed here (CLAUDE.md
# sec 5.7: report, do not tune).  Its consequence for THIS round is concrete:
# two of the five have exactly ONE passing arm on disk (137238 ->
# work-pr130r1-probe98-nuecc48, 292643 -> work-pr134-f086-mcp1k), so those arms
# are single points of failure for an open regression and interlock 8 refuses
# any round that would delete them.
#
# INTERLOCK 7 DEGENERATES THIS ROUND and says so ("gate not required") -- no
# production arm is released, so there is no successor gate to re-derive.
# Interlock 8 is new and is the one that bites; it was PROVEN able to fail
# before being trusted (drop work-pr134-f086-* from KEEP and it refuses on
# 292643; drop work-pr130r1-probe-* and it refuses on 137238).
#
# FORK NOTES.  Copied from retire_20260901.sh by `cp` and cmp-verified before
# editing.  Every date-bearing name was repointed: STATE=state-20260901b,
# REC=archive/records/campaign-close-20260901b, and the INTERPOLATED tier file
# tier${t}_20260901b.txt -- the 09-01 header warns that a missed rename here
# makes the dry run target the previous round's already-deleted list and report
# dirs=0, which reads like success.
set -u
cd -P "$(dirname "$0")/../.." || exit 1     # -P: resolve the symlink
BASE=$PWD
echo "BASE=$BASE"
case "$BASE" in
    */toolkit/sbnd_xin*) echo "!! BASE is still the symlink path -- cd -P failed"; exit 1 ;;
esac
STATE=$BASE/scripts/retire/state-20260901b
REC=$BASE/archive/records/campaign-close-20260901b
TIERS=${1:-A}
CONFIRM=${CONFIRM:-no}

PROTECTED=$(python3 -c "
import json;print(' '.join(json.load(open('$STATE/plan.json'))['PROTECTED']))")
KEEP=$(python3 -c "
import json;print(' '.join(json.load(open('$STATE/plan.json'))['KEEP']))")

# ---- interlock 0: broken symlinks BEFORE the round --------------------------
tierfiles=""
for t in ${TIERS//,/ }; do
    f=$BASE/scripts/retire/tier${t}_20260901b.txt
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
# NOT `[ -s ]`.  archive_records_20260901b.py DROPS 4.96 GiB of
# .groups/g<N>.tar.gz on the strength of verify_group_dupes_20260901b.py's
# member-by-member proof that each is a copy of a surviving grp0825 Q/L root.
# A header-only or short manifest is non-empty and would pass an existence test
# while proving nothing -- the 08-25b lesson, applied to this round's class.
GD=$STATE/group-dupes.tsv
if [ ! -f "$GD" ]; then
    echo "!! $GD MISSING -- run verify_group_dupes_20260901b.py (RETIRE_STATE=...state-20260901b) before deleting"; exit 2
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
# The owner's stated next move reads these.  plan_20260901b.py ASSERT 11 checked
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
    echo "refusing -- re-run plan_20260901b.py (RETIRE_REPLAN=1) and re-read the list."
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

# ---- interlock 8 (NEW): the sentinel suite survives the round --------------
# WHY THIS EXISTS.  Interlock 7 above degenerates to "not required" this round,
# because doc 91 releases no production arm.  A round whose newest interlock
# cannot fire has no new guard at all -- the 09-01 lesson, where a tmp-sweep
# interlock read "clear" only because its glob matched zero arms.  Interlock 8
# is the one that bites here, and it is RE-DERIVED at delete time: the arms are
# still on disk at this point, so "will every regression keep an arm that
# passes it" is answerable now and the plan-time answer cannot have gone stale.
#
# Same implementation as ASSERT 16 -- scripts/retire/sentinel_guard_20260901b.py
# -- so the two cannot drift.  Proven able to FAIL, not merely to pass: with
# work-pr134-f086-* removed from KEEP it refuses (evt 292643 loses its only
# passing arm); same with work-pr130r1-probe-* (evt 137238).
if ! python3 "$BASE/scripts/retire/sentinel_guard_20260901b.py" "$STATE/plan.json" "$BASE"; then
    echo "refusing -- interlock 8: a sentinel regression would lose its last passing arm"
    exit 2
fi

need_archive=$(python3 -c "
import json;print(' '.join(json.load(open('$STATE/plan.json'))['ARCHIVE']))")

MAN=$STATE/removed.tsv
if [ "$CONFIRM" = yes ]; then
    {
      echo "# retire_20260901b.sh  tiers=$TIERS"
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
