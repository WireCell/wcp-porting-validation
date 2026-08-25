#!/bin/bash
# Retirement round 2026-08-25 -- doc 81, the group-mode re-baseline.
# 66 arms removed (72.6 G), 38 kept.
#
# Fork of retire_20260823.sh.  Every earlier fix is carried verbatim: the
# `cd -P` symlink fix, the `-type d` survivor census, the real-tier-file
# Bokeh interlock, the widened interlock-2 grep, interlocks 0/1/2 exiting in
# dry run too, and 08-17's interlock-0 refinement (a broken symlink whose
# top-level dir is itself in the removal set is a WARNING, since it vanishes
# with its dir; any broken link OUTSIDE the removal set still refuses).
#
# Owner scope for this round:
#   * the pr/112 + pr/112i option scan (54 arms): released, now that the
#     doc-81 arms are gated byte-identical against work-pr112i-snapD2-*.
#   * work-pr104-on4-* (4) and work-pr104-flipchk-* (4): released -- the
#     08-23 round's own deferred promise, due now that prod0825 covers all
#     3067 events and is byte-identical on every overlapping one.
#   * the four work-vtx106-*-nuecc48 pr/111 arms: released, pr/111 is closed.
#   * KEEP: work-img-*, work-*-ql0819, work-*-prod0823, the eight new
#     work-*-{grp0825,prod0825} roots, the six sim arms, the four
#     work-vtx105-base-* label-epoch arms, and the git-tracked arms.
#
# ONE THING THIS ROUND DOES THAT NO PRIOR ROUND DID, and it is why doc 81
# sec 8.1 stays checkable: work-pr112i-snapD2-* is the ONLY per-event
# reference at the CURRENT operating point (work-*-prod0823 is pre-flip, doc
# 81 sec 2), so retiring it deletes the reference side of the campaign's own
# gate.  scripts/retire/hash_manifest_20260825.py freezes that side -- BOTH
# halves, archives AND every branch of tracking-pr.root -- into
# state-20260825/hashes/*.tsv, which is git-tracked.  Interlock 4 refuses the
# round if a cited manifest is missing.
#
#   ./retire_20260825.sh A              # dry run (default action)
#   CONFIRM=yes ./retire_20260825.sh A  # actually delete
#
# Tier dispositions: tier A archives (already done via
# archive_records_20260819.py) then deletes -- all removal-set dirs this round.
#
# Pre-flights (all must have passed before CONFIRM=yes):
#   scripts/retire/plan_20260819.py             (9 asserts, "OVERALL: PASS")
#   scripts/retire/archive_records_20260819.py  (integrity gate PASS n/n)
#
# THE cd -P FIX (inherited from 08-13/08-16, carried verbatim): every round
# before 08-13 did `cd "$(dirname "$0")/../.." ; BASE=$PWD`. Invoked through
# toolkit/sbnd_xin -- a SYMLINK to wcp-porting-img/sbnd/sbnd_xin, the normal
# way to reach this tree -- $PWD was the logical path, so $BASE named a
# symlink. `find "$BASE" ...` does not descend a symlink argument and
# `du -sh "$BASE"` measures the link itself, which made interlock 0 and the
# post-round survivor census vacuous in the 08-13 round. `rm -rf "$BASE/$d"`
# was unaffected -- only the final path component matters there. Fixed with
# `cd -P`; this script keeps the explicit BASE echo right after it.
set -u
cd -P "$(dirname "$0")/../.." || exit 1     # -P: resolve the symlink
BASE=$PWD
echo "BASE=$BASE"
case "$BASE" in
    */toolkit/sbnd_xin*) echo "!! BASE is still the symlink path -- cd -P failed"; exit 1 ;;
esac
STATE=$BASE/scripts/retire/state-20260825
REC=$BASE/archive/records/prod0825-groupmode-20260825
TIERS=${1:-A}
CONFIRM=${CONFIRM:-no}

PROTECTED=$(python3 -c "
import json;print(' '.join(json.load(open('$STATE/plan.json'))['PROTECTED']))")
KEEP=$(python3 -c "
import json;print(' '.join(json.load(open('$STATE/plan.json'))['KEEP']))")

# ---- interlock 0: broken symlinks BEFORE the round --------------------------
# See header note: broken links whose top-level dir is already in tier A are
# a WARNING (they vanish with their dir); any other broken link REFUSES,
# same as every prior round.
tierfiles=""
for t in ${TIERS//,/ }; do
    f=$BASE/scripts/retire/tier${t}_20260825.txt
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
    echo "   (doc 73 sec.12.5 harness bug, work-*-vfcbr3off; real data lives in the KEEP"
    echo "    imaging hubs via a separate working symlink) -- WARNING, not a refusal."
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

# ---- interlock 4 (NEW this round): the gate reference must be frozen --------
# work-pr112i-snapD2-* is the ONLY per-event arm at the CURRENT operating
# point; work-*-prod0823 is pre-flip (doc 81 sec 2).  Deleting it without a
# frozen manifest would make doc 81 sec 8.1's PASS unrepeatable forever.
# Both halves of that gate must be on disk: the pr85 archive rollups AND the
# pr94 per-branch rollup of tracking-pr.root, which hash_manifest_20260825.py
# writes into one file per arm.
HASHDIR=$STATE/hashes
for a in work-pr112i-snapD2-nuecc48 work-pr112i-snapD2-ncpi0 \
         work-pr112i-snapD2-mcp1k   work-pr112i-snapD2-mcp2k \
         work-pr112i-flipchk-nuecc48 work-pr112i-flipchk-ncpi0; do
    inlist=no
    for d in $list; do [ "$d" = "$a" ] && inlist=yes; done
    [ "$inlist" = yes ] || continue
    f=$HASHDIR/$a.tsv
    [ -s "$f" ] || { echo "!! $a is being retired but $f is missing -- refusing"; exit 2; }
    nroot=$(awk -F'\t' '$2=="tracking-pr.root"' "$f" | wc -l)
    nevt=$(ls -d "$BASE/$a"/pr_evt* 2>/dev/null | wc -l)
    if [ "$nroot" -ne "$nevt" ]; then
        echo "!! $a: manifest has $nroot tracking-pr.root rows, arm has $nevt events -- refusing"
        exit 2
    fi
    echo "  frozen  $a  ($nroot ROOT + $(awk -F'\t' '$2!="tracking-pr.root" && $0!~/^#/' "$f" | wc -l) archive rollups)"
done

need_archive=$(python3 -c "
import json;print(' '.join(json.load(open('$STATE/plan.json'))['ARCHIVE']))")

MAN=$STATE/removed.tsv
if [ "$CONFIRM" = yes ]; then
    {
      echo "# retire_20260825.sh  tiers=$TIERS"
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
