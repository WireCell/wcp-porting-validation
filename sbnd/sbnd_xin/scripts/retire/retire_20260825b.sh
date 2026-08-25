#!/bin/bash
# Retirement round 2026-08-25b -- the stage-A reference side and the pre-flip
# PR baseline.  12 arms removed (~40 G), 26 kept.
#
# Fork of retire_20260825.sh (same day, second pass -- precedent
# retire_20260819b.sh).  Every earlier fix is carried verbatim: the `cd -P`
# symlink fix, the `-type d` survivor census, the real-tier-file Bokeh
# interlock, the widened interlock-2 grep, interlocks 0/1/2 exiting in dry run
# too, and 08-17's interlock-0 refinement.
#
# Owner scope for this round, given directly:
#   "I assume we can safely retire [work-img-{4 samples}, work-*-ql0819,
#    work-*-prod0823], and recover the disk"
#
#   * work-img-{nuecc48,ncpi0,mcp1k,mcp2k} (~19 G) -- doc 81 sec 7 proved these
#     byte-identical to the imaging half of work-<s>-grp0825 (24536/24536).
#   * work-<s>-ql0819 (~10.3 G) -- the Q/L half of the same gate.
#   * work-<s>-prod0823 (~10.2 G) -- PRE-flip (doc 81 sec 4), superseded by
#     prod0825.  The PROTECTED.txt RELEASED lines are written this round.
#   * KEEP: the two SIM imaging hubs work-img-{r1qlmc,r2mc} (NOT duplicates --
#     no grp0825 arm exists for either sim sample), the eight
#     work-*-{grp0825,prod0825} roots, the four work-vtx105-base-* label-epoch
#     arms, the six sim arms, and the git-tracked / non-reproducible arms.
#
# INTERLOCK 4 IS THE POINT OF THIS ROUND, and it is stricter than 08-25's.
# That round froze a PR arm before deleting it and checked the manifest with
# `[ -s "$f" ]`.  Here BOTH HALVES of one gate reference go at once, so a
# vacuous freeze would take doc 81 sec 7's "PASS 24536/24536" with it -- and
# `[ -s ]` cannot tell a complete manifest from a header-only one.  It is a
# live hazard, not a hypothetical: hash_manifest_20260825.py matches
# `pr_evt(\d+)$` and stage-A arms are `evt<N>`/`ql_evt<N>`, so reusing it here
# would have walked ZERO events and written exactly that header-only file.
# Hence a second freeze tool (hash_manifest_stagea_20260825b.py) and hence
# interlock 4 asserting ROW COUNTS that must sum to 24536.
#
# ALSO DONE BEFORE THIS RUNS, and not by this script: work-probe178410a's
# evt178410/ was a SYMLINK into work-img-mcp2k, with its ql_evt npz linking
# through it.  That arm is PROTECTED precisely because it cannot be
# re-captured, and the round would have left five broken links inside it with
# no error.  plan_20260825b.py ASSERT 4 caught it; the link was replaced by
# the real bytes (cp -rL, 6.7 MB -> 17 MB) and the arm is now self-contained.
#
#   ./retire_20260825b.sh A              # dry run (default action)
#   CONFIRM=yes ./retire_20260825b.sh A  # actually delete
#
# Pre-flights (all must have passed before CONFIRM=yes):
#   scripts/retire/hash_manifest_stagea_20260825b.py nuecc48 ncpi0 mcp1k mcp2k
#   scripts/retire/hash_manifest_pr_20260825b.py work-{4 samples}-prod0823
#   scripts/retire/plan_20260825b.py             (12 asserts, "OVERALL: PASS")
#   scripts/retire/archive_records_20260825b.py  (integrity gate PASS n/n)
#
set -u
cd -P "$(dirname "$0")/../.." || exit 1     # -P: resolve the symlink
BASE=$PWD
echo "BASE=$BASE"
case "$BASE" in
    */toolkit/sbnd_xin*) echo "!! BASE is still the symlink path -- cd -P failed"; exit 1 ;;
esac
STATE=$BASE/scripts/retire/state-20260825b
REC=$BASE/archive/records/stagea-refside-20260825b
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
    f=$BASE/scripts/retire/tier${t}_20260825b.txt
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

# ---- interlock 4: the frozen manifests must be COMPLETE, by ROW COUNT ------
# NOT `[ -s ]`.  A header-only .tsv is non-empty and would pass that test while
# preserving nothing; see the header for why that is a live hazard here and not
# a hypothetical.  Stage A is 8 products/event and must total doc 81 sec 7's own
# 24536; prod0823 is 3 products/event over 3067 events = 9201.
HASHDIR=$STATE/hashes
stagea_total=0
i4_fail=0
check_rows () {           # <file> <expected rows>
    f=$HASHDIR/$1
    if [ ! -f "$f" ]; then
        echo "!! $1 MISSING -- freeze before deleting"; i4_fail=$((i4_fail+1)); return
    fi
    got=$(grep -vc '^#' "$f")
    if [ "$got" -ne "$2" ]; then
        echo "!! $1: $got rows, expected $2"; i4_fail=$((i4_fail+1)); return
    fi
    printf '  frozen  %-30s %6d rows\n' "$1" "$got"
}
for pair in "stagea-nuecc48.tsv 384" "stagea-ncpi0.tsv 152" \
            "stagea-mcp1k.tsv 8000" "stagea-mcp2k.tsv 16000"; do
    set -- $pair; check_rows "$1" "$2"
    [ -f "$HASHDIR/$1" ] && stagea_total=$((stagea_total + $(grep -vc '^#' "$HASHDIR/$1")))
done
for pair in "work-nuecc48-prod0823.tsv 144" "work-ncpi0-prod0823.tsv 57" \
            "work-mcp1k-prod0823.tsv 3000" "work-mcp2k-prod0823.tsv 6000"; do
    set -- $pair; check_rows "$1" "$2"
done
if [ "$stagea_total" -ne 24536 ]; then
    echo "!! stage-A rows total $stagea_total, expected 24536 (doc 81 sec 7's archive count)"
    i4_fail=$((i4_fail+1))
else
    echo "  stage-A total $stagea_total == doc 81 sec 7's 24536 archives"
fi
if [ "$i4_fail" -ne 0 ]; then
    echo "refusing: $i4_fail frozen-manifest check(s) failed"; exit 2
fi

# ---- interlock 5 (new): the two live tools must already be repointed -------
# doc 82's reproducer defaulted to work-<s>-ql0819 and closed TODAY.  A repoint
# is the fix, not an ACK, so it is verified here as well as in the plan.
for pair in "scripts/multi/repro_ql_nondet.sh grp0825" \
            "scripts/multi/ql_legacy_gate.sh work-mcp1k-grp0825"; do
    set -- $pair
    if ! grep -q -- "$2" "$BASE/$1"; then
        echo "!! $1 does not reference $2 -- repoint it before deleting its default"
        exit 2
    fi
    echo "  repointed  $1 -> $2"
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
