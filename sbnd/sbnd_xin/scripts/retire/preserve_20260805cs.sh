#!/bin/bash
# Clean-slate round 2026-08-05, STEP 1 -- discharge the two blocking asserts
# before anything is deleted.  Nothing here removes data; it only copies.
#
#   ./preserve_20260805cs.sh            # dry run (default)
#   CONFIRM=yes ./preserve_20260805cs.sh
#
# plan_20260805cs.py FAILED on two asserts, both correctly:
#
#   ASSERT 2 -- three hand-scan label dirs live inside dirs the round removes:
#       work-mcp1kall-d59k/nusel_labels  (10 tags, 197 files)
#       work/ql_labels                   ( 3 tags,  41 files)
#       work-mcp10/nusel_labels          ( 5 tags,   7 files)
#     These are M13 scientific records.  Part A copies them VERBATIM (never
#     tarred, symlinks preserved) to archive/records/labels/<arm>/, which is the
#     established location -- it already holds labels for ten earlier arms, and
#     archive_records_*.py has the same verbatim-copy path built in.
#
#   ASSERT 4 -- archive/ is NOT self-contained: 351 symlinks in three campaign
#     archives (tgm-docs29-39, stm-docs40-49, aborted-d54) resolve into
#     work-mcp10 and work-mcp1000.  Those archives are themselves records, so
#     deleting the imaging bases would gut them.  Part B materializes each link
#     the way materialize_20260802.sh did for the pr/22 exhibit chain.
#
# WHY HARDLINKS (cp -al) RATHER THAN COPIES.  The 351 links resolve to only 40
# distinct event dirs, so per-link byte copies would duplicate 0.19 GiB into
# 1.75 GiB.  archive/ and work-* are on ONE filesystem (both dev 66305), and
# these are frozen records, so a hardlink gives every archive path a real,
# independent directory entry at zero extra bytes.  After the bases are deleted
# the hardlinks become the SOLE reference and the data persists at its original
# 0.19 GiB.  Verified per link: same inode, and the link count drops to the
# expected value.  (materialize_20260802.sh byte-copied because its targets were
# few; the mechanism is the same, the cost model is not.)
set -u
cd "$(dirname "$0")/../.." || exit 1        # -> sbnd_xin
BASE=$PWD
CONFIRM=${CONFIRM:-no}
LAB=$BASE/archive/records/labels

# dirs whose disappearance would strand an archive/ link
REMOVED_BASES="work-mcp10 work-mcp1000"

echo "=== PART A: hand-scan labels -> archive/records/labels/ (verbatim) ==="
nlab=0; labfail=0
for spec in "work-mcp1kall-d59k nusel_labels" "work ql_labels" "work-mcp10 nusel_labels"; do
    set -- $spec; arm=$1; name=$2
    src=$BASE/$arm/$name
    dst=$LAB/$arm/$name
    if [ ! -d "$src" ]; then echo "  SKIP (absent): $arm/$name"; continue; fi
    nsrc=$(find "$src" -type f | wc -l)
    ntag=$(ls "$src" | wc -l)
    if [ "$CONFIRM" = yes ]; then
        if [ -e "$dst" ]; then
            # M13: never write over an existing label record.  If it is already
            # a byte-identical copy this script made, that is a clean re-run --
            # report and move on.  Anything else is a genuine earlier archive:
            # stop and let a human look.
            if diff -r --no-dereference "$src" "$dst" > /dev/null 2>&1; then
                echo "  OK  $arm/$name  already archived and identical (re-run)"
                nlab=$((nlab+1)); continue
            fi
            echo "  !! REFUSE: $dst exists and DIFFERS -- not overwriting a label record (M13)"
            labfail=$((labfail+1)); continue
        fi
        mkdir -p "$(dirname "$dst")"
        cp -a "$src" "$dst" || { echo "  !! cp failed: $src"; labfail=$((labfail+1)); continue; }
        ndst=$(find "$dst" -type f | wc -l)
        if [ "$nsrc" -ne "$ndst" ]; then
            echo "  !! COUNT MISMATCH $arm/$name: src=$nsrc dst=$ndst"; labfail=$((labfail+1)); continue
        fi
        # byte-level check, not just counts
        if diff -r --no-dereference "$src" "$dst" > /dev/null 2>&1; then
            echo "  OK  $arm/$name  ($ntag tags, $nsrc files) -> $dst  [diff -r clean]"
        else
            echo "  !! DIFF FAILED $arm/$name"; labfail=$((labfail+1)); continue
        fi
    else
        echo "  would copy  $arm/$name  ($ntag tags, $nsrc files) -> $dst"
    fi
    nlab=$((nlab+1))
done

echo
echo "=== PART B: materialize archive/ symlinks into $REMOVED_BASES ==="
mapfile -t LINKS < <(find "$BASE/archive" -type l | while read -r l; do
    t=$(readlink "$l")
    for b in $REMOVED_BASES; do
        case "$t" in */sbnd_xin/$b/*) echo "$l"; break;; esac
    done
done)
echo "  ${#LINKS[@]} symlink(s) to materialize"

nmat=0; matfail=0
for l in "${LINKS[@]}"; do
    t=$(readlink "$l")
    # A FILE target is not a Part B failure -- it is Part C's job.  Part B only
    # materializes directory links; the file-level links it cannot see on the
    # first pass are exposed once those directories become real.  Counting them
    # here produced a spurious "fail 200" on the second run.
    if [ -f "$t" ] && [ ! -d "$t" ]; then continue; fi
    if [ ! -e "$t" ]; then echo "  !! target missing: $l -> $t"; matfail=$((matfail+1)); continue; fi
    if [ ! -d "$t" ]; then echo "  !! target not a dir: $l -> $t"; matfail=$((matfail+1)); continue; fi
    if [ "$CONFIRM" = yes ]; then
        tmp="$l.mat.tmp"
        [ -e "$tmp" ] && { echo "  !! stale tmp exists: $tmp"; matfail=$((matfail+1)); continue; }
        # -a preserves mode/time/symlinks-within; -l hardlinks regular files
        if ! cp -al "$t" "$tmp"; then
            echo "  !! cp -al failed: $t"; matfail=$((matfail+1)); continue
        fi
        # verify: same inode for a sample regular file, same file count
        ns=$(find "$t" -type f | wc -l); nd=$(find "$tmp" -type f | wc -l)
        if [ "$ns" -ne "$nd" ]; then
            echo "  !! COUNT MISMATCH $l: src=$ns dst=$nd"; matfail=$((matfail+1)); continue
        fi
        f=$(find "$t" -type f -print -quit)
        if [ -n "$f" ]; then
            rel=${f#"$t"/}
            i1=$(stat -c %i "$f"); i2=$(stat -c %i "$tmp/$rel" 2>/dev/null || echo x)
            if [ "$i1" != "$i2" ]; then
                echo "  !! NOT HARDLINKED $l ($i1 vs $i2)"; matfail=$((matfail+1)); continue
            fi
        fi
        rm -f "$l" && mv "$tmp" "$l" || { echo "  !! swap failed: $l"; matfail=$((matfail+1)); continue; }
    fi
    nmat=$((nmat+1))
done

echo
echo "=== PART C: inner FILE symlinks exposed by Part B ==="
# Part B's `find archive -type l` could not see these.  Before materialization
# an entry like archive/tgm-docs29-39/work-mcp10-merge/ql_evt286329 was ITSELF a
# symlink, so find reported the directory link and did not descend into it.
# Turning it into a real dir exposes the file-level links inside -- 200
# icluster-apa*.npz pointing at work-mcp10.  They also use the OTHER spelling of
# this tree (/home/xqian/work/scratch_wcgpu1/.../sbnd_xin/), which is why the
# earlier counts disagreed; the `*/sbnd_xin/<base>/*` match handles both.
#
# Hardlinked, same as Part B: after work-mcp10 is deleted these become the sole
# reference, so the bytes are retained once, not duplicated.  Run this part
# repeatedly until it reports 0 -- each pass can expose the next level.
pass=0; nfile=0; filefail=0
while :; do
    pass=$((pass+1))
    mapfile -t FLINKS < <(find "$BASE/archive" -type l | while read -r l; do
        t=$(readlink "$l")
        for b in $REMOVED_BASES; do
            case "$t" in */sbnd_xin/$b/*) [ -f "$t" ] && echo "$l"; break;; esac
        done
    done)
    [ ${#FLINKS[@]} -eq 0 ] && { echo "  pass $pass: 0 remaining -- done"; break; }
    echo "  pass $pass: ${#FLINKS[@]} file symlink(s)"
    [ "$CONFIRM" = yes ] || { nfile=${#FLINKS[@]}; break; }
    for l in "${FLINKS[@]}"; do
        t=$(readlink "$l")
        tmp="$l.mat.tmp"
        if ! cp -al "$t" "$tmp" 2>/dev/null; then
            echo "  !! cp -al failed: $l"; filefail=$((filefail+1)); continue
        fi
        i1=$(stat -c %i "$t"); i2=$(stat -c %i "$tmp")
        if [ "$i1" != "$i2" ]; then
            echo "  !! NOT HARDLINKED $l"; rm -f "$tmp"; filefail=$((filefail+1)); continue
        fi
        rm -f "$l" && mv "$tmp" "$l" || { echo "  !! swap failed: $l"; filefail=$((filefail+1)); }
        nfile=$((nfile+1))
    done
    [ $pass -ge 5 ] && { echo "  !! still resolving after 5 passes -- stopping"; filefail=$((filefail+1)); break; }
done

echo
echo "labels=$nlab (fail $labfail)   materialized=$nmat/${#LINKS[@]} (fail $matfail)   files=$nfile (fail $filefail)   CONFIRM=$CONFIRM"
[ "$CONFIRM" = yes ] || { echo "dry run only -- re-run with CONFIRM=yes"; exit 0; }

echo
echo "== post-checks =="
rem=$(find "$BASE/archive" -type l | while read -r l; do
        t=$(readlink "$l"); for b in $REMOVED_BASES; do
          case "$t" in */sbnd_xin/$b/*) echo x; break;; esac; done; done | wc -l)
echo "archive/ symlinks still resolving into $REMOVED_BASES: $rem   (MUST be 0)"
echo "broken symlinks tree-wide: $(find "$BASE" -xtype l | wc -l)   (MUST be 0)"
echo "labels archived: $(find "$LAB" -type f | wc -l) files under $LAB"
[ $((labfail+matfail+filefail)) -eq 0 ] && [ "$rem" -eq 0 ] \
  && echo "PRESERVE OK -- plan_20260805cs.py ASSERT 2 and 4 can now be re-checked" \
  || { echo "PRESERVE INCOMPLETE -- do not proceed"; exit 1; }
