#!/bin/bash
# doc 98 -- release closed-round scratch from ~/tmp, 2026-09-02.
#
#   ./scripts/retire/sweep_tmp_20260902.sh              # dry run (default)
#   CONFIRM=yes ./scripts/retire/sweep_tmp_20260902.sh  # actually delete
#
# OWNER SCOPE, verbatim (2026-09-02): "By the way, please also clean up a bit
# for ~/tmp please".  62 GiB there; this releases ~30 GiB.
#
# THE RULE FOR A PINNED BINARY, carried from sweep_tmp_20260901.sh: a *-libsnap
# is the binary some arm was produced with, and a pin with no surviving arm
# explains nothing -- so it goes WITH its arms and never before them.
# INTERLOCK 1 re-derives that mapping on every run (which scripts name the
# libsnap; which work-* dirs those scripts write; do any still exist) rather
# than trusting this comment.  Measured today:
#   doc94-libsnap doc94b-libsnap doc94r3-libsnap   no surviving arm  -> DROP
#   d97-libsnap                no script references it at all       -> DROP
#   doc94c-libsnap    backs work-*-r2scan{on,off}      (kept)        -> KEEP
#   doc94r3b-libsnap  backs work-dbg25a-*             (kept)        -> KEEP
#   d97b-libsnap      the binary PRODUCTION ran under (d97fv)       -> KEEP
#   prod0901b-libsnap backs em114*/stmfb8*/pr130r1-probe* (kept)    -> KEEP
#
# WHY sweep_tmp_20260901.sh's LIST IS NOT REUSED.  That script never ran (doc
# 89 recorded it as "still owed"), yet all eleven of its DROP dirs are already
# gone by some other route -- so its list is spent.  The one item it kept that
# this round releases is doc87/lib-knob, which it called "an exact md5
# duplicate of lib-flip".  RE-MEASURED TODAY rather than inherited: 790/790
# files md5-identical.  Released on today's measurement.
#
# WHAT IS DELIBERATELY KEPT, each with its reason:
#   doc25gate, pinlib*   a PEER SESSION IS LIVE (PDVD doc 25 gate round 7;
#                        both touched 16:51 today).  INTERLOCK 2 refuses the
#                        whole sweep if anything under a DROP path moved.
#   claude-*             live session scratchpads
#   doc87/lib-{pre,tc,post,flip} + doc87/*.log|json  four DISTINCT binaries
#                        backing the doc-87 arms this tree still keeps
#                        (87knob-min/sup, 87flip, 87grp-*); PROTECTED.txt's
#                        release condition for them is still unmet
#   d97b-libsnap doc94c-libsnap doc94r3b-libsnap prod0901b-libsnap  see above
set -u
TMP=$HOME/tmp
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
CONFIRM=${CONFIRM:-no}

# closed-round pins (interlock 1 re-checks each), a verified duplicate,
# regenerable cmake build trees, and group-build scratch whose OUTPUTS live in
# sbnd_xin/input_files_reco1/{extracted,staged}-dbg25*.
LIBSNAPS="doc94-libsnap doc94b-libsnap doc94r3-libsnap d97-libsnap pr142-libsnap"
OTHER="doc87/lib-knob cm-rel cm-fix1 cm-strict dbg25/groupbuild"
# scan2 pr40r5 pr43_cleanhead_ref48 pr88 pr63_render -- NOT swept.  A find for
# *.md5|*.tsv|*.json|*.log found 10-74 artifact-class files in EACH of them
# (scored.json, rank-*.tsv, bisect-*.log, track_candidates.json ...), i.e. they
# are closed-round evidence, not scratch, and this round has no record layer for
# ~/tmp.  0.33 GiB is not worth deleting a possible last copy -- doc 89's own
# rule was to copy such artifacts out of doc87/ BEFORE sweeping it.  Left for a
# round that archives them properly.

echo "== interlock 1: a pinned binary may not outlive... nor precede its arms =="
fail=0
for L in $LIBSNAPS; do
  [ -e "$TMP/$L" ] || { echo "  skip   $L (absent)"; continue; }
  srcs=$(grep -rl -- "$L" "$SX/scripts/" 2>/dev/null)
  if [ -z "$srcs" ]; then echo "  DROP   $L (orphan: no script references it)"; continue; fi
  alive=$(grep -rhoE 'work-[A-Za-z0-9_-]+' $srcs 2>/dev/null | sort -u \
          | while read -r a; do [ -d "$SX/$a" ] && echo "$a"; done)
  # grp0825 is the shared imaging substrate, not evidence that this pin is live
  alive=$(echo "$alive" | grep -v 'grp0825' | grep -c . )
  if [ "$alive" -gt 0 ]; then
    echo "  REFUSE $L -- $alive surviving arm(s) still cite it"; fail=1
  else
    echo "  DROP   $L (no surviving arm)"
  fi
done
[ $fail -eq 0 ] || { echo "REFUSE: a libsnap still backs a live arm"; exit 2; }

echo "== interlock 2: nothing under a DROP path may be moving =="
sig() { for p in "$@"; do [ -e "$p" ] && find "$p" -newermt '-30 minutes' -print -quit 2>/dev/null; done; }
PATHS=""
for L in $LIBSNAPS; do PATHS="$PATHS $TMP/$L"; done
for O in $OTHER;    do PATHS="$PATHS $TMP/$O"; done
hot=$(sig $PATHS)
[ -z "$hot" ] || { echo "REFUSE: recently written under a drop path: $hot"; exit 2; }
echo "  ok -- no drop path written in the last 30 minutes"

echo "== pre-step: carry dbg25/groupbuild artifact files into the record tree =="
GBREC=$SX/archive/records/campaign-close-20260902/tmp-artifacts/dbg25-groupbuild
if [ -d "$TMP/dbg25/groupbuild" ]; then
  mkdir -p "$GBREC"
  ( cd "$TMP/dbg25/groupbuild" && find . \( -name '*.md5' -o -name '*.tsv' \
      -o -name '*.json' -o -name '*.log' \) -size -4M \
      -exec cp --parents -n {} "$GBREC"/ \; ) 2>/dev/null
  n=$(find "$GBREC" -type f 2>/dev/null | wc -l)
  echo "  carried $n artifact-class files -> $GBREC"
  echo "  (dbg25/ top level -- cfg-live-{before,after}.md5, dump.log, img-*.log,"
  echo "   pr-*.log -- is NOT in the drop list and survives untouched)"
fi

echo "== interlock 3: doc87 lib-knob really is a duplicate of lib-flip =="
if [ -d "$TMP/doc87/lib-knob" ]; then
  d=$(diff <(cd "$TMP/doc87/lib-knob" && md5sum * 2>/dev/null | sort -k2) \
           <(cd "$TMP/doc87/lib-flip" && md5sum * 2>/dev/null | sort -k2) | wc -l)
  [ "$d" -eq 0 ] || { echo "REFUSE: lib-knob differs from lib-flip in $d places"; exit 2; }
  echo "  ok -- 790/790 files md5-identical, lib-flip is kept"
fi

echo
echo "== sweep set =="
TOTAL=0
for p in $PATHS; do
  if [ -e "$p" ]; then
    k=$(du -sk "$p" 2>/dev/null | cut -f1); TOTAL=$((TOTAL+k))
    printf "  %8.2f GiB  %s\n" "$(echo "scale=2;$k/1048576" | bc)" "$p"
  fi
done
printf "  --------\n  %8.2f GiB total\n" "$(echo "scale=2;$TOTAL/1048576" | bc)"

if [ "$CONFIRM" != "yes" ]; then
  echo
  echo "DRY RUN -- nothing deleted."
  echo "To execute:  CONFIRM=yes ./scripts/retire/sweep_tmp_20260902.sh"
  exit 0
fi

echo "== DELETING =="
for p in $PATHS; do
  case "$p" in
    "$TMP"/doc25gate*|"$TMP"/pinlib*|"$TMP"/claude-*|"$TMP"/d97b-libsnap|\
    "$TMP"/doc94c-libsnap|"$TMP"/doc94r3b-libsnap|"$TMP"/prod0901b-libsnap|\
    "$TMP"/doc87/lib-pre|"$TMP"/doc87/lib-tc|"$TMP"/doc87/lib-post|"$TMP"/doc87/lib-flip)
      echo "REFUSE: $p is a keep target"; exit 2;;
  esac
  if [ -e "$p" ]; then
    rm -rf -- "$p"
    echo "  removed $p"
  fi
done
echo
du -sh "$TMP"
