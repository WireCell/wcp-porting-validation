#!/bin/bash
# doc 89 Phase 4b -- release closed-round library snapshots from ~/tmp.
#
# Vetted, like retire_20260901.sh, because the permission gate declines ad-hoc
# `rm -rf` and should (doc pr/135 sec 11.1 recorded the same refusal).
#
# WHAT THIS RELEASES, AND THE ONE CONDITION IT DEPENDS ON.  Each pin-* dir is
# the binary a CLOSED pr-round's arms were produced with.  A pinned binary with
# no surviving arms reproduces nothing, so these go WITH the arms -- which is
# why this script REFUSES to run until the retire round has actually removed
# them (interlock A).  Running it first would delete the only thing that could
# explain a surprise in an arm still on disk.
#
# WHAT IT DELIBERATELY KEEPS, each for a stated reason:
#   prod0901b-libsnap   the binary THIS round's production arm ran under
#   prod0901b-cfgsnap   the cfg tree it ran under (prod_cfg_gate PASS 21/21)
#   pr142-libsnap       the binary prod0901 ran under -- the ONLY thing that
#                       can explain a doc 89 sec 6 successor-gate diff
#   doc87/lib-{pre,tc,post,flip}   four DISTINCT binaries backing the doc 87
#                       gate arms this round refuses to release (sec 5).
#                       lib-knob was an exact md5 duplicate of lib-flip on all
#                       four libs and is the only one already dropped.
#   doc87/*.txt|json|log  ALREADY COPIED into
#                       archive/records/campaign-close-20260901/doc87-gates/
#                       (95 files) -- they are the "report gates by label"
#                       artifacts and must outlive the scratch dir.
#   pr138_glcolor       a Bokeh viewer is serving out of it on :5032
#   claude-*            live session scratchpads
#
# NOTE ON THE PROJECTED SAVING.  The round doc estimated ~28 G here.  The
# measured figure is ~18 G, because four of the five doc-87 snapshots turned
# out to be DISTINCT binaries rather than copies (md5 on Clus/Root/Gen/Aux) and
# back arms that are being kept.  The smaller honest number is reported rather
# than the projection.
#
#   ./sweep_tmp_20260901.sh              # dry run (default)
#   CONFIRM=yes ./sweep_tmp_20260901.sh  # actually delete
set -u
TMP=$HOME/tmp
SBND=/home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
CONFIRM=${CONFIRM:-no}

# pin-* dirs, each named with the round whose arms it backs
DROP="pin-pr138bare pin-pr138off pin-pr139 pin-pr139b pin-pr140r2 pin-pr140r3
      pin-pr140r4 pin-pr141 pin-pr141b pin-pr141c prod0830-libsnap"

# ---- interlock A: the arms these binaries back must ALREADY be gone ---------
echo "== interlock A: the backed arms must already be retired =="
bad=0
for d in $DROP; do
    case "$d" in
        pin-pr*)  pat="work-${d#pin-}" ;;
        prod0830-libsnap) pat="work-*-prod0830" ;;
        *) pat="" ;;
    esac
    [ -z "$pat" ] && continue
    n=$(find "$SBND" -maxdepth 1 -type d -name "${pat}*" 2>/dev/null | wc -l)
    if [ "$n" -ne 0 ]; then
        echo "  !! $d: $n arm(s) matching ${pat}* are STILL ON DISK -- refusing"
        bad=$((bad+1))
    fi
done
if [ "$bad" -ne 0 ]; then
    echo "refusing -- run the retire round first (a pinned binary goes WITH its arms)"
    exit 2
fi
echo "  all backed arms are gone; their binaries reproduce nothing"

# ---- interlock B: nothing live is using a drop target ----------------------
echo "== interlock B: no live process names a drop target =="
live=$(pgrep -a -f "$TMP" 2>/dev/null | grep -v 'pgrep\|grep -\|sweep_tmp' || true)
for d in $DROP; do
    case "$live" in *"$d"*) echo "  !! a live process names $d -- refusing"; exit 2;; esac
done
echo "  clear"

# ---- interlock C: the doc-87 gate artifacts are preserved off-scratch ------
echo "== interlock C: doc 87 gate artifacts preserved in the record layer =="
G=$SBND/archive/records/campaign-close-20260901/doc87-gates
n=$(ls "$G" 2>/dev/null | wc -l)
if [ "$n" -lt 90 ]; then
    echo "  !! $G has $n files, expected >= 90 -- preserve them before sweeping"
    exit 2
fi
echo "  $n files preserved"

echo
before=$(du -sm "$TMP" | cut -f1)
tot=0
for d in $DROP; do
    p=$TMP/$d
    [ -d "$p" ] || { echo "SKIP (already gone): $d"; continue; }
    sz=$(du -sm "$p" | cut -f1); tot=$((tot+sz))
    if [ "$CONFIRM" = yes ]; then
        rm -rf "$p" && echo "removed  $d  (${sz} MB)" || echo "!! rm FAILED: $d"
    else
        echo "would remove  $d  (${sz} MB)"
    fi
done
echo
echo "total ${tot} MB   CONFIRM=$CONFIRM"
[ "$CONFIRM" = yes ] || { echo; echo "dry run only -- re-run with CONFIRM=yes"; exit 0; }
echo "~/tmp: ${before} MB -> $(du -sm "$TMP" | cut -f1) MB"
