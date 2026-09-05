#!/bin/bash
# doc 100 sec 6.2 -- ~/tmp sweep, PASS 3 (2026-09-04, owner-selected: doc87 libsnaps).
#   ./scripts/retire/sweep_tmp_20260904c.sh              # dry run
#   CONFIRM=yes ./scripts/retire/sweep_tmp_20260904c.sh  # delete
#
# doc87/lib-{flip,post,pre,tc} -- four 1.9 GiB pins from doc 87.
# CHECKED FOR DUPLICATION FIRST and they are NOT duplicates: libWireCellClus.so
# md5 matches across lib-pre/lib-post/lib-tc, but the FULL directory rollups
# (14007 files each) all differ, so a dedupe-to-symlinks would have been wrong.
# Four distinct binaries, regenerable from the commits doc 87 records.
#
# COST, STATED: doc 87's ARMS stay PROTECTED and untouched, but they become
# re-readable rather than re-runnable against the exact binary that made them --
# the same trade the owner took for pinlib2..7 in pass 2.  doc 87 sec 6.1/6.2's
# tables are unaffected; they live in the doc and in the arms.
set -u
TMP=/home/xqian/tmp
CONFIRM=${CONFIRM:-no}
LIVE=$(ps -u "$(id -un)" -o args= | grep -oE 'claude --resume [0-9a-f]{8}-[0-9a-f-]+' | awk '{print $3}' | sort -u)
echo "== live claude sessions (scratchpads protected) =="; echo "${LIVE:-  (none)}" | sed 's/^/  /'
DROP=""
for d in doc87/lib-flip doc87/lib-post doc87/lib-pre doc87/lib-tc; do
  [ -d "$TMP/$d" ] && DROP="$DROP $d"
done
# interlock: doc 87's ARMS must still be on disk -- this sweep drops pins, never arms
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
for a in work-87knob-min-ncpi0 work-87knob-sup-ncpi0 work-87flip-ncpi0 work-87grp-def-ncpi0; do
  [ -d "$SX/$a" ] || { echo "REFUSE: doc 87 arm $a is missing -- drop pins only when the arms are intact"; exit 2; }
done
echo "== interlock: doc 87's four reference arms present =="
# interlock: never touch the text record layer
for d in $DROP; do
  case "$d" in *.log|*.txt|*.md|*.json|*.zip) echo "REFUSE: $d is a record"; exit 2;; esac
  [ -d "$TMP/$d" ] || { echo "REFUSE: $TMP/$d is not a directory"; exit 2; }
done
echo; echo "== sweep set =="
for d in $DROP; do printf "  %-18s %s\n" "$d" "$(du -sh "$TMP/$d" 2>/dev/null|cut -f1)"; done
TOT=$(du -sc --block-size=1 $(for d in $DROP; do echo "$TMP/$d"; done) 2>/dev/null|tail -1|cut -f1)
if [ "$CONFIRM" != "yes" ]; then
  echo; printf "DRY RUN -- would free %.2f GiB from %d dirs.\n" "$(echo "$TOT/1073741824"|bc -l)" "$(echo $DROP|wc -w)"
  echo "To execute:  CONFIRM=yes ./scripts/retire/sweep_tmp_20260904c.sh"; exit 0
fi
echo "== DELETING =="
for d in $DROP; do rm -rf -- "$TMP/$d" && echo "  removed $d"; done
echo; echo "doc87 now: $(du -sh $TMP/doc87 2>/dev/null|cut -f1)   ~/tmp now: $(du -sh $TMP|cut -f1)"
