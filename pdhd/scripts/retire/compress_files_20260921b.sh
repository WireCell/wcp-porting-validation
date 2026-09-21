#!/bin/bash
# Cleanup ROUND G 2026-09-16 (doc sbnd_xin/111 sec 17) -- compression driver.  DRY RUN unless CONFIRM=yes.
#
#   ./compress_files_20260921b.sh                 # dry run: record gate only
#   CONFIRM=yes ./compress_files_20260921b.sh     # re-plan (rc 10/11), record gate (rc 14), compress+verify
#
# THE ORDER IS NOT OPTIONAL:
#     1. python3 plan_compress_20260921b.py --hash   gates C1-C5 + SHA-256 manifest (the freeze, M13)
#     2. this driver, CONFIRM=yes                    each file -> .zst, original removed only after its
#                                                    decompressed SHA-256 equals the manifest
# Undo: python3 restore_compress_20260921b.py --confirm <family ...> | --all
set -u
D="$(cd "$(dirname "$0")" && pwd)"
R=/home/xqian/toolkit-dev/wcp-porting-img
STAMP=20260921b
TREES="pdvd pdhd"
RECD="${MAN_OVERRIDE:-$R/sbnd/sbnd_xin/archive/records/cleanup-${STAMP}}"
for t in $TREES; do
  [ -f "$D/files_compress_${t}_${STAMP}.txt" ] || { echo "REFUSING: files_compress_${t}_${STAMP}.txt missing -- run plan_compress_${STAMP}.py first"; exit 2; }
  [ -s "$RECD/${t}-compress.manifest.tsv" ]    || { echo "REFUSING: $RECD/${t}-compress.manifest.tsv missing -- run plan_compress_${STAMP}.py --hash first"; exit 6; }
done

if [ "${CONFIRM:-no}" = yes ] && [ "${REPLAN:-yes}" = yes ]; then
  echo "== INTERLOCK A: re-planning before compressing"
  if ! PLAN_SUFFIX=confirm python3 "$D/plan_compress_${STAMP}.py" > "$D/plan_compress_${STAMP}.confirm.out" 2>&1; then
    echo "   REFUSING: the plan no longer passes its gates ($D/plan_compress_${STAMP}.confirm.out)"; exit 10
  fi
  for t in $TREES; do
    if ! cmp -s "$D/files_compress_${t}_${STAMP}.txt" "$D/files_compress_${t}_${STAMP}.confirm.txt"; then
      echo "   REFUSING: the $t target list moved between plan and confirm"; exit 11
    fi
  done
  echo "   OK: gates pass, target list unchanged"
fi

rc=0
for t in $TREES; do
  L="$D/files_compress_${t}_${STAMP}.txt"
  n=$(wc -l < "$L")
  [ "$n" -gt 0 ] || { echo "== $t: 0 files in scope, skipped"; continue; }
  echo "== $t/work cold-file compression: $n files"
  python3 "$D/compress_worker_${STAMP}.py" "$L" "$RECD/${t}-compress.manifest.tsv"
  r=$?; echo "   $t worker rc=$r"; [ $r -eq 0 ] || rc=$r
done
if [ "${CONFIRM:-no}" = yes ]; then
  for t in $TREES; do
    L="$D/files_compress_${t}_${STAMP}.txt"
    left=$(while read -r p; do [ -e "$p" ] && echo x; done < "$L" | wc -l)
    zst=$(while read -r p; do [ -e "$p.zst" ] && echo x; done < "$L" | wc -l)
    echo "POST-STATE $t: originals left $left | .zst present $zst | broken symlinks $(find $R/$t -xtype l 2>/dev/null | wc -l) | $t/work $(du -sh $R/$t/work 2>/dev/null | cut -f1)"
  done
  echo "            free $(df -h /home/xqian | awk 'NR==2{print $4}')"
  echo "  UNDO: python3 $D/restore_compress_${STAMP}.py --confirm <family ...> | --all"
else
  echo "DRY RUN.  Re-run with CONFIRM=yes to execute."
fi
exit $rc
