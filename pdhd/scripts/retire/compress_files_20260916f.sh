#!/bin/bash
# Cleanup ROUND G 2026-09-16 (doc sbnd_xin/111 sec 17) -- compression driver.  DRY RUN unless CONFIRM=yes.
#
#   ./compress_files_20260916f.sh                 # dry run: record gate only
#   CONFIRM=yes ./compress_files_20260916f.sh     # re-plan (rc 10/11), record gate (rc 14), compress+verify
#
# THE ORDER IS NOT OPTIONAL:
#     1. python3 plan_compress_20260916f.py --hash   gates C1-C5 + SHA-256 manifest (the freeze, M13)
#     2. this driver, CONFIRM=yes                    each file -> .zst, original removed only after its
#                                                    decompressed SHA-256 equals the manifest
# Undo: python3 restore_compress_20260916f.py --confirm <family ...> | --all
set -u
D="$(cd "$(dirname "$0")" && pwd)"
R=/home/xqian/toolkit-dev/wcp-porting-img
STAMP=20260916f
LIST="$D/files_compress_pdvd_${STAMP}.txt"
MAN="${MAN_OVERRIDE:-$R/sbnd/sbnd_xin/archive/records/cleanup-${STAMP}/pdvd-compress.manifest.tsv}"
[ -s "$LIST" ] || { echo "REFUSING: $LIST missing -- run plan_compress_${STAMP}.py first"; exit 2; }
[ -s "$MAN" ]  || { echo "REFUSING: $MAN missing -- run plan_compress_${STAMP}.py --hash first"; exit 6; }

if [ "${CONFIRM:-no}" = yes ] && [ "${REPLAN:-yes}" = yes ]; then
  echo "== INTERLOCK A: re-planning before compressing"
  if ! PLAN_SUFFIX=confirm python3 "$D/plan_compress_${STAMP}.py" > "$D/plan_compress_${STAMP}.confirm.out" 2>&1; then
    echo "   REFUSING: the plan no longer passes its gates ($D/plan_compress_${STAMP}.confirm.out)"; exit 10
  fi
  if ! cmp -s "$LIST" "$D/files_compress_pdvd_${STAMP}.confirm.txt"; then
    echo "   REFUSING: the target list moved between plan and confirm"; exit 11
  fi
  echo "   OK: gates pass, target list unchanged"
fi

echo "== pdvd/work cold-file compression: $(wc -l < "$LIST") files"
python3 "$D/compress_worker_${STAMP}.py" "$LIST" "$MAN"
rc=$?
echo "   worker rc=$rc"
if [ "${CONFIRM:-no}" = yes ]; then
  left=$(while read -r p; do [ -e "$p" ] && echo x; done < "$LIST" | wc -l)
  zst=$(while read -r p; do [ -e "$p.zst" ] && echo x; done < "$LIST" | wc -l)
  echo "POST-STATE: originals left $left | .zst present $zst | broken symlinks pdvd $(find $R/pdvd -xtype l 2>/dev/null | wc -l)"
  echo "            free $(df -h /home/xqian | awk 'NR==2{print $4}') | pdvd/work $(du -sh $R/pdvd/work 2>/dev/null | cut -f1)"
else
  echo "DRY RUN.  Re-run with CONFIRM=yes to execute."
fi
exit $rc
