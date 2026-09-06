#!/bin/bash
# Cleanup round 2026-09-05 -- deletion driver.  DRY RUN unless CONFIRM=yes.
#   ./retire_20260905.sh pdhd|pdvd|sbnd|all
#
# THE TRAP THIS GUARDS (doc 100 catch 2): the 08-31 driver built its tier
# filename by interpolation, a literal rename missed it, and the first dry run
# silently targeted the PREVIOUS round's already-deleted list and reported
# dirs=0.  So this script REFUSES to run unless the tier file exists, is
# non-empty, every line still exists on disk, and the counts it reports match
# what plan_20260905.py printed.  Always compare the two before CONFIRM=yes.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
CONFIRM=${CONFIRM:-no}
TREES=${1:-all}; [ "$TREES" = all ] && TREES="pdhd pdvd sbnd"

for t in $TREES; do
  TF="$D/tier_${t}_20260905.txt"
  echo "=================================================================="
  echo "== $t   tier file: $TF"
  if [ ! -s "$TF" ]; then echo "   REFUSING: tier file missing or empty. Run plan_20260905.py $t first."; exit 2; fi
  n=$(wc -l < "$TF"); miss=0; kb=0
  while read -r p; do
    if [ -e "$p" ]; then kb=$((kb + $(du -sk "$p" | cut -f1))); else miss=$((miss+1)); fi
  done < "$TF"
  echo "   lines $n | present $((n-miss)) | already gone $miss | $(echo "scale=2; $kb/1048576" | bc) GiB"
  if [ "$miss" -gt 0 ]; then
    echo "   REFUSING: $miss of $n targets are already gone -- this is the"
    echo "   signature of pointing at a PREVIOUS round's tier list. Re-plan."
    exit 3
  fi
  # never delete through a symlink, and never outside the intended tree
  bad=$(while read -r p; do [ -L "$p" ] && echo "$p"; done < "$TF")
  if [ -n "$bad" ]; then echo "   REFUSING: symlink in the target list:"; echo "$bad"; exit 4; fi
  out=$(grep -cv '^/home/xqian/toolkit-dev/wcp-porting-img/' "$TF" || true)
  if [ "$out" != 0 ]; then echo "   REFUSING: $out targets outside wcp-porting-img"; exit 5; fi

  if [ "$CONFIRM" = yes ]; then
    echo "   DELETING..."
    xargs -a "$TF" -d '\n' rm -rf
    echo "   done, rc=$?"
  else
    echo "   DRY RUN -- first 3 targets:"; head -3 "$TF" | sed 's/^/      /'
  fi
done

echo
echo "POST-STATE CHECK (interlock 4 recorded 0 pre-existing broken symlinks in"
echo "all three trees, which is the only reason this number means anything):"
for t in pdhd pdvd sbnd/sbnd_xin; do
  echo "   $t: $(find /home/xqian/toolkit-dev/wcp-porting-img/$t -xtype l 2>/dev/null | wc -l) broken symlinks"
done
[ "$CONFIRM" = yes ] || echo -e "\nDRY RUN.  Re-run with CONFIRM=yes to execute."
