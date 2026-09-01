#!/bin/bash
# doc 87: prove nusel_extract.py's two in-scope sources are interchangeable.
#
# --prbee  reads the set IMPLICITLY, from which clusters mabc-pr.zip contains.
# --prroot reads it EXPLICITLY, from tracking-pr.root's T_cluster (save_in_scope).
#
# If the two ever disagreed, turning the Bee zip off would silently change the
# nusel table -- the exact failure this whole knob exists to avoid.  So the bar
# is byte-identical TSV, not "same labels".
#
# usage: pr87_nusel_source_gate.sh <pr_arm> <ql_root> [<pr_arm> <ql_root> ...]
set -u
SX=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d "${TMPDIR:-/home/xqian/tmp}/pr87nusel.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
ok=0; bad=0; skip=0
while [ $# -ge 2 ]; do
  ARM=$1; QLROOT=$2; shift 2
  for d in "$ARM"/pr_evt*; do
    [ -d "$d" ] || continue
    ID=$(basename "$d"); ID=${ID#pr_evt}
    QL="$QLROOT/ql_evt$ID"
    [ -f "$d/tracking-pr.root" ] && [ -f "$d/mabc-pr.zip" ] || { skip=$((skip+1)); continue; }
    for mode in bee root; do
      if [ "$mode" = bee ]; then SRC=(--prbee "$d/mabc-pr.zip")
      else SRC=(--prroot "$d/tracking-pr.root"); fi
      python3 "$SX/nusel_extract.py" --pctree "$QL/pctree-evt$ID.tar.gz" "${SRC[@]}" \
        --prlog "$d/wct_pr_evt$ID.log" --prtree "$d/pctree-pr-evt$ID.tar.gz" \
        --qlbee "$QL/mabc-all-apa.zip" --beam-window "0.2,2.2" \
        --run 0 --subrun 0 --out "$TMP/$ID.$mode.tsv" 2>/dev/null
    done
    if cmp -s "$TMP/$ID.bee.tsv" "$TMP/$ID.root.tsv"; then ok=$((ok+1))
    else bad=$((bad+1)); echo "  DIFFER evt $ID"; fi
  done
done
echo "# events where --prbee and --prroot give a byte-identical nusel tsv: $ok"
echo "# differing: $bad   skipped(no root/bee): $skip"
[ "$bad" = 0 ] && { echo PASS; exit 0; } || { echo FAIL; exit 1; }
