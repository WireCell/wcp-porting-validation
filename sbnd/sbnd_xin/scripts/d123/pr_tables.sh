#!/bin/bash
# doc sbnd_xin/123: the candidate / truth tables of a stage-B arm, with the doc-115 tools unmodified.
#   scripts/d123/pr_tables.sh <pr_root> <products_dir> [cv|nuecc]     (the sample name adds the truth)
# Then r3_pr_compare.py <products_A> <products_B> compares two arms; d107_selection.py <products>
# <figdir> gives the doc-107 efficiency/purity tables on MC.
set -u
cd -P "$(dirname "$0")/../.." || exit 1
SX=$PWD
PR=${1:?usage: pr_tables.sh <pr_root> <products_dir> [cv|nuecc]}; OUT=${2:?}; S=${3:-}
case "$PR" in /*) ;; *) PR=$SX/$PR;; esac
mkdir -p "$OUT"
TRUTH=()
if [ -n "$S" ]; then
    [ -s "$SX/products/d115/$S/truth_base.tsv" ] || { echo "ERROR: no products/d115/$S/truth_base.tsv" >&2; exit 1; }
    TRUTH=(--truth "$SX/products/d115/$S/truth_base.tsv")
fi
python3 "$SX/d115_truth_join.py" --arm "$PR" "${TRUTH[@]}" --out "$OUT" --jobs "${JOBS:-16}" > "$OUT/truth_join.log" 2>&1
rc=$?
echo "pr_tables $PR -> $OUT rc=$rc candidates=$(( $(wc -l < "$OUT/candidates.tsv" 2>/dev/null || echo 1) - 1 ))"
exit $rc
