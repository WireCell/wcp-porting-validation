#!/bin/bash
# doc sbnd_xin/pr/144 -- re-point the two pi0 denominator manifests at a d144 arm.
#
# pr141_pi0_census2.py reads each event's calib dump from the manifest's own
# `dump` column (work-<something>-<sample>/pr_evt<ID>/calib-pr-evt<ID>.json), so
# running the census on a new arm means rewriting that column -- the denominator
# (which events, which hand labels) is untouched, only where the reco is read from.
#
# Usage: ./scripts/pr144_pi0_manifests.sh <tag>      e.g. d144off | d144on
# Writes /home/xqian/tmp/d144/manifests/<tag>/{denom98,denom141}.tsv
set -eu
TAG=${1:?usage: pr144_pi0_manifests.sh <tag>}
SX=$(cd "$(dirname "$0")/.." && pwd)
OUT=/home/xqian/tmp/d144/manifests/$TAG
mkdir -p "$OUT"
cd "$SX"

rewrite() {  # <src manifest> <dst>
  awk -F'\t' -v OFS='\t' -v tag="$TAG" '
    NR==1 { print; next }
    { n=split($5,p,"/"); $5 = "work-" $1 "-" tag "/" p[2] "/" p[3]; print }
  ' "$1" > "$2"
}
rewrite em_display/em117-132denom98-manifest.tsv  "$OUT/denom98.tsv"
rewrite em_display/em114c-132denom141-manifest.tsv "$OUT/denom141.tsv"

miss=0
while IFS=$'\t' read -r s r sr e d; do
  [ "$s" = sample ] && continue
  [ -f "$d" ] || { miss=$((miss+1)); [ $miss -le 5 ] && echo "  MISSING $d" >&2; }
done < <(cat "$OUT/denom98.tsv" "$OUT/denom141.tsv")
echo "$TAG: wrote $OUT/{denom98,denom141}.tsv ; missing dumps: $miss"
[ "$miss" -eq 0 ] || echo "  (a missing dump means that event produced no calib dump on this arm -- the census counts it as absent-on-arm)"
