#!/bin/bash
# doc sbnd_xin/pr/150 -- population census A/B between two pr150 arm cells.
#
# FORK BY DUPLICATION (CLAUDE.md M10) of the `scores` + `ab` steps of
# scripts/pr142_analyze.sh and scripts/pr144_analyze.sh.  Those two scripts are
# the records of docs pr/142 and pr/144 and stay byte-for-byte untouched; what
# this fork changes:
#   * arm naming work-<sample>-pr150<cell> (pr142/pr144 hardcode their own tags);
#   * every product goes to /home/xqian/tmp/pr150/scorers/, never into
#     products/ or docs/pr/ (M13: those hold committed records);
#   * it works on WHATEVER events exist -- the mcp1k/mcp2k pr150 arms are still
#     being produced -- and prints the per-sample event count of each cell and
#     the joinable intersection BEFORE the compare, and skips (loudly) a sample
#     that is missing on either side rather than feeding a lopsided pair in.
#
# Joins: pr142_campaign_ab.py keys on (sample, run, subrun, event), so the
# "sentinel event ids collide across samples" trap (18255-1-69314 is in both
# mcp2k and nuecc48) is already handled upstream -- nothing to do here.
# pr_scores_table.py's T_tagger row is NOT joinable to a nusel main_id; this
# wrapper never tries (see that script's docstring).
#
# Usage:  ./scripts/analysis/pr150/campaign_ab.sh <cellA> <cellB> [sample ...]
#         (default samples: nuecc48 ncpi0 mcp1k mcp2k -- missing ones are skipped)
# e.g.    ./scripts/analysis/pr150/campaign_ab.sh s0 csp3bw nuecc48 ncpi0
set -u
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
OUT=/home/xqian/tmp/pr150/scorers
mkdir -p "$OUT" || exit 2
cd "$SX" || exit 2

A=${1:?usage: campaign_ab.sh <cellA> <cellB> [samples...]}
B=${2:?usage: campaign_ab.sh <cellA> <cellB> [samples...]}
shift 2
SAMPLES=${*:-"nuecc48 ncpi0 mcp1k mcp2k"}

# One score table per (cell, sample).  Rebuild unless the cached table has
# exactly one row per pr_evt dir -- a killed pr_scores_table.py leaves a SHORT
# file behind and `[ -f ]` alone reads out later as "events only in the other
# arm" (pr142_analyze.sh, 2026-09-01: 966 rows of 1000).
build() {  # <cell> <sample> -> echoes the row count, or "-" if the arm is absent
  local cell=$1 s=$2 arm="work-$2-pr150$1" tsv="$OUT/scores-$1-$2.tsv"
  [ -d "$arm" ] || { echo "-"; return 1; }
  local want have
  want=$(ls -d "$arm"/pr_evt* 2>/dev/null | wc -l)
  have=$(( $([ -f "$tsv" ] && wc -l < "$tsv" || echo 1) - 1 ))
  if [ "$want" -le 0 ]; then echo "0"; return 1; fi
  if [ "$have" -ne "$want" ]; then
    echo "  building $tsv ($have of $want rows cached)" >&2
    nice -n 10 python3 pr_scores_table.py --root "$arm" --sample "$s" --out "$tsv" \
      > "$OUT/scores-$1-$2.log" 2>&1
    local rc=$?
    have=$(( $([ -f "$tsv" ] && wc -l < "$tsv" || echo 1) - 1 ))
    [ "$rc" = 0 ] || echo "  WARNING: pr_scores_table rc=$rc (see $OUT/scores-$1-$2.log)" >&2
  fi
  echo "$have"
}

echo "=== pr150 campaign A/B: A=$A  B=$B  samples=[$SAMPLES]  $(date +%F_%H:%M:%S)"
USE=""
for s in $SAMPLES; do
  na=$(build "$A" "$s"); nb=$(build "$B" "$s")
  ea=$(ls -d "work-$s-pr150$A"/pr_evt* 2>/dev/null | wc -l)
  eb=$(ls -d "work-$s-pr150$B"/pr_evt* 2>/dev/null | wc -l)
  if [ "$na" = "-" ] || [ "$nb" = "-" ] || [ "$na" = 0 ] || [ "$nb" = 0 ]; then
    echo "  SKIP $s: A evt=$ea rows=$na | B evt=$eb rows=$nb  (arm absent or empty on one side)"
    continue
  fi
  # joinable intersection on (run, subrun, event) within this sample
  read -r ni noa nob < <(python3 - "$OUT/scores-$A-$s.tsv" "$OUT/scores-$B-$s.tsv" <<'PY'
import csv, sys
def keys(p):
    return {(r["run"], r["subrun"], r["event"])
            for r in csv.DictReader(open(p), delimiter="\t")}
a, b = keys(sys.argv[1]), keys(sys.argv[2])
print("%d %d %d" % (len(a & b), len(a - b), len(b - a)))
PY
)
  # pr142_campaign_ab.py's section-2 distributions are computed per arm over ALL
  # its rows, not over the join, so a half-produced arm on one side reads as a
  # physics shift.  The mcp1k/mcp2k pr150 arms are still being produced, so a
  # lopsided pair is skipped unless ALLOW_PARTIAL=1 asks for it.
  big=$(( na > nb ? na : nb ))
  if [ "${ALLOW_PARTIAL:-0}" != 1 ] && [ $(( ni * 100 )) -lt $(( big * 90 )) ]; then
    echo "  SKIP $s: A evt=$ea rows=$na | B evt=$eb rows=$nb | common $ni (A-only $noa, B-only $nob)"
    echo "        -> only $ni of $big rows join; one arm is still producing."
    echo "        -> ALLOW_PARTIAL=1 to include it anyway (own-arm distributions"
    echo "           would then be taken over different event sets)."
    continue
  fi
  echo "  USE  $s: A evt=$ea rows=$na | B evt=$eb rows=$nb | common $ni (A-only $noa, B-only $nob)"
  USE="$USE $s"
done
[ -n "$USE" ] || { echo "no sample usable on both cells -- nothing to compare"; exit 2; }

TA=""; TB=""
for s in $USE; do TA="$TA $OUT/scores-$A-$s.tsv"; TB="$TB $OUT/scores-$B-$s.tsv"; done
LOG=$OUT/ab-$A-vs-$B.txt
echo "=== compare (pr142_campaign_ab.py) -> $LOG"
nice -n 10 python3 scripts/pr142_campaign_ab.py \
    --a $TA --b $TB --label-a "$A" --label-b "$B" \
    --movers-tsv "$OUT/movers-$A-$B.tsv" \
    --summary-tsv "$OUT/population-$A-$B.tsv" > "$LOG" 2>&1
rc=$?
echo "  rc=$rc  movers=$OUT/movers-$A-$B.tsv  summary=$OUT/population-$A-$B.tsv"
cat "$LOG"
echo "=== done $(date +%F_%H:%M:%S)"
exit $rc
