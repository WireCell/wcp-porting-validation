#!/bin/bash
# doc sbnd_xin/pr/150 -- regenerate the pr/148 A5 census on a pr150 cell and
# re-score the three committed PID hand scans against it.
#
# The census SCRIPT is the committed scripts/pr148_a5_census.py, unforked: it
# takes --arm/--tsv and reads only things every pr150 arm writes --
# wct_pr_evt<ID>.log (the "A5 hadronic census:" / "A5 hadronic retype:" lines),
# calib-pr-evt<ID>.json (the geometry join) and .wct-cfg-evt<ID>.json (the real
# mip_dqdx_median, which the dump's meta misreports).  So a new arm IS its own
# calibration sample and no reprocessing is needed.
#
# The SCORER is forked (CLAUDE.md M10) into scripts/analysis/pr150/pid_agreement.py:
# pr148_score_scan.py keys verdicts on (event, shower_id) with no sample and then
# indexes the census unconditionally, which raises KeyError the moment an object
# is absent on the new arm -- it cannot be pointed at one as written.  The fork
# adds the identity probe and the absent counts.
#
# Only wct_pr_evt<ID>.log is read, never *.log: stdout.log in the same directory
# carries the identical lines and a glob double-counts every shower (pr148_a5_census
# docstring).  pr148_a5_census.py already opens the right file.
#
# Usage:  ./scripts/analysis/pr150/pid_scan_score.sh <cell> [sample ...]
#         TAG=d102mpr ./scripts/analysis/pr150/pid_scan_score.sh
# Writes /home/xqian/tmp/pr150/scorers/{a5census-<cell>.tsv, pid_<cell>.txt}
set -u
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
OUT=/home/xqian/tmp/pr150/scorers
mkdir -p "$OUT" || exit 2
cd "$SX" || exit 2

CELL=${1:-}
TAG=${TAG:-}
if [ -z "$TAG" ]; then
  [ -n "$CELL" ] || { echo "usage: pid_scan_score.sh <cell> [samples...]" >&2; exit 2; }
  TAG="pr150$CELL"; shift
else
  CELL=$TAG
fi
SAMPLES=${*:-"nuecc48 ncpi0 mcp1k mcp2k"}
CENSUS=$OUT/a5census-$CELL.tsv
REPORT=$OUT/pid_$CELL.txt

{
echo "=== pr150 PID scan re-score :: arm tag $TAG  ($(date +%F_%H:%M:%S))"
ARMS=""
for s in $SAMPLES; do
  d="work-$s-$TAG"
  if [ -d "$d" ]; then
    n=$(ls -d "$d"/pr_evt* 2>/dev/null | wc -l)
    g=$(grep -l 'A5 hadronic census:' "$d"/pr_evt*/wct_pr_evt*.log 2>/dev/null | wc -l)
    echo "  arm $d: $n event(s), $g carry an 'A5 hadronic census:' line"
    ARMS="$ARMS --arm $d"
  else
    echo "  arm $d: ABSENT -- this sample's objects cannot be re-scored on this cell"
  fi
done
[ -n "$ARMS" ] || { echo "no arm found for tag $TAG"; exit 2; }

echo
echo "--- 1. regenerate the A5 census (pr148_a5_census.py, unforked) ---"
python3 scripts/pr148_a5_census.py $ARMS --tsv "$CENSUS"
echo "    rc=$?"

echo
echo "--- 2. identity probe + agreement table (pid_agreement.py) ---"
python3 scripts/analysis/pr150/pid_agreement.py --census "$CENSUS" --label "$CELL"
echo "    rc=$?"

echo
echo "--- 3. doc pr/148 sec 12 tables, unforked scorer against this census ---"
echo "    pr148_scan_combined.py indexes byk[k] unguarded, so it ABORTS (KeyError)"
echo "    as soon as a labelled object is missing on this arm.  That abort is a"
echo "    result, not a bug: read step 2's absent counts for which objects."
python3 scripts/pr148_scan_combined.py --census "$CENSUS" 2>&1 | head -12
echo "    rc=${PIPESTATUS[0]}"
echo
echo "=== done $(date +%F_%H:%M:%S)"
} 2>&1 | tee "$REPORT"
echo "wrote $REPORT"
