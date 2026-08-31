#!/bin/bash
# doc pr/130 -- produce the arm the sentinel registry needs.
#
# pr127_sentinels.py locates an event by scanning arm roots for pr_evt<N>/.
# Twelve target events of shipped, SBND-ON fixes are in NO standard manifest
# (docs/pr/pr130-sentinel-manifest.tsv), so every one of them reported SKIP --
# which reads like a pass.  This runs exactly those events at the current
# production point into a FRESH arm (M13: never reuse a label).
#
#   ./scripts/pr130_sentinel_arm.sh [tag]      # default tag: 130
#
# Then:  ./scripts/pr127_sentinels.py work-sent<tag>-mcp1k work-sent<tag>-mcp2k <other arms...>
set -u
TAG=${1:-130}
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
MAN=docs/pr/pr130-sentinel-manifest.tsv
export PR_EXTRA_STAGES=pr_display PR_JOBS=${PR_JOBS:-6}
rc_all=0
for s in mcp1k mcp2k; do
  ev=$(awk -F'\t' -v s="$s" '!/^#/ && $1==s {printf "%s ", $2}' "$MAN")
  ql=$(awk -F'\t' -v s="$s" '!/^#/ && $1==s {print $3; exit}' "$MAN")
  [ -z "$ev" ] && continue
  echo "arm sent$TAG-$s: $ev"
  ./run_pr_chain_batch.sh "$ql" "work-sent$TAG-$s" data $ev \
      > /home/xqian/tmp/pr130_sent_${TAG}_$s.log 2>&1
  rc=$?; echo "  rc=$rc"; [ $rc -ne 0 ] && rc_all=$rc
done
exit $rc_all
