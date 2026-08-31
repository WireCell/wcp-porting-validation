#!/bin/bash
# doc pr/136 round 2 -- the full adjudication of one ON arm against the gated
# f086 OFF point.  Runs the three pre-registered kill instruments (doc sec 11.2)
# plus the escape attribution.  READ-ONLY apart from the docs/pr TSVs it writes.
#
#   pr136_r2_analyze.sh <TAG>          e.g. onV1, onV1c90, off1
#
# For off1 it additionally prints the byte-identity gate against the probe arm.
set -u
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
TAG=$1
./scripts/pr136_manifests.sh "$TAG"
mkdir -p em_display/emprep-136$TAG
python3 em_display/prep_em_scan.py --prepdir em_display/emprep-136$TAG \
    --out /home/xqian/tmp/pr136-parse-$TAG.tsv --no-bee-index \
    --parse-probes work-pr136-$TAG-mcp1k work-pr136-$TAG-mcp2k \
                   work-pr136-$TAG-ncpi0 work-pr136-$TAG-nuecc48 \
    > /home/xqian/tmp/pr136_parse_$TAG.log 2>&1
echo "parse-probes rc=$?  sidecars=$(ls em_display/emprep-136$TAG | wc -l)"

if [ "$TAG" = "off1" ]; then
  echo "=== BYTE-IDENTITY GATE vs work-pr136-f086probe-* ==="
  for s in mcp1k mcp2k ncpi0 nuecc48; do
    python3 scripts/pr85_hash_gate.py work-pr136-f086probe-$s work-pr136-off1-$s \
        > /home/xqian/tmp/pr136_r2gate_$s.log 2>&1
    echo "  $s rc=$?  $(tail -1 /home/xqian/tmp/pr136_r2gate_$s.log)"
  done
fi

echo "=== 1. mass closure (OFF = classes 11/37/8, R>1 class median 1.98, 19/56 impossible) ==="
scripts/pr136_mass_closure.py --manifest98 em117-136${TAG}98-manifest.tsv \
    --manifest141 em114c-136${TAG}141-manifest.tsv --overlay-tag pi0scan-0829-agent \
    --fudge 0.86 --tsv docs/pr/pr136-mass-closure-$TAG.tsv \
    > /home/xqian/tmp/pr136_mc_$TAG.log 2>&1
echo "  rc=$?"; sed -n '/class /,/^$/p' /home/xqian/tmp/pr136_mc_$TAG.log | head -6
grep -E "R < 1|outside the" /home/xqian/tmp/pr136_mc_$TAG.log | head -3

echo "=== 2. hand-scan attribution (joins the closure TSV written above) ==="
( cd em_display
  ./em117_score.py --tag emscan-0827 --manifest em117-136${TAG}98-manifest.tsv \
      --prepdir emprep-136$TAG --tsv ../docs/pr/pr136-completeness-$TAG-98.tsv \
      > /home/xqian/tmp/pr136_score98_$TAG.log 2>&1; echo "  98 rc=$?"
  ./em117_score.py --tag emscan-0828-agent5 --manifest em114c-136${TAG}141-manifest.tsv \
      --prepdir emprep-136$TAG --tsv ../docs/pr/pr136-completeness-$TAG-141.tsv \
      > /home/xqian/tmp/pr136_score141_$TAG.log 2>&1; echo "  141 rc=$?" )
scripts/pr136_completeness.py --src98 pr136-completeness-$TAG-98.tsv \
    --src141 pr136-completeness-$TAG-141.tsv \
    --closure docs/pr/pr136-mass-closure-$TAG.tsv \
    --tsv docs/pr/pr136-completeness-$TAG.tsv 2>/dev/null | sed -n '1,20p'

echo "=== 3. pi0 census exact (OFF = 32/66 = 48.5%) ==="
scripts/pr132_pi0_census.py --manifest98 em117-136${TAG}98-manifest.tsv \
    --manifest141 em114c-136${TAG}141-manifest.tsv --fudge 0.86 \
    --overlay-tag pi0scan-0829-agent --tsv docs/pr/pr136-census-$TAG.tsv \
    > /home/xqian/tmp/pr136_census_$TAG.log 2>&1
echo "  rc=$?"; grep -E "exact|sharing a gamma" /home/xqian/tmp/pr136_census_$TAG.log | head -3

if [ "$TAG" != "off1" ]; then
  echo "=== 4. escape attribution + membership delta ==="
  scripts/pr136_escape_census.py --on-arm "work-pr136-$TAG-*" \
      --on-prep emprep-136$TAG --off-prep emprep-136f086 \
      --tsv docs/pr/pr136-escape-census-$TAG.tsv
fi
