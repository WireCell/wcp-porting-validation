#!/bin/bash
# doc pr/139 phase 1 -- score ONE arm on the owner's instruments (fork of
# pr138_flipcheck.sh; that script stays byte-untouched, M10).
#
#   ./scripts/pr139_score.sh <armtag>
#
# 1  hand-scan charge attribution (q_miss / q_extra / median q_f1)
# 2  pi0 census exact  -- the headline; baseline 35/66
# 3  mass closure      -- fails if the R>1 (over-clustering) class grows
# 4  vertex movers     -- ADVERSE is stop-the-line
#
# --out for prep_em_scan goes to a TEMP path: its default is the tracked
# em114-manifest.tsv and a probe-parsing run would truncate that scan record
# to its header (M13).
set -u
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
TAG=$1
./scripts/pr139_manifests.sh $TAG
python3 em_display/prep_em_scan.py --prepdir em_display/emprep-139$TAG \
    --out /home/xqian/tmp/pr139-parse-$TAG.tsv --no-bee-index \
    --parse-probes ${PR139_ARM:-work-pr139r1}-$TAG-{mcp1k,mcp2k,ncpi0,nuecc48} \
    > /home/xqian/tmp/pr139-prep-$TAG.log 2>&1
echo "prep $TAG rc=$?"
( cd em_display
  ./em117_score.py --tag emscan-0827 --manifest em117-139${TAG}98-manifest.tsv \
      --prepdir emprep-139$TAG --tsv ../docs/pr/pr139-completeness-$TAG-98.tsv \
      > /home/xqian/tmp/pr139-score-$TAG-98.log 2>&1
  echo "  score98 $TAG rc=$?"
  ./em117_score.py --tag emscan-0828-agent5 --manifest em114c-139${TAG}141-manifest.tsv \
      --prepdir emprep-139$TAG --tsv ../docs/pr/pr139-completeness-$TAG-141.tsv \
      > /home/xqian/tmp/pr139-score-$TAG-141.log 2>&1
  echo "  score141 $TAG rc=$?" )
scripts/pr136_completeness.py --src98 pr139-completeness-$TAG-98.tsv \
    --src141 pr139-completeness-$TAG-141.tsv \
    --tsv docs/pr/pr139-completeness-$TAG.tsv > /home/xqian/tmp/pr139-comp-$TAG.txt 2>&1
echo "completeness $TAG rc=$?"
python3 scripts/pr132_pi0_census.py \
    --manifest98 em117-139${TAG}98-manifest.tsv \
    --manifest141 em114c-139${TAG}141-manifest.tsv \
    --fudge 0.86 --overlay-tag pi0scan-0829-agent \
    --tsv docs/pr/pr139-census-$TAG.tsv > /home/xqian/tmp/pr139-census-$TAG.txt 2>&1
echo "census $TAG rc=$?"
python3 scripts/pr136_mass_closure.py \
    --manifest98 em117-139${TAG}98-manifest.tsv \
    --manifest141 em114c-139${TAG}141-manifest.tsv \
    --overlay-tag pi0scan-0829-agent --fudge 0.86 \
    --tsv docs/pr/pr139-closure-$TAG.tsv > /home/xqian/tmp/pr139-closure-$TAG.txt 2>&1
echo "closure $TAG rc=$?"
for s in mcp1k mcp2k ncpi0 nuecc48; do
  python3 scripts/pr90_movers.py ${PR139_BASE:-work-pr139r1-off}-$s ${PR139_ARM:-work-pr139r1}-$TAG-$s --tags vtx105 \
      > /home/xqian/tmp/pr139-movers-$TAG-$s.txt 2>&1
  echo "movers $TAG $s rc=$?"
done
echo "SCORE $TAG DONE"
