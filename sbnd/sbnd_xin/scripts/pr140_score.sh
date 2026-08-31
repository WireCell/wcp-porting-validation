#!/bin/bash
# doc pr/139 sec 8 -- score ONE pr140 arm.  FORK of pr139_score.sh (M10); that
# script stays byte-untouched and keeps writing docs/pr/pr139-*.tsv, which are
# the tracked record of the pr/139 arms.  Three things differ:
#
#   * output paths are pr140-*, so this cannot overwrite pr/139's tracked TSVs;
#   * the BASELINE is the FLIPPED production config.  pr139_score.sh defaults
#     PR139_BASE=work-pr139r1-off, which is PRE-flip; using it here would
#     attribute shower_split_em_start's movers to the knobs under test.  The
#     default below is work-pr139r1-onemst, proven byte-identical to the
#     post-flip config arm work-pr139r3-flipchk by a 478/478 gate;
#   * completeness is scored TWICE -- em117_score (single target, the number
#     every doc from pr/117 quotes) and em140_score --split-tag (the per-part
#     target, doc pr/139 item 2).  Both, always: only 12 of ~90 rows change
#     target, so a metric change and a reco change must never land in one
#     number (the doc pr/130 census bug that looked like physics).
#
#   ./scripts/pr140_score.sh <armtag>
set -u
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
TAG=$1
ARM=${PR140_ARM:-work-pr140r1}
BASE=${PR140_BASE:-work-pr139r1-onemst}
SPLITTAG=${PR140_SPLIT_TAG:-splitscan-0902-pi0}
./scripts/pr140_manifests.sh $TAG
python3 em_display/prep_em_scan.py --prepdir em_display/emprep-140$TAG \
    --out /home/xqian/tmp/pr140-parse-$TAG.tsv --no-bee-index \
    --parse-probes $ARM-$TAG-{mcp1k,mcp2k,ncpi0,nuecc48} \
    > /home/xqian/tmp/pr140-prep-$TAG.log 2>&1
echo "prep $TAG rc=$?"
( cd em_display
  ./em117_score.py --tag emscan-0827 --manifest em117-140${TAG}98-manifest.tsv \
      --prepdir emprep-140$TAG --tsv ../docs/pr/pr140-completeness-$TAG-98.tsv \
      > /home/xqian/tmp/pr140-score-$TAG-98.log 2>&1
  echo "  score98 $TAG rc=$?"
  ./em117_score.py --tag emscan-0828-agent5 --manifest em114c-140${TAG}141-manifest.tsv \
      --prepdir emprep-140$TAG --tsv ../docs/pr/pr140-completeness-$TAG-141.tsv \
      > /home/xqian/tmp/pr140-score-$TAG-141.log 2>&1
  echo "  score141 $TAG rc=$?"
  ./em140_score.py --split-tag $SPLITTAG \
      --tag emscan-0827 --manifest em117-140${TAG}98-manifest.tsv \
      --prepdir emprep-140$TAG --tsv ../docs/pr/pr140-perpart-$TAG-98.tsv \
      > /home/xqian/tmp/pr140-perpart-$TAG-98.log 2>&1
  echo "  perpart98 $TAG rc=$?"
  ./em140_score.py --split-tag $SPLITTAG \
      --tag emscan-0828-agent5 --manifest em114c-140${TAG}141-manifest.tsv \
      --prepdir emprep-140$TAG --tsv ../docs/pr/pr140-perpart-$TAG-141.tsv \
      > /home/xqian/tmp/pr140-perpart-$TAG-141.log 2>&1
  echo "  perpart141 $TAG rc=$?" )
scripts/pr136_completeness.py --src98 pr140-completeness-$TAG-98.tsv \
    --src141 pr140-completeness-$TAG-141.tsv \
    --tsv docs/pr/pr140-completeness-$TAG.tsv > /home/xqian/tmp/pr140-comp-$TAG.txt 2>&1
echo "completeness $TAG rc=$?"
python3 scripts/pr132_pi0_census.py \
    --manifest98 em117-140${TAG}98-manifest.tsv \
    --manifest141 em114c-140${TAG}141-manifest.tsv \
    --fudge 0.86 --overlay-tag pi0scan-0829-agent \
    --tsv docs/pr/pr140-census-$TAG.tsv > /home/xqian/tmp/pr140-census-$TAG.txt 2>&1
echo "census $TAG rc=$?"
for s in mcp1k mcp2k ncpi0 nuecc48; do
  python3 scripts/pr90_movers.py $BASE-$s $ARM-$TAG-$s --tags vtx105 \
      > /home/xqian/tmp/pr140-movers-$TAG-$s.txt 2>&1
  echo "movers $TAG $s rc=$?"
done
echo "SCORE $TAG DONE"
