#!/bin/bash
# doc pr/138 -- the FLIP question, answered on the owner's own instruments.
#
# "Should shower_split be turned on for SBND running?"  The Phase B round scored
# the splitter against 172 hand labels; it never asked what it does to a physics
# output.  This runs doc pr/136 sec 11.2's three instruments plus the vertex
# movers, OFF vs ON, on two arms that differ ONLY by SBND_SHOWER_SPLIT:
#
#   work-pr138r2-poff-<s>   production config, knob off   (239 events, dumps on)
#   work-pr138r2-pon-<s>    production config, knob ON
#
#   1  hand-scan charge attribution (90 marked showers): q_miss / q_extra / q_f1
#      OFF reference: q_miss 14.0 %, q_extra 7.0 %, median q_f1 0.918
#      FAILS IF q_extra rises by more than q_miss falls
#   2  pi0 census exact   OFF reference 32/66 = 48.5 %   FAILS IF it drops
#   3  mass closure       FAILS IF the R>1 (over-clustering) class grows
#   4  vertex movers      ADVERSE is stop-the-line
#
# READ-ONLY on every input.  --out for prep_em_scan goes to a temp path: its
# default is the TRACKED em114-manifest.tsv and a probe-parsing run would
# truncate that scan record to its header (M13).
set -u
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
for TAG in poff pon; do
  ./scripts/pr138_manifests.sh $TAG
  python3 em_display/prep_em_scan.py --prepdir em_display/emprep-138$TAG \
      --out /home/xqian/tmp/pr138-parse-$TAG.tsv --no-bee-index \
      --parse-probes work-pr138r2-$TAG-{mcp1k,mcp2k,ncpi0,nuecc48} \
      > /home/xqian/tmp/pr138-prep-$TAG.log 2>&1
  echo "prep $TAG rc=$?"
  ( cd em_display
    ./em117_score.py --tag emscan-0827 --manifest em117-138${TAG}98-manifest.tsv \
        --prepdir emprep-138$TAG --tsv ../docs/pr/pr138-completeness-$TAG-98.tsv \
        > /home/xqian/tmp/pr138-score-$TAG-98.log 2>&1
    echo "  score98 $TAG rc=$?"
    ./em117_score.py --tag emscan-0828-agent5 --manifest em114c-138${TAG}141-manifest.tsv \
        --prepdir emprep-138$TAG --tsv ../docs/pr/pr138-completeness-$TAG-141.tsv \
        > /home/xqian/tmp/pr138-score-$TAG-141.log 2>&1
    echo "  score141 $TAG rc=$?" )
  scripts/pr136_completeness.py --src98 pr138-completeness-$TAG-98.tsv \
      --src141 pr138-completeness-$TAG-141.tsv \
      --tsv docs/pr/pr138-completeness-$TAG.tsv > /home/xqian/tmp/pr138-comp-$TAG.txt 2>&1
  echo "completeness $TAG rc=$?"
  python3 scripts/pr132_pi0_census.py \
      --manifest98 em117-138${TAG}98-manifest.tsv \
      --manifest141 em114c-138${TAG}141-manifest.tsv \
      --fudge 0.86 --overlay-tag pi0scan-0829-agent \
      --tsv docs/pr/pr138-census-$TAG.tsv > /home/xqian/tmp/pr138-census-$TAG.txt 2>&1
  echo "census $TAG rc=$?"
  python3 scripts/pr136_mass_closure.py \
      --manifest98 em117-138${TAG}98-manifest.tsv \
      --manifest141 em114c-138${TAG}141-manifest.tsv \
      --overlay-tag pi0scan-0829-agent --fudge 0.86 \
      --tsv docs/pr/pr138-closure-$TAG.tsv > /home/xqian/tmp/pr138-closure-$TAG.txt 2>&1
  echo "closure $TAG rc=$?"
done
for s in mcp1k mcp2k ncpi0 nuecc48; do
  python3 scripts/pr90_movers.py work-pr138r2-poff-$s work-pr138r2-pon-$s --tags vtx105 \
      > /home/xqian/tmp/pr138-movers-$s.txt 2>&1
  echo "movers $s rc=$?"
done
echo FLIPCHECK DONE
