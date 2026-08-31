#!/bin/bash
# doc pr/109 -- the whole grid: 3 case families x 6 events x 2 arms x 3 box sizes,
# anchors taken from BOTH arms.  Writes one TSV per family into $OUT.
set -u
here=$(cd "$(dirname "$0")" && pwd); W=$(dirname "$here")
R=$here/pr109_2d_resid.py
S=/nfs/data/1/xqian/toolkit-dev/toolkit/qlport/scripts/sweep
OUT=${1:-/home/xqian/tmp/pr109}
mkdir -p "$OUT"; sfx=${2:-}; extra=${3:-}; rm -f $OUT/{ub_wct,ub_wcp,sbnd}$sfx.tsv
declare -A IDX=( [6505]=1 [6528]=4 [6532]=6 [6650]=16 [6805]=22 [6806]=23 )
declare -A SR=( [6505]=130 [6528]=130 [6532]=130 [6650]=132 [6805]=136 [6806]=136 )
for box in 3.0 2.25 3.75; do
  for ev in 6505 6528 6532 6650 6805 6806; do
    i=${IDX[$ev]}; sr=${SR[$ev]}
    python3 $R --box-cm $box --max-junctions 3 --sigma-from OFF --anchors-from OFF --anchors-from ON \
      --arm ON=$S/pr108_wct_on/${i}_${ev}/track_com_5384_${ev}.root:wct \
      --arm OFF=$S/pr108_wct_off/${i}_${ev}/track_com_5384_${ev}.root:wct \
      --tag ub-wct-$ev --tsv $OUT/ub_wct$sfx.tsv $extra
    python3 $R --box-cm $box --max-junctions 3 --sigma-from OFF --anchors-from OFF --anchors-from ON \
      --arm ON=$S/pr108_wcp_on/nue_5384_${sr}_${ev}.root:wcp \
      --arm OFF=$S/pr108_wcp_off/nue_5384_${sr}_${ev}.root:wcp \
      --tag ub-wcp-$ev --tsv $OUT/ub_wcp$sfx.tsv $extra
  done
  for ev in 10550 46363 81597 360535 256587 433451; do
    python3 $R --box-cm $box --max-junctions 3 --sigma-from OFF --anchors-from OFF --anchors-from ON \
      --arm ON=$W/work-pr109-on-nuecc48/pr_evt$ev/tracking-pr.root:wct \
      --arm OFF=$W/work-pr109-off-nuecc48/pr_evt$ev/tracking-pr.root:wct \
      --tag sbnd-$ev --tsv $OUT/sbnd$sfx.tsv $extra
  done
done
