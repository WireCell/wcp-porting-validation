#!/bin/bash
# doc pr/109 sec 9 -- the sec 4 grid re-pointed at the sec 9 factorial arms:
# SBND work-pr109f-{on,off}-fbcoff-nuecc48, i.e. the same post-fix binary as
# sec 8's pr109e arms but with fit_blob_coverage = -1 (the C++ default, uBooNE's
# setting) instead of SBND production's 0.  Forked rather than parameterised so
# the sec 8 numbers stay reproducible from pr109b_run_all.sh (M13).
#
# SBND only: uBooNE never sets fit_blob_coverage, so its arms are unchanged and
# the sec 8 pr109e_wct_{on,off} numbers remain the reference.
#
# Usage: pr109f_run_all.sh [OUTDIR] [suffix] [extra args to pr109_2d_resid.py]
set -u
here=$(cd "$(dirname "$0")" && pwd); W=$(dirname "$here")
R=$here/pr109_2d_resid.py
OUT=${1:-/home/xqian/tmp/pr109f}
mkdir -p "$OUT"; sfx=${2:-}; extra=${3:-}; rm -f $OUT/sbnd_fbcoff$sfx.tsv
for box in 3.0 2.25 3.75; do
  for ev in 10550 46363 81597 360535 256587 433451; do
    python3 $R --box-cm $box --max-junctions 3 --sigma-from OFF --anchors-from OFF --anchors-from ON \
      --arm ON=$W/work-pr109f-on-fbcoff-nuecc48/pr_evt$ev/tracking-pr.root:wct \
      --arm OFF=$W/work-pr109f-off-fbcoff-nuecc48/pr_evt$ev/tracking-pr.root:wct \
      --tag sbnd-$ev --tsv $OUT/sbnd_fbcoff$sfx.tsv $extra
  done
done
