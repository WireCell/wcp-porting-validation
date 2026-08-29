#!/bin/bash
# doc pr/129 -- the owner's pointing discriminator on the guard-freed kine pool.
#   "the key difference is the direction, if the direction of the track is point
#    to the main vertex, it is more likely to be part of neutrino.  For
#    overclustering they are generally not point to neutrino vertex."
#                                                       -- owner, 2026-08-29
# Verdicts this must reproduce (owner, on the pr129gf Bee pair):
#   171572  KEEP the 304.75 MeV   ("784.9 MeV should be the right energy")
#   393505  DROP the 268.70 MeV   ("should be the lower energy")
#   94392   either                ("is OK")
#
# MEAS: knob armed wide open (impact 1000 cm, miss 180 deg) so every candidate
#       is still counted -- behaviour identical to production -- but the tape
#       line prints d_vtx / impact / miss for each.  Read the real C++ numbers
#       here before choosing a threshold.
# ON:   the chosen operating point.
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-meas}"
case "$mode" in
  meas) export SBND_KINE_GF_IMPACT=1000 SBND_KINE_GF_MISS_DEG=180; tag=meas ;;
  on)   export SBND_KINE_GF_IMPACT="${IMPACT:?set IMPACT}" SBND_KINE_GF_MISS_DEG="${MISS:?set MISS}"; tag=on ;;
  *) echo "usage: $0 [meas|on]"; exit 2 ;;
esac
PR_JOBS=3 PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh \
  work-mcp2k-grp0825 "work-pr129pt-$tag-mcp2k" data 171572 393505 94392 \
  > "/home/xqian/tmp/pr129_pt_$tag.log" 2>&1
echo "rc=$?"
