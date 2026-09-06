#!/bin/bash
# doc sbnd_xin/pr/144 -- the two full-population arms for the fit_exclusion T0 frame patch.
#
#   OFF = today's production (both TLAs at their default false; the compiled config is
#         byte-identical to pre-change production, proof T0 md5 3bfd2a80d0201d22e9a1b5db37c774eb)
#   ON  = excl_t0_frame + kine_dqdx_skip_zero_dx, the PDVD production pair, via PR_EXTRA_TLA.
#
# Per-SAMPLE out_roots (work-<sample>-<tag>), not one merged root, because pr_scores_table.py
# takes --sample and pr142_campaign_ab.py joins the four per-sample TSVs.
#
# Usage: [JOBS=16] [PIN=/home/xqian/tmp/d144_libpin2] [TAG=d144] ./scripts/pr144_arms.sh off|on|prod|frameonly
#   TAG names the arm set, so a re-run on a new binary gets fresh out_roots (M13):
#     TAG=d144  -> work-<sample>-d144{off,on}   (pin1, b46179b2..., ON arm has the 494297 SIGSEGV)
#     TAG=d144b -> work-<sample>-d144b{off,on}  (pin2, 251bc37c..., the null-transform guard)
#   The guard-attribution arm reuses pin1: PIN=/home/xqian/tmp/d144_libpin TAG=d144
#   ./scripts/pr144_arms.sh frameonly -> work-<sample>-d144frameonly.
# Reality is `data` for ALL FOUR samples -- that is what the production arms
# (work-*-d97fvpr2) carry.  doc pdvd/45 block I used sim for ncpi0; production does not.
set -u
ARM=${1:?usage: pr144_arms.sh off|on|prod|frameonly}
JOBS=${JOBS:-16}
PIN=${PIN:-/home/xqian/tmp/d144_libpin2}
TAG=${TAG:-d144}
SX=$(cd "$(dirname "$0")/.." && pwd)
cd "$SX" || exit 2

export LD_LIBRARY_PATH="$PIN:${LD_LIBRARY_PATH:-}"
export PR_EXTRA_STAGES=pr_display          # calib-pr-evt<ID>.json, which the census scripts read
unset PR_GROUP_SIZE                        # per-event mode => per-event wall_s / maxrss_kb

case "$ARM" in
  off) unset PR_EXTRA_TLA ;;
  # doc 144 sec 13: the committed default with NO TLA at all.  Same compiled
  # config as `on` once the 2026-09-06 flip landed, but it PROVES that -- an
  # arm that forces the keys cannot show that the default carries them.
  prod) unset PR_EXTRA_TLA ;;
  on)  export PR_EXTRA_TLA="$SX/docs/pr/pr144-on.tla" ;;
  # doc 144 sec 4.5: excl_t0_frame alone, to attribute the kine dx<=0 guard by
  # a byte gate against the ON arm.  The score tables cannot do it -- the guard
  # masks the NaN kine_reco_Enu that is its own fire signature.
  frameonly) export PR_EXTRA_TLA="$SX/docs/pr/pr144-frameonly.tla" ;;
  *)   echo "arm must be off|on|prod|frameonly" >&2; exit 2 ;;
esac

echo "=== pr144 arm $TAG$ARM  jobs=$JOBS  pin=$PIN  $(date +%F_%H:%M:%S)"
md5sum "$PIN/libWireCellClus.so"
[ -n "${PR_EXTRA_TLA:-}" ] && { echo "--- PR_EXTRA_TLA ---"; grep -v '^#' "$PR_EXTRA_TLA"; }

for s in ncpi0 nuecc48 mcp1k mcp2k; do          # small samples first: fail fast
  OUT="work-$s-$TAG$ARM"
  echo "--- $s -> $OUT  $(date +%H:%M:%S)"
  PR_JOBS=$JOBS ./run_pr_chain_batch.sh "work-$s-d97fv" "$OUT" data
  echo "--- $s rc=$? events=$(ls -d $OUT/pr_evt*/ 2>/dev/null | wc -l)  $(date +%H:%M:%S) loadavg=$(cut -d' ' -f1 /proc/loadavg)"
done
echo "=== pr144 arm $TAG$ARM DONE $(date +%F_%H:%M:%S)"
