#!/bin/bash
# doc sbnd_xin/pr/144 sec 16 -- negative controls for the sentinel re-baseline.
#
# WHY.  Fourteen of the thirty pr/127 sentinels stopped holding when the
# excl_t0_frame patch was turned on (doc 144 sec 7.1).  The doc 91 sec 12
# discipline for re-baselining one is: measure BOTH sides at the new pinned
# point -- the fix's knob ON (that is the production arm) and the SAME event with
# that fix's own knob OFF -- then assert the STRUCTURAL property the fix shipped,
# with the threshold placed between the two measured values.  A sentinel
# re-baselined from the production side alone is the trap pr127_sentinels.py
# warns about at :119: "a threshold taken from a stale table passes the negative
# control on a dead knob".
#
# THE EPOCH TRAP.  Every arm here must run with excl_t0_frame ON.  Since the
# 2026-09-06 flip that is the compiled default, so no TLA is needed -- but do NOT
# run these against an older binary or an older cfg checkout, or the control
# measures the biased frame and the re-baseline is invalid.
#
# One arm per knob, so a FAIL is attributable.  Events that KEEP failing on
# purpose (177536, 347890, 393505 -- doc 144 secs 13/14) get no control here:
# they are open defects, not stale thresholds.
#
# THE 2x2.  Run once with no TLA (frame ON, the new default) and once with
# TLA=docs/pr/pr144-legacyframe.tla SUF=leg (frame OFF).  Together with the two
# positive arms that is {frame on, frame off} x {fix on, fix off}, which is what
# says whether the frame patch made a shipped fix inert or whether it already was.
#
# Usage: [PIN=/home/xqian/tmp/d144_libpin4] [JOBS=4] [TLA=<file>] [SUF=<suffix>]
#        ./scripts/pr144_sentinel_neg.sh
# Then:  ./scripts/pr127_sentinels.py --arms 'work-*-d144fixprod' 'work-s144neg*'
set -u
PIN=${PIN:-/home/xqian/tmp/d144_libpin4}
JOBS=${JOBS:-4}
SX=$(cd "$(dirname "$0")/.." && pwd)
cd "$SX" || exit 2
export LD_LIBRARY_PATH="$PIN:${LD_LIBRARY_PATH:-}"
export PR_EXTRA_STAGES=pr_display
unset PR_GROUP_SIZE
SUF=${SUF:-}
if [ -n "${TLA:-}" ]; then export PR_EXTRA_TLA="$SX/$TLA"; else unset PR_EXTRA_TLA; fi

LOG=/home/xqian/tmp/d144
mkdir -p "$LOG"
echo "=== pr144 sentinel negative controls  suffix=${SUF:-<none>}  tla=${TLA:-<none>}  pin=$PIN  $(date +%F_%H:%M:%S)"
md5sum "$PIN/libWireCellClus.so"

# name  env-assignment  ql-root  out-tag  events...
run() {
  local name=$1 env=$2 ql=$3 tag=$4; shift 4
  echo "--- $name  ($env)  events: $*  $(date +%H:%M:%S)"
  env $env PR_JOBS=$JOBS ./run_pr_chain_batch.sh "$ql" "work-s144neg$SUF-$tag" data "$@" \
      > "$LOG/s144neg_$tag$SUF.log" 2>&1
  echo "--- $name rc=$? events=$(ls -d work-s144neg$SUF-$tag/pr_evt*/ 2>/dev/null | wc -l)"
}

run "pr/130 B back-guard dvtx"     SBND_STEM_BACKFILL_BACK_DVTX=0        work-mcp2k-d97fv   dvtx     179369
run "pr/120 stem_backfill guard"   SBND_STEM_BACKFILL_BACK_GUARD=0       work-mcp2k-d97fv   backg    47212
run "pr/130 pass3 backfill guard"  SBND_SHOWER_PASS3_BACKFILL_GUARD_LEN=0 work-mcp1k-d97fv  bfill    175896
run "pr/129 pointing test"         SBND_KINE_GF_IMPACT=0                 work-mcp2k-d97fv   gf       94392 171572
run "pr/125 pass3_cone guard"      SBND_SHOWER_PASS3_CONE_GUARD_LEN=0    work-mcp2k-d97fv   cone     52693
run "doc 84 r2 members_geometry"   SBND_LONG_MUON_MEMBERS_GEOMETRY=0     work-mcp1k-d97fv   memgeo   281595
run "pr/130 pass4 prox guard"      SBND_SHOWER_PASS4_PROX_GUARD_LEN=0    work-mcp2k-d97fv   prox     100222
run "pr/127 sccc_max_gap 10 -> 6"  SBND_SCCC_MAX_GAP=6                   work-nuecc48-d97fv sccc     137238

echo "=== pr144 sentinel negative controls DONE $(date +%F_%H:%M:%S)"
