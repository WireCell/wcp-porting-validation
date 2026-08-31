#!/bin/bash
# doc pr/136 arms (fork of pr130_arms.sh -- prefix work-pr136-, and the probe
# set gains WCT_SHOWER_XCLUS_DEBUG, which did not exist when pr130_arms.sh was
# written (toolkit deca3467).  pr130_arms.sh / pr134_arms.sh stay byte-untouched
# (M10).
#
# WHY THIS ARM EXISTS.  Doc pr/136 sec 5's completeness numbers are computed on
# the kept pr130r1-probe arms, which sit at kine_shower_fudge_factor 0.80 and
# predate the pr/133+134 chain.  Every other pr/136 number (mass closure,
# containment) is at the f086 PRODUCTION point.  This arm mints an emprep-
# membership sidecar AT THE PRODUCTION POINT so the two halves of the doc stop
# being a cross-arm join.
#
# BYTE-NEUTRAL.  probes=1 sets four getenv-gated stderr tapes only; none of them
# is a knob and none is read by any predicate:
#   WCT_SHOWER_ABSORB_DEBUG   absorb/seed/dedup call-site tape (doc pr/93 r3)
#   WCT_SHOWER_CONTENT_DEBUG  per-shower membership dump      (doc pr/91 r1)
#   WCT_SHOWER_XCLUS_DEBUG    cross-cluster acquisition tape  (doc pr/130 i4 p10)
#   WCT_SHOWER_BLOCKED_DEBUG  shower-walk contention tape     (doc pr/130 i4 p8)
# No SBND_* env is set, so the job runs the shipped production config: fudge
# 0.86 (toolkit b5cc3a3f) + the pr/133/134 chain.
#
# Usage: pr136_arms.sh <98|141> <TAG> <PROBES> [KEY=VAL ...]
#   The 98 and 141 event lists are disjoint and both write into
#   work-pr136-<TAG>-<sample>; run_pr_chain_batch.sh permits the merge because
#   it created the dir itself.
set -u
MAN=$1; TAG=$2; PROBES=$3; shift 3
for kv in "$@"; do export "$kv"; done
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
if [ "$PROBES" = "1" ]; then
  export WCT_SHOWER_ABSORB_DEBUG=1 WCT_SHOWER_CONTENT_DEBUG=1 \
         WCT_SHOWER_XCLUS_DEBUG=1 WCT_SHOWER_BLOCKED_DEBUG=1
fi
export PR_EXTRA_STAGES=pr_display PR_JOBS=${PR_JOBS:-24}
declare -A EV
if [ "$MAN" = "98" ]; then
  SAMPLES="mcp1k mcp2k ncpi0 nuecc48"
  EV[mcp1k]="166870 169356 169626 172942 174752 278794 281485 281639 284235 347129 394532 399052 409634 64591"
  EV[mcp2k]="165157 173093 176502 176986 281165 281567 281781 282909 396222 409888 415278 47212 475096 54332 76346 76350 99838"
  EV[ncpi0]="105946 114446 142421 180801 18625 21073 259542 285567 314838 359980 37112 399860 463565 506114 506746 521075 56982 71372 84229"
  EV[nuecc48]="10550 111412 116962 122660 131357 137238 138009 163543 168596 172230 174637 196649 214469 219295 234638 235435 239794 246579 256587 267597 268067 268784 269774 271851 30504 342199 350186 360535 388 38856 389538 400474 42280 422851 423981 433451 437699 444187 447477 46363 469665 489330 52672 54095 69314 74544 81597 90055"
elif [ "$MAN" = "141" ]; then
  SAMPLES="mcp1k mcp2k"
  EV[mcp1k]=$(awk -F'\t' 'NR>1&&$1=="mcp1k"{printf "%s ",$4}' em_display/em114c-manifest.tsv)
  EV[mcp2k]=$(awk -F'\t' 'NR>1&&$1=="mcp2k"{printf "%s ",$4}' em_display/em114c-manifest.tsv)
else
  echo "unknown manifest $MAN"; exit 2
fi
for s in $SAMPLES; do
  [ -z "${EV[$s]:-}" ] && continue
  ./run_pr_chain_batch.sh work-$s-grp0825 work-pr136-$TAG-$s data ${EV[$s]} \
      > /home/xqian/tmp/pr136_${TAG}_${MAN}_$s.log 2>&1
  echo "arm $TAG-$MAN-$s rc=$?"
done
