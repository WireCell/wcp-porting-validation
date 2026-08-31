#!/bin/bash
# pr/125 round 1 arms (fork of pr124_arms.sh: prefix change only).  Usage: pr125_arms.sh <manifest 98|141> <armtag> <probes 0|1> [env assignments...]
# probes=1 => WCT_SHOWER_{ABSORB,CONTENT,PID,TOPO}_DEBUG=1 (byte-neutral stderr census:
#   absorb/seed/dedup lines, membership dumps, pid vote/copy trace, topo classifier features)
# Events: 98-manifest = em114-manifest.tsv family (pr/117-120 arms);
#         141-manifest = em114c-manifest.tsv (doc pr/115 sec 17 out-of-sample scan).
set -u
MAN=$1; TAG=$2; PROBES=$3; shift 3
for kv in "$@"; do export "$kv"; done
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
if [ "$PROBES" = "1" ]; then
  export WCT_SHOWER_ABSORB_DEBUG=1 WCT_SHOWER_CONTENT_DEBUG=1 WCT_SHOWER_PID_DEBUG=1 WCT_SHOWER_TOPO_DEBUG=1
fi
export PR_EXTRA_STAGES=pr_display PR_JOBS=${PR_JOBS:-24}
declare -A EV
if [ "$MAN" = "98" ]; then
  SAMPLES="mcp1k mcp2k ncpi0 nuecc48"
  EV[mcp1k]="166870 169356 169626 172942 174752 278794 281485 281639 284235 347129 394532 399052 409634 64591"
  EV[mcp2k]="165157 173093 176502 176986 281165 281567 281781 282909 396222 409888 415278 47212 475096 54332 76346 76350 99838"
  EV[ncpi0]="105946 114446 142421 180801 18625 21073 259542 285567 314838 359980 37112 399860 463565 506114 506746 521075 56982 71372 84229"
  EV[nuecc48]="10550 111412 116962 122660 131357 137238 138009 163543 168596 172230 174637 196649 214469 219295 234638 235435 239794 246579 256587 267597 268067 268784 269774 271851 30504 342199 350186 360535 388 38856 389538 400474 42280 422851 423981 433451 437699 444187 447477 46363 469665 489330 52672 54095 69314 74544 81597 90055"
else
  SAMPLES="mcp1k mcp2k"
  EV[mcp1k]=$(awk -F'\t' 'NR>1&&$1=="mcp1k"{printf "%s ",$4}' em_display/em114c-manifest.tsv)
  EV[mcp2k]=$(awk -F'\t' 'NR>1&&$1=="mcp2k"{printf "%s ",$4}' em_display/em114c-manifest.tsv)
fi
for s in $SAMPLES; do
  ./run_pr_chain_batch.sh work-$s-grp0825 work-pr125r1-$TAG-$s data ${EV[$s]} > /home/xqian/tmp/pr125_${TAG}_$s.log 2>&1
  echo "arm $TAG-$s rc=$?"
done
