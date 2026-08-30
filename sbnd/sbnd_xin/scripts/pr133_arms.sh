#!/bin/bash
# pr/133 arms (fork of pr132_arms.sh: work-pr133- prefix, same manifests/probe
# env -- pr132_arms.sh stays byte-untouched, M10).
# Usage: pr132_arms.sh <98|141|dbg98|dbg141> <TAG> <PROBES> [KEY=VAL ...]
# probes=1 => WCT_PI0_PAIR_DEBUG=1, the doc pr/132 pi0-finder attribution tape
#   (stderr only, no knob, no production gate relaxed).  See
#   NeutrinoShowerClustering.cxx "WCT_PI0_PAIR_DEBUG".
# dbg98/dbg141 run only the pi0 miss-attribution subset (census no-group/none
# events at the pr131-denom point + the pr125->pr131 exact-match losses + the
# owner-flagged rescan heads); 98/141 run the full standard manifests.
# Extra KEY=VAL args are exported (e.g. SBND_KINE_SHOWER_FUDGE=0.84).
set -u
MAN=$1; TAG=$2; PROBES=$3; shift 3
for kv in "$@"; do export "$kv"; done
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
if [ "$PROBES" = "1" ]; then
  export WCT_PI0_PAIR_DEBUG=1
fi
export PR_EXTRA_STAGES=pr_display PR_JOBS=${PR_JOBS:-24}
declare -A EV
case "$MAN" in
98)
  SAMPLES="mcp1k mcp2k ncpi0 nuecc48"
  EV[mcp1k]="166870 169356 169626 172942 174752 278794 281485 281639 284235 347129 394532 399052 409634 64591"
  EV[mcp2k]="165157 173093 176502 176986 281165 281567 281781 282909 396222 409888 415278 47212 475096 54332 76346 76350 99838"
  EV[ncpi0]="105946 114446 142421 180801 18625 21073 259542 285567 314838 359980 37112 399860 463565 506114 506746 521075 56982 71372 84229"
  EV[nuecc48]="10550 111412 116962 122660 131357 137238 138009 163543 168596 172230 174637 196649 214469 219295 234638 235435 239794 246579 256587 267597 268067 268784 269774 271851 30504 342199 350186 360535 388 38856 389538 400474 42280 422851 423981 433451 437699 444187 447477 46363 469665 489330 52672 54095 69314 74544 81597 90055"
  ;;
141)
  SAMPLES="mcp1k mcp2k"
  EV[mcp1k]=$(awk -F'\t' 'NR>1&&$1=="mcp1k"{printf "%s ",$4}' em_display/em114c-manifest.tsv)
  EV[mcp2k]=$(awk -F'\t' 'NR>1&&$1=="mcp2k"{printf "%s ",$4}' em_display/em114c-manifest.tsv)
  ;;
dbg98)
  SAMPLES="mcp1k mcp2k ncpi0 nuecc48"
  EV[mcp1k]="169626 278794 347129 281485"
  EV[mcp2k]="47212 176502 415278"
  EV[ncpi0]="285567 359980 506746 37112 142421"
  EV[nuecc48]="342199"
  ;;
dbg141)
  SAMPLES="mcp1k mcp2k"
  EV[mcp1k]="54341 56243 71178 168432 285443"
  EV[mcp2k]="52044 71872 99782 103798 403023 74123"
  ;;
*) echo "unknown manifest $MAN"; exit 2;;
esac
for s in $SAMPLES; do
  [ -z "${EV[$s]}" ] && continue
  ./run_pr_chain_batch.sh work-$s-grp0825 work-pr133-$TAG-$s data ${EV[$s]} > /home/xqian/tmp/pr133_${TAG}_$s.log 2>&1
  echo "arm $TAG-$s rc=$?"
done
