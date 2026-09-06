#!/bin/bash
# doc sbnd_xin/pr/144 -- the whole analysis of the d144off/d144on pair, in order.
# REFUSES to run while either arm is still writing (feedback_gate_against_a_running_arm:
# a completion marker is the runner's own "arm DONE" line, never a directory count).
# Usage: [TAG=d144b] ./scripts/pr144_analyze.sh [step ...]   default: all
# TAG must match the one pr144_arms.sh ran with, or the DONE-line guard below
# never matches and this script refuses forever (fail-safe, but useless).
set -u
SX=$(cd "$(dirname "$0")/.." && pwd); cd "$SX"
D=/home/xqian/tmp/d144; mkdir -p $D
TAG=${TAG:-d144}
SAMPLES="nuecc48 ncpi0 mcp1k mcp2k"
STEPS=${*:-scores ab bytes movers pi0 sentinels gaps}

for a in off on; do
  # The first d144 arms were produced before this script was parameterised, so
  # their logs are arm_<off|on>.log carrying "=== pr144 arm <off|on> DONE";
  # later tags write arm_<TAG><off|on>.log.  Accept either -- but ONLY a real
  # DONE line, never a directory count (a pr_evt dir exists from job start).
  _lg=""
  for _c in "$D/arm_$TAG$a.log" "$D/arm_$a.log"; do
    [ -f "$_c" ] && grep -q "=== pr144 arm \(.*\)\?$a DONE" "$_c" && { _lg="$_c"; break; }
  done
  [ -n "$_lg" ] || { echo "REFUSING: no DONE line for arm $TAG$a in $D/arm_$TAG$a.log or $D/arm_$a.log" >&2; exit 2; }
  echo "  arm $TAG$a complete per $_lg"
done
pgrep -u "$USER" -f wct-pr-perevt >/dev/null 2>&1 && {
  echo "REFUSING: a wire-cell PR job is still running" >&2; exit 2; }
echo "both arms complete; starting $(date +%F_%H:%M:%S)"

run() { echo; echo "############ $*"; }

case " $STEPS " in *" scores "*)
  run "1. score tables"
  for t in ${TAG}off ${TAG}on; do mkdir -p products/$t
    for s in $SAMPLES; do
      [ -s products/$t/$s-scores-$t.tsv ] && { echo "  $t/$s cached"; continue; }
      nice -n 10 python3 pr_scores_table.py --root work-$s-$t --sample $s \
          --out products/$t/$s-scores-$t.tsv >/dev/null 2>&1
      echo "  $t/$s rows=$(($(wc -l < products/$t/$s-scores-$t.tsv)-1))"
    done
  done ;; esac

case " $STEPS " in *" ab "*)
  run "2. systematic physics A/B (pr142_campaign_ab.py)"
  nice -n 10 python3 scripts/pr142_campaign_ab.py \
    --a products/${TAG}off/*.tsv --b products/${TAG}on/*.tsv \
    --label-a ${TAG}off --label-b ${TAG}on \
    --movers-tsv docs/pr/pr144-movers.tsv --summary-tsv docs/pr/pr144-population.tsv \
    > $D/campaign_ab.log 2>&1
  echo "  rc=$? -> $D/campaign_ab.log" ;; esac

case " $STEPS " in *" bytes "*)
  run "3. per-sample byte gate + leaf attribution"
  for s in $SAMPLES; do
    nice -n 10 python3 scripts/analysis/pr143/pr143_compare_arms.py \
       work-$s-${TAG}off work-$s-${TAG}on --jobs 6 > $D/cmp_$s.log 2>&1
    echo "  $s rc=$? -> $D/cmp_$s.log"
  done
  for s in $SAMPLES; do
    nice -n 10 python3 scripts/analysis/pr143/pr143_branch_diff.py \
       work-$s-${TAG}off work-$s-${TAG}on --jobs 6 > $D/branch_$s.log 2>&1
    echo "  branch $s rc=$? -> $D/branch_$s.log"
  done ;; esac

case " $STEPS " in *" movers "*)
  run "4. doc-144 mover classes and the Bee pick"
  nice -n 10 python3 scripts/pr144_pick_movers.py --movers docs/pr/pr144-movers.tsv \
     --a products/${TAG}off/*.tsv --b products/${TAG}on/*.tsv --n 12 \
     --pick-tsv docs/pr/pr144-beepick.tsv > $D/movers.log 2>&1
  echo "  rc=$? -> $D/movers.log" ;; esac

case " $STEPS " in *" pi0 "*)
  run "5. pi0 66-set census (the doc-45 sec 11 precondition)"
  # fudge: BOTH arms run the same production kine_shower_fudge_factor, so the
  # same --fudge is right for both -- this pair differs only by the frame knob.
  F=$(python3 - <<'PY'
import json,glob
f=sorted(glob.glob('work-ncpi0-${TAG}off/pr_evt*/.wct-cfg-evt*.json'))[0]
d=json.load(open(f)); r=[]
def w(o):
    if isinstance(o,dict):
        if o.get('type')=='TaggerCheckNeutrino': r.append(o['data'].get('kine_shower_fudge_factor'))
        for v in o.values(): w(v)
    elif isinstance(o,list): [w(v) for v in o]
w(d); print(r[0] if r and r[0] is not None else 0.80)
PY
)
  echo "  arm kine_shower_fudge_factor = $F (read from the compiled config)"
  for t in ${TAG}off ${TAG}on; do
    ./scripts/pr144_pi0_manifests.sh $t > $D/pi0_manifest_$t.log 2>&1
    tail -1 $D/pi0_manifest_$t.log
    nice -n 10 python3 scripts/pr141_pi0_census2.py \
      --manifest98  /home/xqian/tmp/d144/manifests/$t/denom98.tsv \
      --manifest141 /home/xqian/tmp/d144/manifests/$t/denom141.tsv \
      --fudge $F --chain "pi0mass-0904-owner,pi0scan-0829-agent" \
      --tsv docs/pr/pr144-pi0census-$t.tsv > $D/pi0_$t.log 2>&1
    echo "  $t rc=$? -> $D/pi0_$t.log"
  done ;; esac

case " $STEPS " in *" sentinels "*)
  run "6. sentinel suite (pr142 baseline: 6 pre-existing FAIL)"
  for t in ${TAG}off ${TAG}on; do
    nice -n 10 python3 scripts/pr127_sentinels.py --arms "work-*-$t" > $D/sent_$t.log 2>&1
    echo "  $t rc=$? -> $D/sent_$t.log"
  done ;; esac

case " $STEPS " in *" gaps "*)
  run "7. track_fit gap census (the mechanism number)"
  for t in ${TAG}off ${TAG}on; do
    nice -n 10 python3 ../../pdvd/docs/nf_sp_img_clus/scripts/d45_trackfit_vs_stmfit.py \
       --all-clusters --tsv docs/pr/pr144-gapcensus-$t.tsv \
       work-nuecc48-$t/pr_evt* work-ncpi0-$t/pr_evt* > $D/gaps_$t.log 2>&1
    echo "  $t rc=$? -> $D/gaps_$t.log"
  done ;; esac

echo; echo "analysis done $(date +%F_%H:%M:%S)"
