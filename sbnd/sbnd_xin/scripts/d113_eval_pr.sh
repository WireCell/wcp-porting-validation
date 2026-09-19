#!/bin/bash
# doc sbnd_xin/113 sec 7 -- evaluate a stage-B-only arm (the PR knob) against production pr150s0, in the order of
# 113_pred_pr.txt: pr150 metrics + compare (vertex on labels, nu_evaluated / event_label migrations, working points,
# |dEnu|, TGM/STM/FC flips, pi0, resources), sentinels, the PR-stage census on the ON arm, the adoption census.
#   scripts/d113_eval_pr.sh <stageB tag, e.g. d113adopt>
set -u
SX=/home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin; cd $SX
B=$1; SAMPLES="nuecc48 ncpi0 mcp1k mcp2k"; OUT=docs/113_figs/eval/$B; mkdir -p $OUT
echo "== metrics extract $B  $(date +%T)"; python3 scripts/analysis/pr150/pr150_metrics.py extract --arm $B --samples $SAMPLES > $OUT/metrics.log 2>&1; echo rc=$?
[ -f docs/pr/150_figs/metrics/pr150s0-mcp2k.tsv ] || { python3 scripts/analysis/pr150/pr150_metrics.py extract --arm pr150s0 --samples $SAMPLES >> $OUT/metrics.log 2>&1; echo "extracted pr150s0 rc=$?"; }
echo "== compare pr150s0 -> $B  $(date +%T)"; python3 scripts/analysis/pr150/pr150_metrics.py compare --a pr150s0 --b $B --samples $SAMPLES --ref pr150s0 --movers $OUT/movers.tsv > $OUT/compare.log 2>&1; echo rc=$?
echo "== sentinels on $B  $(date +%T)"; python3 scripts/analysis/pr149/sentinels_tolerant.py --arms "work-*-$B" > $OUT/sentinels.log 2>&1; echo rc=$?
echo "== PR-stage census on $B  $(date +%T)"; python3 scripts/analysis/d113/d113_pr_uncover.py --jobs 8 --stageB $B --outdir $OUT/pr_census > $OUT/pr_census.log 2>&1; echo rc=$?
python3 scripts/analysis/d113/d113_pr_summary.py --census $OUT/pr_census --out $OUT/pr_summary.txt --flags $OUT/pr_flags.tsv --exhibits $OUT/pr_exhibits.tsv > /dev/null 2>&1; echo rc=$?
echo "== adoption census  $(date +%T)"; python3 scripts/analysis/d113/d113_adopt_census.py --on $B --out $OUT/adopt_census.txt > /dev/null 2>&1; echo rc=$?
echo "== done $(date +%T)"
