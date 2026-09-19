#!/bin/bash
# doc sbnd_xin/pr/150 -- post-process one cell against the baseline on one stage.
#   post_cell.sh <stage 1|3> <cell> [base=s0]
# Stage 1 = nuecc48 ncpi0; stage 3 = mcp1k mcp2k.  Writes docs/pr/150_figs/150_s<stage>_*_<cell>.txt and the
# metrics / traj TSVs.  Every step prints its rc; nothing is judged through a pipe.
set -u
ST=${1:?stage}; C=${2:?cell}; B=${3:-s0}
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin; cd $SX || exit 2
D=docs/pr/150_figs; P=scripts/analysis/pr150; L=/home/xqian/tmp/pr150/logs
case $ST in 1) SAMPLES="nuecc48 ncpi0";; 3) SAMPLES="mcp1k mcp2k";; *) echo "stage 1|3"; exit 2;; esac
A=pr150$B; X=pr150$C
echo "=== post stage $ST cell $C vs $B  $(date +%F_%T)"
for arm in $A $X; do
  for s in $SAMPLES; do [ -s $D/metrics/$arm-$s.tsv ] || { python3 $P/pr150_metrics.py extract --arm $arm --samples $s > $L/metrics_${arm}_$s.log 2>&1; echo "metrics extract $arm $s rc=$?"; }; done
  for s in $SAMPLES; do [ -s $D/traj/$arm-$s.tsv ] || { python3 $P/pr150_traj_eval.py extract --arm $arm --samples $s --jobs 8 > $L/traj_${arm}_$s.log 2>&1; echo "traj extract $arm $s rc=$?"; }; done
done
[ $C = $B ] && exit 0
python3 $P/pr150_metrics.py compare --a $A --b $X --samples $SAMPLES --movers $D/150_s${ST}_movers_$C.tsv > $D/150_s${ST}_compare_$C.txt 2>&1; echo "compare rc=$?"
python3 $P/pr150_traj_eval.py compare --a $A --b $X --samples $SAMPLES $([ $ST = 3 ] && echo "--strata $D/../149_figs/149_stage2_selection.tsv") > $D/150_s${ST}_traj_$C.txt 2>&1; echo "traj compare rc=$?"
python3 $P/r2_local_jitter.py --stage $([ $ST = 3 ] && echo 2 || echo 1) --pairs $X:$A > $D/150_s${ST}_jitter_$C.txt 2>&1; echo "jitter rc=$?"
python3 $P/r2_zzi_sign.py --stage $([ $ST = 3 ] && echo 2 || echo 1) --pairs $X:$A > $D/150_s${ST}_zzi_$C.txt 2>&1; echo "zzi rc=$?"
python3 $P/vertex_tolerance.py --stage $ST --base $A --arms $X > $D/150_s${ST}_vtx_$C.txt 2>&1; echo "vertex_tolerance rc=$?"
python3 $P/pr150_vtx_movers.py --a $A --b $X --samples $SAMPLES --out $D/150_s${ST}_vtxmovers_$C.tsv > $L/vtxmovers_${ST}_$C.log 2>&1; echo "vtx movers rc=$? $(cat $L/vtxmovers_${ST}_$C.log)"
python3 $P/pr150_stm_margin.py --a $A --b $X --samples $SAMPLES --tsv $D/150_s${ST}_stm_$C.tsv > $D/150_s${ST}_stm_$C.txt 2>&1; echo "stm margin rc=$?"
ARMS=""; for s in $SAMPLES; do ARMS="$ARMS work-$s-$X"; done
python3 scripts/analysis/pr149/sentinels_tolerant.py --arms $ARMS > $D/150_s${ST}_sentinels_$C.txt 2>&1; echo "sentinels rc=$? PASS $(grep -c '^PASS' $D/150_s${ST}_sentinels_$C.txt) FAIL $(grep -c '^FAIL' $D/150_s${ST}_sentinels_$C.txt) SKIP $(grep -c '^SKIP' $D/150_s${ST}_sentinels_$C.txt)"
if [ ! -s $D/150_s${ST}_sentinels_$B.txt ]; then ARMS=""; for s in $SAMPLES; do ARMS="$ARMS work-$s-$A"; done; python3 scripts/analysis/pr149/sentinels_tolerant.py --arms $ARMS > $D/150_s${ST}_sentinels_$B.txt 2>&1; echo "sentinels base rc=$?"; fi
echo "=== post stage $ST cell $C DONE $(date +%F_%T)"
