#!/bin/bash
# doc pdvd/119 sec 8 gates + knob-ON arms: A3 = clean 1e6b2905, B3 = + op_cluster_anodes.
W=/home/xqian/toolkit-dev/wcp-porting-img; K=$W/pdvd/kaon
A=/home/xqian/tmp/d119inst/A3; B=/home/xqian/tmp/d119inst/B3
OUT=/home/xqian/tmp/d119s_gates; mkdir -p $OUT
log() { echo "[$(date +%T)] $*" >> $OUT/driver.log; }
pin() { local p=$1; shift; PATH="$p/bin:$PATH" LD_LIBRARY_PATH="$p/lib:$LD_LIBRARY_PATH" "$@"; }
EV="0 1 3 5 9 15 22"
# 1. PDVD op path, 039349 (knob OFF: A3 vs B3off; knob ON: B3 vs B3off)
log "pdvd op A3";    PREFIX=$A TAG=d119sA3    CLUS_TLA="-S bee_cluster_anodes=false" $K/run_d119s_side_arms.sh $EV > $OUT/op_A3.log 2>&1; log "rc=$?"
log "pdvd op B3off"; PREFIX=$B TAG=d119sB3off CLUS_TLA="-S bee_cluster_anodes=false" $K/run_d119s_side_arms.sh $EV > $OUT/op_B3off.log 2>&1; log "rc=$?"
log "pdvd op B3";    PREFIX=$B TAG=d119sB3    $K/run_d119s_side_arms.sh $EV > $OUT/op_B3.log 2>&1; log "rc=$?"
# 2. 039252 idx 0 (evt 298567, the bee3 934d031 validation event), knob ON
log "039252 B3"; PREFIX=$B TAG=d119sB3 RUN=039252 LIGHT_SUFFIX=_tot BEAM=0 $K/run_d119s_side_arms.sh 0 > $OUT/op_252.log 2>&1; log "rc=$?"
# 3. 39305 x10, knob ON (doc-118 scratch settings)
log "39305 B3"; PREFIX=$B TAG=d119sidescratch $K/run_d119s_kaon_side_scratch.sh > $OUT/k305.log 2>&1; log "rc=$?"
# 4. PDHD + PDVD harness (q0)
cd $W/abtest
log "pd A3"; MAXJ=3 ./libab_clus_run.sh d119sgA $A > $OUT/pd_A.log 2>&1; log "rc=$?"
log "pd B3"; MAXJ=3 ./libab_clus_run.sh d119sgB $B > $OUT/pd_B.log 2>&1; log "rc=$?"
./libab_clus_compare.sh d119sgA d119sgB $OUT/pd > $OUT/pd_compare.log 2>&1; log "pd compare rc=$?"
# 5. SBND Q/L stage on nuecc48 group 0
SB=$W/sbnd/sbnd_xin; IN=$SB/input_files_reco1/data_filtered_decoded_reco1-fe6033f3-07a0-4971-cea5-16ce59269fba_eventidfiltered_frameshift.root
cd $SB
for arm in A B; do
  R=work-nuecc48-d119sg$arm; [ -e $R ] && { log "$R exists, refusing"; continue; }
  mkdir -p $R/g0; touch $R/.chain_group
  for f in frames-dnn.tar.bz2 icluster-apa0-active.npz icluster-apa0-masked.npz icluster-apa1-active.npz icluster-apa1-masked.npz opflash_apa0.tar.gz opflash_apa1.tar.gz rse.json events.txt; do ln -s $SB/work-nuecc48-d123flip/g0/$f $R/g0/$f; done
  p=$A; [ $arm = B ] && p=$B
  log "sbnd arm $arm"; pin $p env SBND_QL_KEEP_ICLUSTER=1 SBND_MAX_JOBS=3 setarch x86_64 -R ./run_chain_group.sh $IN $R data --size 16 --layout perevt --group 0 --from ql > $OUT/sbnd_$arm.log 2>&1; log "rc=$?"
done
for arm in A B; do
  (cd $SB/work-nuecc48-d119sg$arm && find . -type f \( -name '*.npz' -o -name '*.zip' -o -name '*.tar.gz' \) | sort | while read f; do
     echo "$(python3 $W/abtest/hash_archive.py $f | awk '{print $1, $2}') $f"; done) > $OUT/sbnd_hashes_$arm.txt
done
if diff $OUT/sbnd_hashes_A.txt $OUT/sbnd_hashes_B.txt > $OUT/sbnd_diff.txt; then log "sbnd compare PASS $(wc -l < $OUT/sbnd_hashes_A.txt) lines"; else log "sbnd compare FAIL"; fi
# 6. uBooNE
cd $W/qlport/scripts
log "uboone A"; pin $A ./sweep_5384.sh d119subA 6 > $OUT/ub_A.log 2>&1; log "rc=$?"
log "uboone B"; pin $B ./sweep_5384.sh d119subB 6 > $OUT/ub_B.log 2>&1; log "rc=$?"
pin $B ./ab_check.sh d119subB d119subA > $OUT/ub_check.log 2>&1; log "uboone ab_check rc=$?"
log "ALL DONE"
