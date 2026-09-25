#!/bin/bash
# doc pdvd/118: drive the production PDVD runners over the run-39305 kaon
# candidates (top drift only => anodes 4-7).  One stage per call:
#   ./run_kaon_chain.sh nfsp|img|light|clus|pr [evt ...]
# Inputs come from convert_kaon.sh (pdvd/input_kaon, via PDVD_INPUT_DATA).
# Per-event logs: pdvd/kaon/logs/<stage>_<evt>.log with a final "rc=N" line.
# Parallel cap: KAON_MAX_JOBS (default 5).  Extra env is passed through to the
# runners (e.g. clus-stage PDVD_* knobs).
KDIR=$(cd "$(dirname "$0")" && pwd)
PDVD=$(dirname "$KDIR")
export PDVD_INPUT_DATA=$PDVD/input_kaon
STAGE=$1; shift
MAXJ=${KAON_MAX_JOBS:-5}
if [ $# -gt 0 ]; then EVTS=("$@"); else
    mapfile -t EVTS < <(ls -d $PDVD_INPUT_DATA/run039305/evt_* | sed 's/.*evt_//' | sort -n)
fi
mkdir -p $KDIR/logs
one() {
    local e=$1 rc=0 a
    cd $PDVD
    case $STAGE in
        nfsp)  for a in 4 5 6 7; do ./run_nf_sp_dnnroi_evt.sh -a $a 39305 $e || { rc=$?; break; }; done ;;
        img)   for a in 4 5 6 7; do ./run_img_evt.sh -a $a 39305 $e || { rc=$?; break; }; done ;;
        light) ./run_light_evt.sh -f $PDVD_INPUT_DATA/light/np02vd_raw_run039305_evt${e}_rawwf.root 39305 $e; rc=$? ;;
        clus)  ./run_clus_evt.sh -a 4,5,6,7 -calib -save-pctree 39305 $e; rc=$? ;;
        pr)    ./run_pr_evt.sh ${KAON_PR_ARGS:-} 39305 $e; rc=$? ;;
        *) echo "unknown stage $STAGE"; rc=9 ;;
    esac
    echo "rc=$rc"
    return $rc
}
nfail=0
for e in "${EVTS[@]}"; do
    while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do wait -n || nfail=$((nfail+1)); done
    one $e > $KDIR/logs/${STAGE}_$e.log 2>&1 &
done
while [ $(jobs -rp | wc -l) -gt 0 ]; do wait -n || nfail=$((nfail+1)); done
echo "run_kaon_chain $STAGE: ${#EVTS[@]} events, $nfail failed"
grep -H '^rc=' $(for e in "${EVTS[@]}"; do echo $KDIR/logs/${STAGE}_$e.log; done)
exit $nfail
