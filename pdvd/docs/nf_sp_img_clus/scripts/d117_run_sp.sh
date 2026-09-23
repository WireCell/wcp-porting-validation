#!/bin/bash
# doc pdvd/117: production-default SP for the events holding doc 98's latest-arm Michel candidates, into
# work/<run6>_<idx>_<ARM>.  ARM=d117sp (no EXTRA) reproduces p98von (doc pdvd/100 F2; proven by d117_cell_identity.py).
# ARM=d117w3sp EXTRA="--wire-col-x 3.0" is study S4.  setarch -R; 6 jobs; never overwrites a complete event.
#   ARM=d117w3sp EXTRA="--wire-col-x 3.0" ./d117_run_sp.sh 039252_0 039252_1 ...
ARM=${ARM:-d117sp}; EXTRA=${EXTRA:-}
PDVD=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
LOGD=/home/xqian/tmp/d117/sp_logs_$ARM; mkdir -p $LOGD
md5sum /nfs/data/1/xqian/toolkit-dev/local/lib/libWireCell*.so > $LOGD/libs_start.md5
echo "ARM=$ARM EXTRA=[$EXTRA] start $(date -Is)"
one() {
    local key=$1 run=${1%_*} idx=${1##*_}
    local d=$PDVD/work/${run}_${idx}_$ARM
    if [ "$(ls $d/protodune-sp-dnnroi-frames-anode*.tar.bz2 2>/dev/null | wc -l)" = 8 ]; then echo "skip $key"; return 0; fi
    (cd $PDVD && setarch x86_64 -R ./run_nf_sp_dnnroi_evt.sh $EXTRA -O _$ARM $((10#$run)) $idx) > $LOGD/$key.log 2>&1
    echo "rc=$? frames=$(ls $d/protodune-sp-dnnroi-frames-anode*.tar.bz2 2>/dev/null | wc -l) $key"
}
export -f one; export PDVD LOGD ARM EXTRA
echo "$@" | tr ' ' '\n' | xargs -P 6 -I{} bash -c 'one {}'
md5sum /nfs/data/1/xqian/toolkit-dev/local/lib/libWireCell*.so > $LOGD/libs_end.md5
cmp -s $LOGD/libs_start.md5 $LOGD/libs_end.md5 && echo "libs unchanged" || echo "LIBS CHANGED DURING RUN"
