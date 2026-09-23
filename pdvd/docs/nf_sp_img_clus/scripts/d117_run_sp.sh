#!/bin/bash
# doc pdvd/117: production-default SP (== p98von frames per doc pdvd/100 F2) for the 61 events holding doc 98's
# latest-arm Michel candidates, into work/<run6>_<idx>_d117sp.  setarch -R; 6 jobs; never overwrites a complete event.
PDVD=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
LOGD=/home/xqian/tmp/d117/sp_logs
md5sum /nfs/data/1/xqian/toolkit-dev/local/lib/libWireCell*.so > $LOGD/libs_start.md5
one() {
    local key=$1 run=${1%_*} idx=${1##*_}
    local d=$PDVD/work/${run}_${idx}_d117sp
    if [ "$(ls $d/protodune-sp-dnnroi-frames-anode*.tar.bz2 2>/dev/null | wc -l)" = 8 ]; then echo "skip $key"; return 0; fi
    (cd $PDVD && setarch x86_64 -R ./run_nf_sp_dnnroi_evt.sh -O _d117sp $((10#$run)) $idx) > $LOGD/$key.log 2>&1
    echo "rc=$? frames=$(ls $d/protodune-sp-dnnroi-frames-anode*.tar.bz2 2>/dev/null | wc -l) $key"
}
export -f one; export PDVD LOGD
echo "$@" | tr ' ' '\n' | xargs -P 6 -I{} bash -c 'one {}'
md5sum /nfs/data/1/xqian/toolkit-dev/local/lib/libWireCell*.so > $LOGD/libs_end.md5
cmp -s $LOGD/libs_start.md5 $LOGD/libs_end.md5 && echo "libs unchanged" || echo "LIBS CHANGED DURING RUN"
