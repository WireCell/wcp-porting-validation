#!/bin/bash
# doc pdvd/99 -- NF + SP + DNN-ROI + L1SP (production runner defaults) for every event of
# pdvd/stm/events.txt into work/<run6>_<idx>_<ARM>/, reading the raw frames IN PLACE from
# pdvd/input_data (xning's tree; nothing is copied).
#
#   ARM=p98voff                                   d99_sp_arm.sh
#   ARM=p98von  EXTRA="--top-gain-scale 0.889"    d99_sp_arm.sh
#
# Every job runs under `setarch x86_64 -R` (G2 showed the frames are then bit-reproducible).
# run_nf_sp_dnnroi_evt.sh:26 PREPENDS ${WCT_BASE}/local/lib itself, so an outer pin cannot take
# effect for SP; instead every local/lib libWireCell*.so is md5-summed at start and at end and the
# driver reports any change (SP loads no libWireCellClus, the only library that differs from
# libpin_p96).  An event already complete (8 frame archives and "DNN-ROI done" in its log) is
# skipped, never overwritten.
set -u
ARM=${ARM:?set ARM}
EXTRA=${EXTRA:-}
JOBS=${JOBS:-8}
PDVD=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
LIB=/nfs/data/1/xqian/toolkit-dev/local/lib
LOGD=/home/xqian/tmp/p98/sp_$ARM
mkdir -p "$LOGD"
md5sum "$LIB"/libWireCell*.so > "$LOGD/libs_start.md5"
echo "ARM=$ARM EXTRA=[$EXTRA] JOBS=$JOBS start $(date -Is)" | tee "$LOGD/driver.log"

one() {
    local run=$1 idx=$2 r6 d n
    r6=$(printf %06d "$run")
    d=$PDVD/work/${r6}_${idx}_$ARM
    n=$(ls "$d"/protodune-sp-dnnroi-frames-anode*.tar.bz2 2>/dev/null | wc -l)
    if [ "$n" = 8 ] && grep -q "DNN-ROI done" "$LOGD/${r6}_${idx}.log" 2>/dev/null; then
        echo "skip ${r6}_${idx}"; return 0
    fi
    (cd "$PDVD" && setarch x86_64 -R ./run_nf_sp_dnnroi_evt.sh $EXTRA -O "_$ARM" "$run" "$idx") \
        > "$LOGD/${r6}_${idx}.log" 2>&1
    local rc=$?
    n=$(ls "$d"/protodune-sp-dnnroi-frames-anode*.tar.bz2 2>/dev/null | wc -l)
    echo "rc=$rc frames=$n ${r6}_${idx} $(grep -c 'DNN-ROI done' "$LOGD/${r6}_${idx}.log")"
}
export -f one
export ARM EXTRA PDVD LOGD

grep -v '^#' "$PDVD/stm/events.txt" | awk 'NF>=2 {print $1, $2}' \
    | xargs -P "$JOBS" -n 2 bash -c 'one "$@"' _ >> "$LOGD/driver.log" 2>&1

md5sum "$LIB"/libWireCell*.so > "$LOGD/libs_end.md5"
if cmp -s "$LOGD/libs_start.md5" "$LOGD/libs_end.md5"; then libs=UNCHANGED; else libs=CHANGED; fi
ok=$(grep -c '^rc=0 frames=8 .* 1$\|^skip ' "$LOGD/driver.log")
echo "done $(date -Is) complete=$ok/120 local_libs=$libs" | tee -a "$LOGD/driver.log"
