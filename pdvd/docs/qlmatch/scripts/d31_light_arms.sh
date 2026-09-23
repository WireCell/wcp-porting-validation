#!/bin/bash
# doc qlmatch/31 -- light reco of the 120 production events (pdvd/stm/events.txt) under one arm.
#   ARM=_g31old PIN=/home/xqian/tmp/p31/libpin_flash_old ./d31_light_arms.sh   pre-change libWireCellFlash (gate A side)
#   ARM=_g31off  ./d31_light_arms.sh                                            new lib, knob off (gate B side; QL control light)
#   ARM=_g31offb ./d31_light_arms.sh                                            null pair of _g31off
#   ARM=_q31tot ENVS="PDVD_SAT_REPAIR_MODE=tot" ./d31_light_arms.sh             new lib, ToT fill
# Every arm uses the argument set of the production light _keep (= runner defaults with PDVD_FLASH_TAIL_MERGE=0;
# compiled config equal to work/*_keep/.wct-light.json apart from _pnode metadata, doc 31 sec 4).  The raw file of
# each event is the one its _keep config read.  setarch -R; at most PDVD_MAX_JOBS (6) at once; existing arm dirs refused.
#   ARM=_f31ts18 ONLY_RUN=39252 ENVS="PDVD_LIGHT_FRAMES=1" ./d31_light_arms.sh    frames dump, run 039252 only (sec 5)
set -u
ARM=${ARM:?set ARM}; ENVS=${ENVS:-}; PIN=${PIN:-}; ONLY_RUN=${ONLY_RUN:-}
PDVD=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
LOGD=/home/xqian/tmp/p31/light$ARM
[ -e "$LOGD" ] && { echo "REFUSE existing log dir $LOGD" >&2; exit 2; }
mkdir -p "$LOGD"
[ -n "$PIN" ] && { [ -s "$PIN/libWireCellFlash.so" ] || { echo "no pin $PIN" >&2; exit 2; }; }
export ARM ENVS PIN PDVD LOGD
echo "[$ARM] start $(date -Is) envs '$ENVS' pin '${PIN:-none}' flash $(md5sum < "${PIN:-$PDVD/../../local/lib}/libWireCellFlash.so" | cut -c1-12)" | tee "$LOGD/arm.log"
one() {
    local run=$1 evt=$2 r6 keep raw d rc
    r6=$(printf %06d "$run"); keep=$PDVD/work/${r6}_light${evt}_keep/.wct-light.json
    d=$PDVD/work/${r6}_light${evt}$ARM
    [ -e "$d" ] && { echo "REFUSE $d"; return 0; }
    raw=$(python3 -c "import json,sys; print(next(n['data']['filename'] for n in json.load(open(sys.argv[1])) if n.get('type')=='PDVDOpWaveformSource' or 'filename' in n.get('data',{})))" "$keep") || { echo "FAIL raw $r6 $evt"; return 0; }
    (cd "$PDVD" && env ${PIN:+LD_LIBRARY_PATH=$PIN:${LD_LIBRARY_PATH:-}} PDVD_FLASH_TAIL_MERGE=0 $ENVS \
        setarch x86_64 -R ./run_light_evt.sh -f "$raw" -s "$ARM" "$run" "$evt") > "$LOGD/${r6}_${evt}.log" 2>&1; rc=$?
    [ "$rc" = 0 ] && [ -s "$d/opflash_pdvd-wct.tar.gz" ] && echo "ok $r6 $evt" || echo "FAIL rc=$rc $r6 $evt"
}
export -f one
grep -v '^#' "$PDVD/stm/events.txt" | awk -v r="$ONLY_RUN" 'NF>=3 && (r == "" || $1 == r) {print $1, $3}' | xargs -P "${PDVD_MAX_JOBS:-6}" -n 2 bash -c 'one "$@"' _ | tee -a "$LOGD/arm.log"
echo "[$ARM] done $(date -Is) ok $(grep -c '^ok' "$LOGD/arm.log") fail $(grep -c '^FAIL' "$LOGD/arm.log")" | tee -a "$LOGD/arm.log"
