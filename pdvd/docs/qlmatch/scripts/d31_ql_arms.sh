#!/bin/bash
# doc qlmatch/31 -- Q/L matching of the 120 production events (pdvd/stm/events.txt) on a chosen light arm.
# Fork of d29_stm_arms.sh (that script stays as committed): production staging (scripts/stage_ql_tag.sh, default
# imaging tag), run_clus_evt.sh -calib with PDVD_LIGHT_SUFFIX=$LIGHT, clustering + QL only (no PR, no pctree).
#
#   ARM=q31ctl LIGHT=_g31off ./d31_ql_arms.sh          control: production with the knob-off light (== _keep, gate g31off)
#   ARM=q31tot LIGHT=_q31tot ./d31_ql_arms.sh          ToT light, everything else identical
#   ARM=q31toti LIGHT=_q31tot ENVS="PDVD_QL_CHI2_SAT_INFLATE=x" ./d31_ql_arms.sh   (only if pre-registered sec 5 says so)
#
# PDVD_READOUT_NTICKS, PDVD_QTOL, PDVD_CLUS_WIRES unset as in d29; setarch -R; clustering pin prepended (same pin as
# q29flip); existing arm or log dirs are refused (M13).
set -u
ARM=${ARM:?set ARM}; LIGHT=${LIGHT:?set LIGHT}; ENVS=${ENVS:-}
PDVD=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
PIN=${PIN:-/home/xqian/tmp/p100/libpin_p100b}
LOGD=/home/xqian/tmp/p31/arm_$ARM
[ -e "$LOGD" ] && { echo "REFUSE existing log dir $LOGD" >&2; exit 2; }
mkdir -p "$LOGD"
[ -s "$PIN/libWireCellClus.so" ] || { echo "no pin $PIN" >&2; exit 2; }
export LD_LIBRARY_PATH="$PIN:${LD_LIBRARY_PATH:-}"
EVS=$(grep -v '^#' "$PDVD/stm/events.txt" | awk 'NF>=2 {printf "%06d_%s\n", $1, $2}')
export ARM LIGHT PDVD LOGD ENVS
echo "[$ARM] light '$LIGHT' envs '${ENVS}' start $(date -Is) pin Clus $(md5sum < "$PIN/libWireCellClus.so" | cut -c1-12) events $(echo $EVS | wc -w)" | tee -a "$LOGD/arm.log"
clus_ok() { ls "$1"/calib-evt*.json* >/dev/null 2>&1 && ls "$1"/mabc-*.zip >/dev/null 2>&1; }
export -f clus_ok
one() {   # stage run idx
    local st=$1 run=$2 idx=$3 e d rc
    e=$(printf %06d "$run")_$idx; d=$PDVD/work/${e}_$ARM
    case $st in
      stage)
        [ -e "$d" ] && { echo "REFUSE $d"; return 0; }
        (cd "$PDVD" && ./scripts/stage_ql_tag.sh "$run" "$idx" "$ARM") > "$LOGD/stage_${e}.log" 2>&1; rc=$?
        [ "$rc" = 0 ] && echo "staged $e" || echo "FAIL stage rc=$rc $e" ;;
      clus)
        clus_ok "$d" && { echo "skip clus $e"; return 0; }
        (cd "$PDVD" && env -u PDVD_READOUT_NTICKS -u PDVD_QTOL -u PDVD_CLUS_WIRES $ENVS \
            PDVD_LIGHT_SUFFIX="$LIGHT" setarch x86_64 -R ./run_clus_evt.sh -s "$ARM" -calib "$run" "$idx") \
            > "$LOGD/clus_${e}.log" 2>&1; rc=$?
        clus_ok "$d" && echo "ok clus rc=$rc $e" || echo "FAIL clus rc=$rc $e" ;;
    esac
}
export -f one
run_stage() {
    echo "$EVS" | tr ' ' '\n' | awk 'NF {split($1,a,"_"); print a[1]+0, a[2]}' \
        | xargs -P "$2" -n 2 bash -c "one $1 \"\$@\"" _ >> "$LOGD/$1.status" 2>&1
    echo "[$ARM] $1 done $(date -Is): ok $(grep -c '^ok \|^skip \|^staged' "$LOGD/$1.status") FAIL $(grep -c '^FAIL' "$LOGD/$1.status") no-input $(grep -c '^NO\|^REFUSE' "$LOGD/$1.status") loadavg $(cut -d' ' -f1 /proc/loadavg)" | tee -a "$LOGD/arm.log"
}
run_stage stage 8
grep -q '^FAIL\|^REFUSE' "$LOGD/stage.status" && { echo "[$ARM] staging problems, not running" | tee -a "$LOGD/arm.log"; exit 1; }
run_stage clus ${PDVD_MAX_JOBS:-6}
echo "[$ARM] finished $(date -Is)" | tee -a "$LOGD/arm.log"
