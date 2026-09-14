#!/bin/bash
# doc qlmatch/29 -- the STM gate (d29/prereg.md sec 3.4) and F1 (sec 3.5): production as shipped on all 120 events of
# pdvd/stm/events.txt, clustering + PR, with the candidate lever given as runner env assignments (ENVS) and/or clustering
# TLA (TLA).  Fork of pdvd/docs/nf_sp_img_clus/scripts/d100r3_arms.sh (that script stays as committed).
#
#   ARM=q29stm ENVS="PDVD_QL_LASSO_LAMBDA=0.3" ./d29_stm_arms.sh      the gate arm (production + lever)
#   ARM=q29flip                                ./d29_stm_arms.sh      F1: production after the default flip, no overrides
#
# Production staging (scripts/stage_ql_tag.sh, default imaging tag), run_clus_evt.sh -calib -save-pctree with
# PDVD_LIGHT_SUFFIX=_keep (window from readout_window_ticks.txt), run_pr_evt.sh -nu -stm-fit.  PDVD_READOUT_NTICKS,
# PDVD_QTOL, PDVD_CLUS_WIRES, PDVD_PR_TLA unset; PDVD_CLUS_TLA=$TLA.  setarch -R, pin prepended, completion by output.
# Existing arm or log dirs are refused (M13).
set -u
ARM=${ARM:?set ARM}; ENVS=${ENVS:-}; TLA=${TLA:-}
PDVD=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
PIN=${PIN:-/home/xqian/tmp/p100/libpin_p100b}
LOGD=/home/xqian/tmp/p29/arm_$ARM
[ -e "$LOGD" ] && { echo "REFUSE existing log dir $LOGD" >&2; exit 2; }
mkdir -p "$LOGD"
[ -s "$PIN/libWireCellClus.so" ] || { echo "no pin $PIN" >&2; exit 2; }
export LD_LIBRARY_PATH="$PIN:${LD_LIBRARY_PATH:-}"
EVS=$(grep -v '^#' "$PDVD/stm/events.txt" | awk 'NF>=2 {printf "%06d_%s\n", $1, $2}')
export ARM PDVD LOGD ENVS TLA
echo "[$ARM] production staging, envs '${ENVS}' tla '${TLA}' start $(date -Is) pin Clus $(md5sum < "$PIN/libWireCellClus.so" | cut -c1-12) events $(echo $EVS | wc -w)" | tee -a "$LOGD/arm.log"
clus_ok() { ls "$1"/pctree-evt*.tar.gz >/dev/null 2>&1 && ls "$1"/pctree-evt*.tlas >/dev/null 2>&1; }
pr_ok()   { [ -s "$1/tracking-pr.root" ] && grep -q "CheckSTM_Michel: .* candidate(s)" "$1"/wct_pr_*.log 2>/dev/null; }
export -f clus_ok pr_ok
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
        (cd "$PDVD" && env -u PDVD_READOUT_NTICKS -u PDVD_QTOL -u PDVD_CLUS_WIRES PDVD_CLUS_TLA="$TLA" $ENVS \
            PDVD_LIGHT_SUFFIX=_keep setarch x86_64 -R ./run_clus_evt.sh -s "$ARM" -calib -save-pctree "$run" "$idx") \
            > "$LOGD/clus_${e}.log" 2>&1; rc=$?
        clus_ok "$d" && echo "ok clus rc=$rc $e" || echo "FAIL clus rc=$rc $e" ;;
      pr)
        clus_ok "$d" || { echo "NOCLUS $e"; return 0; }
        pr_ok "$d" && { echo "skip pr $e"; return 0; }
        (cd "$PDVD" && env -u PDVD_PR_TLA PDVD_LIGHT_SUFFIX=_keep setarch x86_64 -R ./run_pr_evt.sh -nu -stm-fit -s "$ARM" "$run" "$idx") \
            > "$LOGD/pr_${e}.log" 2>&1; rc=$?
        pr_ok "$d" && echo "ok pr rc=$rc $e" || echo "FAIL pr rc=$rc $e" ;;
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
run_stage clus 12
run_stage pr 8
echo "[$ARM] finished $(date -Is) pin Clus $(md5sum < "$PIN/libWireCellClus.so" | cut -c1-12)" | tee -a "$LOGD/arm.log"
