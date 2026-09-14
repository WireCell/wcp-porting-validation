#!/bin/bash
# doc pdvd/100 round 3 -- the QtoL arms (fork of d100_arms.sh WAVE=flip; that script stays as committed).
#
#   ARM=p101q QTOL=0.0783 ./d100r3_arms.sh      the measurement arm: production as shipped + PDVD_QTOL
#   ARM=p101flip          ./d100r3_arms.sh      F1: production as shipped after the driver default flip, no overrides
#
# Production as shipped = stm/run_campaign.sh's staging (scripts/stage_ql_tag.sh, its default imaging tag), then
# run_clus_evt.sh -calib -save-pctree with PDVD_LIGHT_SUFFIX=_keep (the window from readout_window_ticks.txt), then
# run_pr_evt.sh -nu -stm-fit.  PDVD_READOUT_NTICKS, PDVD_CLUS_TLA and PDVD_PR_TLA are unset; PDVD_QTOL is set only from
# QTOL.  Per event, setarch -R, the pin prepended, completion judged by output (pctree + tlas; tracking-pr.root +
# "CheckSTM_Michel: N candidate(s)").  Existing arm or log dirs are refused (M13).
set -u
ARM=${ARM:?set ARM}; QTOL=${QTOL:-}
PDVD=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
PIN=${PIN:-/home/xqian/tmp/p100/libpin_p100b}
LOGD=/home/xqian/tmp/p101/arm_$ARM
[ -e "$LOGD" ] && { echo "REFUSE existing log dir $LOGD" >&2; exit 2; }
mkdir -p "$LOGD"
[ -s "$PIN/libWireCellClus.so" ] || { echo "no pin $PIN" >&2; exit 2; }
export LD_LIBRARY_PATH="$PIN:${LD_LIBRARY_PATH:-}"
[ "$(pgrep -c '^wire-cell')" = 0 ] || echo "[$ARM] note: $(pgrep -c '^wire-cell') wire-cell job(s) already live"
if [ -n "${EVENTS:-}" ]; then EVS=$EVENTS; else
    EVS=$(grep -v '^#' "$PDVD/stm/events.txt" | awk 'NF>=2 {printf "%06d_%s\n", $1, $2}'); fi
export ARM PDVD LOGD QTOL
PIN_MD5=$(md5sum < "$PIN/libWireCellClus.so" | cut -c1-12)
echo "[$ARM] production staging, PDVD_QTOL '${QTOL}' start $(date -Is) pin Clus $PIN_MD5 events $(echo $EVS | wc -w)" | tee -a "$LOGD/arm.log"
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
        if [ -n "$QTOL" ]; then
            (cd "$PDVD" && env -u PDVD_READOUT_NTICKS -u PDVD_CLUS_TLA PDVD_QTOL="$QTOL" PDVD_LIGHT_SUFFIX=_keep \
                setarch x86_64 -R ./run_clus_evt.sh -s "$ARM" -calib -save-pctree "$run" "$idx") > "$LOGD/clus_${e}.log" 2>&1; rc=$?
        else
            (cd "$PDVD" && env -u PDVD_READOUT_NTICKS -u PDVD_CLUS_TLA -u PDVD_QTOL PDVD_LIGHT_SUFFIX=_keep \
                setarch x86_64 -R ./run_clus_evt.sh -s "$ARM" -calib -save-pctree "$run" "$idx") > "$LOGD/clus_${e}.log" 2>&1; rc=$?
        fi
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
run_stage() {   # stage jobs
    echo "$EVS" | tr ' ' '\n' | awk 'NF {split($1,a,"_"); print a[1]+0, a[2]}' \
        | xargs -P "$2" -n 2 bash -c "one $1 \"\$@\"" _ >> "$LOGD/$1.status" 2>&1
    echo "[$ARM] $1 done $(date -Is): ok $(grep -c '^ok \|^skip \|^staged' "$LOGD/$1.status") FAIL $(grep -c '^FAIL' "$LOGD/$1.status") no-input $(grep -c '^NO\|^REFUSE' "$LOGD/$1.status") loadavg $(cut -d' ' -f1 /proc/loadavg)" | tee -a "$LOGD/arm.log"
}
run_stage stage 8
grep -q '^FAIL\|^REFUSE' "$LOGD/stage.status" && { echo "[$ARM] staging problems, not running" | tee -a "$LOGD/arm.log"; exit 1; }
run_stage clus 12
run_stage pr 8
echo "[$ARM] finished $(date -Is) pin Clus $(md5sum < "$PIN/libWireCellClus.so" | cut -c1-12) (start $PIN_MD5)" | tee -a "$LOGD/arm.log"
