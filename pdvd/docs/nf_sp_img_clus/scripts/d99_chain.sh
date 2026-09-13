#!/bin/bash
# doc pdvd/99 -- imaging -> clustering (+Q/L, calib, pctree, assoc) -> PR (-nu -stm-fit, production
# wct-pr-perevt.jsonnet, no TLA) for the 120 events of pdvd/stm/events.txt on an SP arm written by
# d99_sp_arm.sh, all in work/<run6>_<idx>_<ARM>/.
#
#   ARM=p98voff d99_chain.sh [img|clus|pr|all]
#
# Every stage is per event (not the runners' `all`, whose discovery reaches input_data events outside
# the 120), under `setarch x86_64 -R`, with the production library pin prepended (the imaging,
# clustering and PR runners do not prepend local/lib themselves; run_nf_sp_dnnroi_evt.sh does).
# A stage refuses to start unless the previous one is complete for every event it needs; a complete
# event is skipped, never overwritten.  Completion is judged by output, not rc:
#   img   16 clusters-apa-anode*-ms-{active,masked}.tar.gz
#   clus  pctree-evt*.tar.gz + pctree-evt*.tlas
#   pr    tracking-pr.root non-empty and "CheckSTM_Michel: .* candidate(s)" in the PR log
#         (039252_11 has no STM-tagged main cluster: expected incomplete, as on p96vprod)
set -u
ARM=${ARM:?set ARM}
STAGE=${1:-all}
PDVD=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
PIN=/home/xqian/tmp/p96/libpin_p96
LOGD=/home/xqian/tmp/p98/chain_$ARM
mkdir -p "$LOGD"
[ -d "$PIN" ] || { echo "no pin $PIN" >&2; exit 2; }
export LD_LIBRARY_PATH="$PIN:${LD_LIBRARY_PATH:-}"
export ARM PDVD LOGD
EVENTS=$(grep -v '^#' "$PDVD/stm/events.txt" | awk 'NF>=2 {print $1, $2}')

img_ok()  { [ "$(ls "$1"/clusters-apa-anode*-ms-active.tar.gz "$1"/clusters-apa-anode*-ms-masked.tar.gz 2>/dev/null | wc -l)" = 16 ]; }
clus_ok() { ls "$1"/pctree-evt*.tar.gz >/dev/null 2>&1 && ls "$1"/pctree-evt*.tlas >/dev/null 2>&1; }
pr_ok()   { [ -s "$1/tracking-pr.root" ] && grep -q "CheckSTM_Michel: .* candidate(s)" "$1"/wct_pr_*.log 2>/dev/null; }
export -f img_ok clus_ok pr_ok

one() {
    local stage=$1 run=$2 idx=$3 r6 d rc
    r6=$(printf %06d "$run"); d=$PDVD/work/${r6}_${idx}_$ARM
    case $stage in
      img)
        [ "$(ls "$d"/protodune-sp-dnnroi-frames-anode*.tar.bz2 2>/dev/null | wc -l)" = 8 ] || { echo "NOSP ${r6}_${idx}"; return 0; }
        img_ok "$d" && { echo "skip img ${r6}_${idx}"; return 0; }
        (cd "$PDVD" && setarch x86_64 -R ./run_img_evt.sh -O "_$ARM" "$run" "$idx") > "$LOGD/img_${r6}_${idx}.log" 2>&1; rc=$?
        img_ok "$d" && echo "ok img rc=$rc ${r6}_${idx}" || echo "FAIL img rc=$rc ${r6}_${idx}" ;;
      clus)
        img_ok "$d" || { echo "NOIMG ${r6}_${idx}"; return 0; }
        clus_ok "$d" && { echo "skip clus ${r6}_${idx}"; return 0; }
        (cd "$PDVD" && env PDVD_LIGHT_SUFFIX=_keep setarch x86_64 -R ./run_clus_evt.sh -s "$ARM" -calib -save-pctree "$run" "$idx") \
            > "$LOGD/clus_${r6}_${idx}.log" 2>&1; rc=$?
        clus_ok "$d" && echo "ok clus rc=$rc ${r6}_${idx}" || echo "FAIL clus rc=$rc ${r6}_${idx}" ;;
      pr)
        clus_ok "$d" || { echo "NOCLUS ${r6}_${idx}"; return 0; }
        pr_ok "$d" && { echo "skip pr ${r6}_${idx}"; return 0; }
        (cd "$PDVD" && env PDVD_LIGHT_SUFFIX=_keep PDVD_PR_TLA="" setarch x86_64 -R ./run_pr_evt.sh -nu -stm-fit -s "$ARM" "$run" "$idx") \
            > "$LOGD/pr_${r6}_${idx}.log" 2>&1; rc=$?
        pr_ok "$d" && echo "ok pr rc=$rc ${r6}_${idx}" || echo "FAIL pr rc=$rc ${r6}_${idx}" ;;
    esac
}
export -f one

run_stage() {
    local st=$1 jobs=$2
    echo "[$ARM] stage $st jobs=$jobs start $(date -Is) pin Clus $(md5sum < $PIN/libWireCellClus.so | cut -c1-12)" | tee -a "$LOGD/chain.log"
    echo "$EVENTS" | xargs -P "$jobs" -n 2 bash -c "one $st \"\$@\"" _ >> "$LOGD/$st.status" 2>&1
    echo "[$ARM] stage $st done $(date -Is): ok $(grep -c '^ok \|^skip ' "$LOGD/$st.status") FAIL $(grep -c '^FAIL' "$LOGD/$st.status") missing-input $(grep -c '^NO' "$LOGD/$st.status") loadavg $(cut -d' ' -f1 /proc/loadavg)" | tee -a "$LOGD/chain.log"
}

case $STAGE in
  img)  run_stage img 6 ;;
  clus) run_stage clus 12 ;;
  pr)   run_stage pr 5 ;;
  all)  run_stage img 6; run_stage clus 12; run_stage pr 5 ;;
  *) echo "stage must be img|clus|pr|all" >&2; exit 2 ;;
esac
echo "[$ARM] chain $STAGE finished $(date -Is)" | tee -a "$LOGD/chain.log"
