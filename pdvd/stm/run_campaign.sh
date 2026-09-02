#!/bin/bash
# doc pdvd/25: the stopping-muon / Michel data campaign over the 120 events of
# record (stm/events.txt), as one fresh tag arm.
#
# Usage: ./stm/run_campaign.sh <tag> [stage|clus|pr|all]      default all
#   stage : stage work/<RUN6>_<idx>_<tag>/ from _keep (scripts/stage_ql_tag.sh; refuses existing tags)
#   clus  : run_clus_evt.sh -s <tag> -calib -save-pctree  (Q/L + pctree + sidecar; light from _keep)
#   pr    : run_pr_evt.sh -s <tag> -stm-fit               (full PDVD PR chain + tracking-stm.root)
# Env: PDVD_MAX_JOBS (default 6), STM_RUNS="39252 39253 39349" to restrict runs,
#      STM_PR_MODE=-stm for the cosmic-tagger-only chain (default: the full -nu chain).
# Every step logs to stm/logs/<tag>/ and the per-event runner logs live in work/.
set -o pipefail
PDVD_DIR=$(cd "$(dirname "$0")/.." && pwd)
TAG=${1:?tag}; STEP=${2:-all}
export PDVD_MAX_JOBS=${PDVD_MAX_JOBS:-6}
export PDVD_LIGHT_SUFFIX=${PDVD_LIGHT_SUFFIX:-_keep}
RUNS=${STM_RUNS:-"39252 39253 39349"}
LOGD="$PDVD_DIR/stm/logs/$TAG"; mkdir -p "$LOGD"
cd "$PDVD_DIR" || exit 1

do_stage() {
    local n=0
    while read -r run idx evt light; do
        [ "${run:0:1}" = "#" ] && continue
        case " $RUNS " in *" $run "*) ;; *) continue ;; esac
        ./scripts/stage_ql_tag.sh "$run" "$idx" "$TAG" >> "$LOGD/stage.log" 2>&1 && n=$((n+1))
    done < stm/events.txt
    echo "staged $n event dirs for tag $TAG (see $LOGD/stage.log)"
}
do_clus() {
    for run in $RUNS; do
        echo "== clus run $run ($(date +%T))"
        ./run_clus_evt.sh -s "$TAG" -calib -save-pctree "$run" all > "$LOGD/clus_$run.log" 2>&1
        echo "   rc=$? ; $(grep -c 'Clustering done' "$LOGD/clus_$run.log") done lines; pctrees: $(ls work/$(printf '%06d' $((10#$run)))_*_$TAG/pctree-evt*.tar.gz 2>/dev/null | wc -l)"
    done
}
do_pr() {
    for run in $RUNS; do
        echo "== pr run $run ($(date +%T))"
        ./run_pr_evt.sh -s "$TAG" ${STM_PR_MODE:-} -stm-fit "$run" all > "$LOGD/pr_$run.log" 2>&1
        echo "   rc=$? ; $(grep -c 'PR done' work/.batch_pr_$(printf '%06d' $((10#$run)))_*_$TAG.log 2>/dev/null | awk -F: '{s+=$2}END{print s+0}') done; calib-pr: $(ls work/$(printf '%06d' $((10#$run)))_*_$TAG/calib-pr-evt*.json 2>/dev/null | wc -l); stm root: $(ls work/$(printf '%06d' $((10#$run)))_*_$TAG/tracking-stm.root 2>/dev/null | wc -l)"
    done
}
case "$STEP" in
    stage) do_stage ;;
    clus) do_clus ;;
    pr) do_pr ;;
    all) do_stage; do_clus; do_pr ;;
    *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
echo "campaign step '$STEP' finished for tag $TAG ($(date +%T)); loadavg $(cut -d' ' -f1 /proc/loadavg)"
