#!/bin/bash
# doc pdvd/25 sec 13.7: the full PDVD PR chain (tagger_check_neutrino + Michel
# finder) on a SUBSET of events, into tag <tag> seeded from tag <from>'s pctrees.
# Usage: ./stm/run_michel_subset.sh <from_tag> <tag> <events.txt> [jobs]
#   events.txt: one <RUN6>_<idx> per line (e.g. the events of stm/sample_index.tsv)
# Env: PDVD_PR_TLA extra knobs (default: nu_per_bundle_min_length=50 -- the
#      per-bundle neutrino PR is the expensive stage on PDVD, doc 25 sec 13.9).
set -o pipefail
PDVD_DIR=$(cd "$(dirname "$0")/.." && pwd); cd "$PDVD_DIR" || exit 1
FROM=${1:?from_tag}; TAG=${2:?tag}; LIST=${3:?events.txt}; JOBS=${4:-6}
export PDVD_PR_TLA=${PDVD_PR_TLA:-"-S nu_per_bundle_min_length=50"}
. "$PDVD_DIR/_runlib.sh"; export PDVD_MAX_JOBS=$JOBS; batch_init
mkdir -p stm/logs/$TAG
while read -r ev; do
    [ -z "$ev" ] || [ "${ev:0:1}" = "#" ] && continue
    run=${ev%%_*}; idx=${ev#*_}; src=work/${ev}_$FROM; dst=work/${ev}_$TAG
    [ -d "$src" ] || { echo "no $src" >&2; continue; }
    mkdir -p "$dst"; for f in "$src"/pctree-evt*.tar.gz; do ln -sfn "../$(basename $src)/$(basename $f)" "$dst/"; done; cp -p "$src"/pctree-evt*.tlas "$dst/"
    batch_wait_slot
    ( ./run_pr_evt.sh -s "$TAG" -stm-fit "$((10#$run))" "$idx" ) > "work/.batch_pr_${ev}_$TAG.log" 2>&1 &
    BATCH_PIDS[$!]=$ev
done < "$LIST"
batch_drain; batch_summary
echo "michel subset done $(date +%T): calib=$(ls work/*_$TAG/calib-pr-evt*.json 2>/dev/null | wc -l) failed=$(grep -l 'ERROR: wire-cell' work/.batch_pr_*_$TAG.log 2>/dev/null | wc -l)"
