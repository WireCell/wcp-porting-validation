#!/bin/bash
# doc pdvd/37 R1 -- run one PDVD PR arm of the Steiner terminal thinning round.
#
# Usage: ./doc37_run_thinning_arms.sh <tag> <libdir> <sep_cm> [max_jobs]
#   tag      an ALREADY-STAGED work tag (scripts/stage_pr_tag.sh ... d34base)
#   libdir   LD_LIBRARY_PATH pin.  The tree is shared and a peer rebuilds
#            local/lib mid-campaign, so every arm names the binary it ran
#            (see the doc's Repro block for the two md5s).
#   sep_cm   steiner_terminal_min_sep_cm.  0 = the pre-flip arm.
#
# All arms pass -S dl_weights='' : the DL/SCN vertex is not bit-stable (M4), so
# it is the only configuration in which an OFF/OFF pair can be a byte-identity
# gate AND an OFF/ON pair can attribute a difference to the knob.  These arms
# are therefore NOT the production vertex configuration; they isolate the knob.
set -u
TAG=${1:?usage: doc37_run_thinning_arms.sh <tag> <libdir> <sep_cm> [max_jobs]}
LIBDIR=${2:?libdir}
SEPCM=${3:?sep_cm}
JOBS=${4:-16}
PDVD=$(cd "$(dirname "$0")/../../.." && pwd)

export LD_LIBRARY_PATH="$LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PDVD_MAX_JOBS="$JOBS"
export PDVD_PR_TLA="-S dl_weights='' -S steiner_terminal_min_sep_cm=$SEPCM"

cd "$PDVD"
for run in 039252 039253 039349; do
    ls -d "work/${run}_"*"_${TAG}" >/dev/null 2>&1 || { echo "skip $run (no staged dirs)"; continue; }
    echo "=== $TAG run=$run sep_cm=$SEPCM lib=$LIBDIR ==="
    ./run_pr_evt.sh -s "$TAG" "$run" all 0
    echo "   rc=$?"
done

n_done=$(ls work/*_"${TAG}"/calib-pr-evt*.json 2>/dev/null | wc -l)
n_dir=$(ls -d work/*_"${TAG}" 2>/dev/null | wc -l)
echo "=== $TAG: $n_done / $n_dir events produced a calib-pr dump ==="
