#!/bin/bash
# doc pdvd/37 R1 -- run one PDVD PR arm of the Steiner terminal thinning round.
#
# Usage: ./doc37_run_thinning_arms.sh <tag> <libdir> <sep_cm> [max_jobs] [dl]
#   tag      an ALREADY-STAGED work tag (scripts/stage_pr_tag.sh ... d34base)
#   libdir   LD_LIBRARY_PATH pin.  The tree is shared and a peer rebuilds
#            local/lib mid-campaign, so every arm names the binary it ran
#            (see the doc's Repro block for the two md5s).
#   sep_cm   steiner_terminal_min_sep_cm.  0 = the pre-flip arm.
#   dl       "geom" (default) forces -S dl_weights='' ; "dl" leaves the driver
#            default, i.e. the uBooNE SCN vertex that has been PDVD production
#            since 2026-09-04 (doc 28 sec 27).
#
# WHICH ONE TO USE.  geom is the ONLY configuration in which an OFF/OFF pair can
# be a byte-identity gate and an OFF/ON pair can attribute a difference to the
# knob, because DL inference is not bit-stable (M4) -- round 2 sec 12/13 used it.
# But it is NOT what production runs, so a "does this cost physics" answer taken
# there does not transfer.  dl grades the knob where production actually lives,
# at the price that the comparison is statistical: pair every dl arm with a
# REPEAT of its own OFF side to measure DL's own noise floor, or the churn DL
# generates by itself will be read as the knob's effect (round 3, sec 15).
set -u
TAG=${1:?usage: doc37_run_thinning_arms.sh <tag> <libdir> <sep_cm> [max_jobs] [geom|dl]}
LIBDIR=${2:?libdir}
SEPCM=${3:?sep_cm}
JOBS=${4:-16}
DLMODE=${5:-geom}
PDVD=$(cd "$(dirname "$0")/../../.." && pwd)

export LD_LIBRARY_PATH="$LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PDVD_MAX_JOBS="$JOBS"
case "$DLMODE" in
    geom) export PDVD_PR_TLA="-S dl_weights='' -S steiner_terminal_min_sep_cm=$SEPCM" ;;
    dl)   export PDVD_PR_TLA="-S steiner_terminal_min_sep_cm=$SEPCM" ;;
    *)    echo "ERROR: dl mode must be 'geom' or 'dl', got '$DLMODE'" >&2; exit 2 ;;
esac

cd "$PDVD"
for run in 039252 039253 039349; do
    ls -d "work/${run}_"*"_${TAG}" >/dev/null 2>&1 || { echo "skip $run (no staged dirs)"; continue; }
    echo "=== $TAG run=$run sep_cm=$SEPCM vertex=$DLMODE lib=$LIBDIR ==="
    ./run_pr_evt.sh -s "$TAG" "$run" all 0
    echo "   rc=$?"
done

n_done=$(ls work/*_"${TAG}"/calib-pr-evt*.json 2>/dev/null | wc -l)
n_dir=$(ls -d work/*_"${TAG}" 2>/dev/null | wc -l)
echo "=== $TAG: $n_done / $n_dir events produced a calib-pr dump ==="
