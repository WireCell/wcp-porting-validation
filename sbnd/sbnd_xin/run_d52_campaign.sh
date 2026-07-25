#!/usr/bin/env bash
# Doc 52 A/B campaign: the isolated-grouping provenance (assoc_cluster_id /
# assoc_cluster_main) + its un-merge, over the standard 30-event MCP2025C scan.
#
#   arm "off"  = new binary, knobs OFF.  Gate: every archive must be identical
#                (member-content hash) to the pre-change products already on disk
#                (work-*-mainreal for the Q/L stage, work-*-d49son for nusel).
#   arm "on"   = -save-assoc (Q/L) + -unmerge-assoc (nusel).
#
# Both arms run under `setarch x86_64 -R`: this chain is ASLR-non-deterministic
# at +/-7 STM tags out of ~44 (doc 49 4a), which is larger than the effect.
#
# Imaging is SYMLINKED from work-*-d49son -- never regenerated (M11/M13).
#
# Usage:  ./run_d52_campaign.sh [off|on|both] [tagbase]   (default both d52)
#   tagbase names the work dirs work-<suff>-<tagbase><arm>.  Use a fresh
#   tagbase for each campaign round (M13): "d52" was the pre-realign (VOID,
#   doc 52 §9/§10) round; "d52r" is the post-realign_perblob redo.
set -u
cd "$(dirname "$0")"
TAGBASE="${2:-d52}"

QLF="-save-pctree -save-rcid -lm"
NUF="-chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair-real -fvx 2.5 -fvy 3 -stm-fit -mip 56000"
S=$PWD/input_files_reco1/staged-mcp2025c-1000evt
TEN=$PWD/input_files_reco1/extracted-mcp2025c-10evt
: "${SBND_MAX_JOBS:=5}"
export SBND_MAX_JOBS

ARMS="${1:-both}"
[ "$ARMS" = both ] && ARMS="off on"

for arm in $ARMS; do
    tag="$TAGBASE$arm"
    QLX=""; NUX=""
    if [ "$arm" = on ]; then QLX="-save-assoc"; NUX="-unmerge-assoc"; fi

    for suff in mcp10 mcp1000 mcp1000b; do
        mkdir -p "work-$suff-$tag"
        for d in work-$suff-d49son/evt*; do
            ln -sfn "$(readlink -f "$d")" "work-$suff-$tag/$(basename "$d")"
        done
    done

    echo "=== arm $arm : mcp10 (10 events, batched) ==="
    SBND_INPUT_DIR=$TEN SBND_WORK_ROOT=$PWD/work-mcp10-$tag \
        setarch x86_64 -R ./run_ql_evt.sh data all $QLF $QLX
    SBND_INPUT_DIR=$TEN SBND_WORK_ROOT=$PWD/work-mcp10-$tag \
        setarch x86_64 -R ./run_nusel_evt.sh data all $NUF $NUX

    echo "=== arm $arm : mcp1000 (staged e10..e19) ==="
    for e in $(seq 10 19); do
        SBND_INPUT_DIR=$S/e$e SBND_WORK_ROOT=$PWD/work-mcp1000-$tag \
            setarch x86_64 -R ./run_ql_evt.sh data 1 $QLF $QLX
        SBND_INPUT_DIR=$S/e$e SBND_WORK_ROOT=$PWD/work-mcp1000-$tag \
            setarch x86_64 -R ./run_nusel_evt.sh data 1 $NUF $NUX
    done

    echo "=== arm $arm : mcp1000b (staged e20..e29) ==="
    for e in $(seq 20 29); do
        SBND_INPUT_DIR=$S/e$e SBND_WORK_ROOT=$PWD/work-mcp1000b-$tag \
            setarch x86_64 -R ./run_ql_evt.sh data 1 $QLF $QLX
        SBND_INPUT_DIR=$S/e$e SBND_WORK_ROOT=$PWD/work-mcp1000b-$tag \
            setarch x86_64 -R ./run_nusel_evt.sh data 1 $NUF $NUX
    done
done
echo "=== campaign done ==="
