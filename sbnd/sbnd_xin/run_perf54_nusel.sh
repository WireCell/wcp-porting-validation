#!/usr/bin/env bash
# Doc 54 TGM/STM perf campaign: nusel-only rerun of the 30-event MCP2025C scan
# with per-event wall/RSS metering (abtest/timecmd.py), SEQUENTIAL for timing
# fidelity, under `setarch x86_64 -R` (this chain is ASLR-non-deterministic at
# +/-7 STM tags, doc 49 4a -- every A/B here must disable ASLR).
#
# Imaging (evt*) and Q/L products (ql_evt*) are SYMLINKED from work-*-d55ton --
# never regenerated (M11/M13); only nusel_evt* is made, in fresh
# work-<suff>-<tagbase><arm> roots.  Tagger flags = the d55ton arm's exactly
# (doc 53): the d52 NUF set + -unmerge-assoc.
#
# Usage: ./run_perf54_nusel.sh <arm> [tagbase] [extra run_nusel_evt.sh flags...]
#   arm      names the work roots work-<suff>-<tagbase><arm>
#   tagbase  default p54; use a fresh tagbase per campaign round (M13)
#   extra    appended to the fixed flag set, e.g. -no-bwonly for the doc-56
#            knob-off gate arm.  Also selects events: pass nothing to run all 30.
# Env: PERF54_EVENTS="i3 e13 e27" restricts the run to those labels (smoke tests).
set -u
cd "$(dirname "$0")"
ARM="${1:?arm (base|opt)}"
TAGBASE="${2:-p54}"
shift $(( $# > 2 ? 2 : $# ))
EXTRA="$*"
tag="$TAGBASE$ARM"

# The production flag set: everything else is now a config default (doc 68).
NUF="-stm-fit $EXTRA"
S=$PWD/input_files_reco1/staged-mcp2025c-1000evt
TEN=$PWD/input_files_reco1/extracted-mcp2025c-10evt
TC=$(cd ../../abtest && pwd)/timecmd.py

for suff in mcp10 mcp1000 mcp1000b; do
    mkdir -p "work-$suff-$tag"
    for d in work-$suff-d55ton/evt* work-$suff-d55ton/ql_evt*; do
        ln -sfn "$(readlink -f "$d")" "work-$suff-$tag/$(basename "$d")"
    done
done

run1() {   # <suff> <input_dir> <idx> <label>
    local suff=$1 inp=$2 idx=$3 lab=$4 rc
    local root="$PWD/work-$suff-$tag"
    SBND_INPUT_DIR=$inp SBND_WORK_ROOT=$root \
        setarch x86_64 -R python3 "$TC" "$root/.time_${lab}.meta" \
        ./run_nusel_evt.sh data $NUF "$idx" > "$root/.perf_${lab}.log" 2>&1
    rc=$?
    echo "[$tag $suff $lab] rc=$rc $(tr '\n' ' ' < "$root/.time_${lab}.meta" 2>/dev/null)"
    return $rc
}

want() {   # <label> -- honor PERF54_EVENTS if set
    [ -z "${PERF54_EVENTS:-}" ] && return 0
    case " $PERF54_EVENTS " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}

fail=0
for i in $(seq 1 10);  do want "i$i" && { run1 mcp10    "$TEN"   "$i" "i$i" || fail=1; }; done
for e in $(seq 10 19); do want "e$e" && { run1 mcp1000  "$S/e$e" 1   "e$e" || fail=1; }; done
for e in $(seq 20 29); do want "e$e" && { run1 mcp1000b "$S/e$e" 1   "e$e" || fail=1; }; done
echo "=== $tag done fail=$fail ==="
exit $fail
