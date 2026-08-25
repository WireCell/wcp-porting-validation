#!/bin/bash
# Smoke test for the doc-82-r2 rc=0 fix: exercise the exact decision block
# lifted from run_pr_chain_batch.sh against (a) a complete group and (b) the
# same group with one member's product removed.  Scratch only -- the real arms
# are reached through read-only symlinks and are never written to.
set -u
SX=$(cd "$(dirname "$0")/../.." && pwd -P)
REAL=$SX/work-mcp1k-prod0825
T=${1:-/home/xqian/tmp/d82r2/smoke}
rm -rf "$T"; mkdir -p "$T"

# Lift the block out of the runner so the test cannot drift from the source.
sed -n '/doc 82 round 2\.  This used to be an unconditional/,/^        fi$/p' \
    "$SX/run_pr_chain_batch.sh" > "$T/block.sh"
grep -q 'pctree-pr-evt' "$T/block.sh" || { echo "FAIL: could not lift the block"; exit 1; }

mapfile -t EVTS < <(ls -d "$REAL"/pr_evt* 2>/dev/null | head -4 | sed 's#.*/pr_evt##')
[ ${#EVTS[@]} -eq 4 ] || { echo "FAIL: need 4 events, got ${#EVTS[@]}"; exit 1; }

run_case() {
    local name=$1 drop=$2 GIDX=0
    local OUTROOT="$T/$name"
    mkdir -p "$OUTROOT"
    for evt in "${EVTS[@]}"; do
        mkdir -p "$OUTROOT/pr_evt$evt"
        [ "$evt" = "$drop" ] && continue
        ln -sf "$REAL/pr_evt$evt/pctree-pr-evt$evt.tar.gz" \
               "$OUTROOT/pr_evt$evt/pctree-pr-evt$evt.tar.gz"
    done
    local _NOK=0; local -a _MISSING=()
    for evt in "${EVTS[@]}"; do
        local PRDIR="$OUTROOT/pr_evt$evt"
        . "$T/block.sh"
    done
    echo "[$name] _NOK=$_NOK/${#EVTS[@]} missing='${_MISSING[*]:-}' rc.txt: $(cat "$OUTROOT"/pr_evt*/rc.txt | tr '\n' ' ')"
}
run_case complete ""
run_case onemissing "${EVTS[1]}"
