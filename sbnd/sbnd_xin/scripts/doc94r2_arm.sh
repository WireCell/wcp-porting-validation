#!/bin/bash
# doc 94 ROUND 2 -- run one PR-tail arm with the ENTRY-RISE guard on.
#   Usage: doc94r2_arm.sh <ql_root> <out_root> <data|sim> <min_cm> [extra TLA line ...]
# min_cm = 1000 is the PROBE point: above the feature's range, so the DEBUG
# line prints shoulder / shoulder_nofirst / excess on every STM-evaluated
# bundle and NO verdict can move -- which also makes the arm a byte-identity
# check against the round-2 baseline work-*-d94hadron.
#
# Baseline note: vertex_hadron_guard is SBND PRODUCTION as of 2026-09-02
# (ref/prod-2026-09-02), so it is ON by default here and the comparison arm is
# work-*-d94hadron, NOT work-*-prod0901b.  Diffing against prod0901b would
# re-attribute round 1's three recoveries to this guard.
#
# Separate file, not an edit of doc94_arm2.sh: bash re-reads a running script
# at a byte offset, and those scripts are the record of round 1.
# Binary pinned to doc94c-libsnap (a peer session shares local/lib).
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
QL=$1; OUT=$2; REALITY=$3; MINCM=$4; shift 4
[ -e "$BASE/$OUT" ] && { echo "SKIP: $OUT exists (M13)" >&2; exit 1; }
export LD_LIBRARY_PATH=$HOME/tmp/doc94c-libsnap:${LD_LIBRARY_PATH:-}
export PR_JOBS=${PR_JOBS:-6}
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES:-pr_display}
TLA=/home/xqian/tmp/doc94r2/tla-$(basename "$OUT").txt
mkdir -p /home/xqian/tmp/doc94r2
{ echo "stm_entry_rise_guard=true"; echo "stm_entry_min_cm=$MINCM"; for x in "$@"; do echo "$x"; done; } > "$TLA"
export PR_EXTRA_TLA=$TLA
echo "=== $OUT  min_cm=$MINCM  jobs=$PR_JOBS  start $(date -Is)"; sed 's/^/    /' "$TLA"
./run_pr_chain_batch.sh "$QL" "$OUT" "$REALITY" > "/home/xqian/tmp/doc94r2/$(basename "$OUT").log" 2>&1
rc=$?
n=$(find "$BASE/$OUT" -maxdepth 1 -type d -name 'pr_evt*' | wc -l)
echo "=== $OUT rc=$rc pr_evt=$n end $(date -Is)"
exit $rc
