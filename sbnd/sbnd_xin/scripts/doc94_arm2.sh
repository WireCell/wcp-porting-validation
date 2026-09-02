#!/bin/bash
# doc 94 -- run one PR-tail arm with the VERTEX-HADRON guard on (round 2).
# Separate file, not an edit of doc94_arm.sh: that script is still executing
# the mcp2k probe arm, and bash re-reads a running script at a byte offset.
# Pinned to doc94b-libsnap (the probe arm must keep the OLD binary).
#   Usage: doc94_arm.sh <ql_root> <out_root> <data|sim> <cos_y> [extra TLA line ...]
# cos_y = 1.01 is the PROBE point: above the feature's range, so the DEBUG
# line prints on every evaluated bundle and no verdict can move.
# Binary pinned to the doc-94 snapshot (a peer session shares local/lib).
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
QL=$1; OUT=$2; REALITY=$3; COSY=$4; shift 4
[ -e "$BASE/$OUT" ] && { echo "SKIP: $OUT exists (M13)" >&2; exit 1; }
export LD_LIBRARY_PATH=$HOME/tmp/doc94b-libsnap:${LD_LIBRARY_PATH:-}
export PR_JOBS=${PR_JOBS:-6}
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES:-pr_display}
TLA=/home/xqian/tmp/doc94/tla-$(basename "$OUT").txt
mkdir -p /home/xqian/tmp/doc94
{ echo "stm_vertex_hadron_guard=true"; for x in "$@"; do echo "$x"; done; } > "$TLA"
export PR_EXTRA_TLA=$TLA
echo "=== $OUT  cos_y=$COSY  jobs=$PR_JOBS  start $(date -Is)"; cat "$TLA" | sed 's/^/    /'
./run_pr_chain_batch.sh "$QL" "$OUT" "$REALITY" > "/home/xqian/tmp/doc94/$(basename "$OUT").log" 2>&1
rc=$?
n=$(find "$BASE/$OUT" -maxdepth 1 -type d -name 'pr_evt*' | wc -l)
echo "=== $OUT rc=$rc pr_evt=$n end $(date -Is)"
exit $rc
