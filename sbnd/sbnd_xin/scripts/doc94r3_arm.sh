#!/bin/bash
# doc 94 ROUND 3 -- entry_rise_guard WITH the owner's kink requirement.
#   Usage: doc94r3_arm.sh <ql_root> <out_root> <data|sim> <min_cm> [extra TLA ...]
# min_cm=1000 is the PROBE point (above the feature's range => inert, and the
# DEBUG line now also prints the kink angle and where it is).
# Separate file, not an edit of doc94r2_arm.sh: that script is the record of
# round 2 and bash re-reads a running script at a byte offset.
# Binary pinned to doc94r3-libsnap (round 2 used doc94c-libsnap).
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
QL=$1; OUT=$2; REALITY=$3; MINCM=$4; shift 4
[ -e "$BASE/$OUT" ] && { echo "SKIP: $OUT exists (M13)" >&2; exit 1; }
export LD_LIBRARY_PATH=$HOME/tmp/doc94r3-libsnap:${LD_LIBRARY_PATH:-}
export PR_JOBS=${PR_JOBS:-6}
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES:-pr_display}
TLA=/home/xqian/tmp/doc94r2/tla3-$(basename "$OUT").txt
{ echo "stm_entry_rise_guard=true"; echo "stm_entry_min_cm=$MINCM"; for x in "$@"; do echo "$x"; done; } > "$TLA"
export PR_EXTRA_TLA=$TLA
echo "=== $OUT  min_cm=$MINCM  jobs=$PR_JOBS  start $(date -Is)"; sed 's/^/    /' "$TLA"
./run_pr_chain_batch.sh "$QL" "$OUT" "$REALITY" > "/home/xqian/tmp/doc94r2/$(basename "$OUT").log" 2>&1
rc=$?
n=$(find "$BASE/$OUT" -maxdepth 1 -type d -name 'pr_evt*' | wc -l)
echo "=== $OUT rc=$rc pr_evt=$n end $(date -Is)"
exit $rc
