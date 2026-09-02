#!/bin/bash
# doc 94 ROUND 2 -- small INSTRUMENTED arms over the hand-scan set: the two
# data bundles entry_rise_guard releases, the one it declines at max_cm, and
# three below-cut controls.  save_stm_fit=true so the Bee carries the
# stm_fit trajectory layer and the entry profile can be plotted.
#
# Two arms per sample, OFF and ON, so the owner scans an A/B pair.  OFF is the
# production point (ref/prod-2026-09-02: vertex_hadron_guard on, entry guard
# absent); ON adds stm_entry_rise_guard=true at the operating point.
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
export LD_LIBRARY_PATH=$HOME/tmp/doc94c-libsnap:${LD_LIBRARY_PATH:-}
export PR_JOBS=${PR_JOBS:-6}
export PR_EXTRA_STAGES=pr_display,stm_magnify
LOGD=/home/xqian/tmp/doc94r2; mkdir -p "$LOGD"

# sample : events
run_one() {   # $1=ql_root $2=out $3=tla-file $4...=events
    local ql=$1 out=$2 tla=$3; shift 3
    if [ -e "$BASE/$out" ]; then echo "SKIP $out (M13)"; return 0; fi
    export PR_EXTRA_TLA=$tla
    echo "=== $out  events: $*  start $(date -Is)"
    ./run_pr_chain_batch.sh "$ql" "$out" data "$@" > "$LOGD/$out.log" 2>&1
    local rc=$?
    echo "=== $out rc=$rc dirs=$(find "$BASE/$out" -maxdepth 1 -type d -name 'pr_evt*' | wc -l) end $(date -Is)"
    return $rc
}
printf 'save_stm_fit=true\n' > "$LOGD/tla-scan-off.txt"
printf 'save_stm_fit=true\nstm_entry_rise_guard=true\nstm_entry_min_cm=5.0\n' > "$LOGD/tla-scan-on.txt"

rc_all=0
run_one work-mcp1k-grp0825 work-mcp1k-r2scanoff "$LOGD/tla-scan-off.txt" 350099 290316 282033 || rc_all=1
run_one work-mcp1k-grp0825 work-mcp1k-r2scanon  "$LOGD/tla-scan-on.txt"  350099 290316 282033 || rc_all=1
run_one work-mcp2k-grp0825 work-mcp2k-r2scanoff "$LOGD/tla-scan-off.txt" 164466 95500 56257 || rc_all=1
run_one work-mcp2k-grp0825 work-mcp2k-r2scanon  "$LOGD/tla-scan-on.txt"  164466 95500 56257 || rc_all=1
echo "ALL DONE rc_all=$rc_all $(date -Is)"
exit $rc_all
