#!/bin/bash
# doc 94 round 2 -- vertex_hadron_guard ON over the full 3067-event data
# sample.  Waits for the probe arms to finish first (M5: do not oversubscribe).
set -u
cd -P "$(dirname "$0")/.." || exit 1
# NOTE: an earlier version waited here on `pgrep -f d94probe`.  That
# deadlocked: the driver's own parent shell carried the string 'd94probe' in
# its command line (the heredoc that wrote this file), so the guard matched
# itself forever.  Wait on the driver's DONE marker instead, which is a fact
# about the work rather than about process tables.
while [ -f /home/xqian/tmp/doc94/probe_arms2.out ] && \
      ! grep -q 'ALL DONE' /home/xqian/tmp/doc94/probe_arms2.out; do sleep 30; done
echo "probe arms clear, starting $(date -Is)"
export PR_JOBS=${PR_JOBS:-24}
export PR_EXTRA_STAGES=pr_display
rc_all=0
for s in ncpi0 nuecc48 mcp1k mcp2k; do
    ./scripts/doc94_arm2.sh "work-$s-grp0825" "work-$s-d94hadron" data - || rc_all=1
done
echo "ALL DONE rc_all=$rc_all $(date -Is)"
exit $rc_all
