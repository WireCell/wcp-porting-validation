#!/bin/bash
# doc pr/142 -- the probe pair, for the labeled-subset completeness metric.
#
# em117_score.py's q_miss / q_extra needs per-shower membership sidecars, which
# prep_pr117.py parses out of `SHOWER_CONTENT` probe lines.  Those lines only
# exist when WCT_SHOWER_CONTENT_DEBUG is set, and the two full-sample arms did
# NOT set it (a production arm should not carry a debug probe).  So the metric
# gets its own pair, on the 239-event hand-scan manifest only, same knobs as
# the corresponding full arm.  The probe is byte-neutral (stderr/stdout only).
set -u
cd "$(dirname "$0")/.." || exit 1
SXD=$PWD
export LD_LIBRARY_PATH=${PR142_LIBSNAP:-/home/xqian/tmp/pr142-libsnap}:${LD_LIBRARY_PATH:-}
export WCT_SHOWER_CONTENT_DEBUG=1
export PR_JOBS=${PR_JOBS:-16}
export PR_EXTRA_STAGES=pr_display
TLA=$SXD/docs/pr/pr142-restore-empre.tla
LOGD=${PR142_LOGDIR:-/home/xqian/tmp/pr142}
mkdir -p "$LOGD"
for s in nuecc48 ncpi0 mcp1k mcp2k; do
    EVTS=$(tr '\n' ' ' < /home/xqian/tmp/pr139-manifest-$s.lst)
    for arm in empre prod; do
        out=work-pr142probe-$arm-$s
        [ -d "$out" ] && { echo "SKIP $out"; continue; }
        echo "=== $(date +%H:%M:%S) $out ($(echo $EVTS | wc -w) events) ==="
        if [ "$arm" = empre ]; then
            PR_EXTRA_TLA=$TLA ./run_pr_chain_batch.sh work-$s-grp0825 "$out" data $EVTS \
                > "$LOGD/$out.log" 2>&1
        else
            ./run_pr_chain_batch.sh work-$s-grp0825 "$out" data $EVTS \
                > "$LOGD/$out.log" 2>&1
        fi
        echo "    rc=$?  n=$(ls -d $out/pr_evt* 2>/dev/null | wc -l)"
    done
done
echo "PROBE ARMS DONE $(date +%H:%M:%S)"
