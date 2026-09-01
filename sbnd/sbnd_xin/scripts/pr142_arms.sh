#!/bin/bash
# doc pr/142 -- the two full-sample arms of the campaign before/after.
#
#   empre0901  HEAD binary + HEAD cfg with the 39 EM/pi0 campaign knobs restored
#              to their pre-campaign (8d93260d, 2026-08-25) values via
#              PR_EXTRA_TLA.  doc 84's MCS / long-muon knobs stay ON -- the
#              owner scoped this round to the EM/pi0 rounds (pr/117-141).
#   prod0901   HEAD binary + HEAD cfg as shipped.  This IS the new production.
#
# Both arms: same binary (pinned via LD_LIBRARY_PATH to a snapshot, because a
# peer wcbuild has swapped local/lib mid-campaign before), same PR_JOBS, PER-EVENT
# mode (PR_GROUP_SIZE unset) so .time.meta lands per event -- group mode writes
# one per 16-event group, which is why every earlier product table has blank
# wall_s / maxrss_kb columns.  Samples are INTERLEAVED (empre then prod, per
# sample) so each A/B pair sees the same box conditions.
set -u
cd "$(dirname "$0")/.." || exit 1
SXD=$PWD
SNAP=${PR142_LIBSNAP:-/home/xqian/tmp/pr142-libsnap}
TLA=$SXD/docs/pr/pr142-restore-empre.tla
LOGD=${PR142_LOGDIR:-/home/xqian/tmp/pr142}
mkdir -p "$LOGD"
[ -r "$TLA" ] || { echo "ERROR: missing $TLA" >&2; exit 1; }
[ -d "$SNAP" ] || { echo "ERROR: missing lib snapshot $SNAP" >&2; exit 1; }
export LD_LIBRARY_PATH=$SNAP:${LD_LIBRARY_PATH:-}
export PR_JOBS=${PR_JOBS:-16}
export PR_EXTRA_STAGES=pr_display
for s in nuecc48 ncpi0 mcp1k mcp2k; do
    for arm in empre0901 prod0901; do
        out=work-$s-$arm
        [ -d "$out" ] && { echo "SKIP $out (exists)"; continue; }
        echo "=== $(date +%H:%M:%S) $out  jobs=$PR_JOBS ==="
        if [ "$arm" = empre0901 ]; then
            PR_EXTRA_TLA=$TLA ./run_pr_chain_batch.sh work-$s-grp0825 "$out" data \
                > "$LOGD/$out.log" 2>&1
        else
            ./run_pr_chain_batch.sh work-$s-grp0825 "$out" data \
                > "$LOGD/$out.log" 2>&1
        fi
        echo "    rc=$?  n=$(ls -d $out/pr_evt* 2>/dev/null | wc -l)  loadavg=$(cut -d' ' -f1 /proc/loadavg)"
    done
done
echo "ARMS DONE $(date +%H:%M:%S)"
