#!/bin/bash
# doc 109: one arm of the tracking-pr.root output A/B on the sbnd_xin data samples.
#
# Usage: scripts/d109_arms.sh <label> <libsnap> [VAR=value ...]
#   label    arm tag; outputs go to work-<sample>-<label>/ (a FRESH dir, M13)
#   libsnap  pinned copy of local/lib (LD_LIBRARY_PATH), e.g. ~/tmp/d109-libsnap/base
#   VAR=val  extra env for run_pr_chain_batch.sh, e.g. SBND_NO_DL=1 SBND_ROOT_OUTPUT=1
#            PR_CFG_TREE=<cfg dir> pins the jsonnet tree as well
#
# Stage-A input: the doc 102 arms work-<sample>-d102m (read-only).  Samples:
# nuecc48 (48), ncpi0 (19), and the first 200 mcp1k events by id
# (docs/109_logs/events_mcp1k200.txt).  setarch -R for determinism (M4);
# PR_EXTRA_STAGES=pr_display as in production.
set -u
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
LABEL=${1:?usage: d109_arms.sh <label> <libsnap> [VAR=value ...]}
PIN=${2:?usage: d109_arms.sh <label> <libsnap> [VAR=value ...]}
shift 2
cd "$SX" || exit 1
for kv in "$@"; do export "$kv"; done
export LD_LIBRARY_PATH=$PIN:${LD_LIBRARY_PATH:-}
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES-pr_display}
export PR_JOBS=${PR_JOBS:-16}
LOGD=$HOME/tmp/d109
mkdir -p "$LOGD"
EV=$SX/docs/109_logs/events_mcp1k200.txt
echo "=== $LABEL start $(date +%F_%T) pin=$PIN env: $*"
(cd "$PIN" && md5sum libWireCellClus.so libWireCellRoot.so)
for s in nuecc48 ncpi0 mcp1k; do
    ids=()
    [ "$s" = mcp1k ] && mapfile -t ids < "$EV"
    echo "=== $LABEL $s $(date +%T)"
    setarch x86_64 -R ./run_pr_chain_batch.sh "work-$s-d102m" "work-$s-$LABEL" data "${ids[@]}" \
        > "$LOGD/$LABEL-$s.log" 2>&1
    rc=$?
    echo "=== $LABEL $s rc=$rc dirs=$(ls -d work-$s-$LABEL/pr_evt* 2>/dev/null | wc -l)" \
         "not_rc0=$(grep -L 'rc=0' work-$s-$LABEL/pr_evt*/rc.txt 2>/dev/null | wc -l)"
done
(cd "$PIN" && md5sum libWireCellClus.so libWireCellRoot.so)
echo "=== $LABEL done $(date +%F_%T)"
