#!/bin/bash
# sbnd_xin/docs/109 rev 3: one arm of the nu_dedup_flash_group census.
#
# The doc 109 manifest (nuecc48 + ncpi0 + first 200 mcp1k) holds almost no
# multi-candidate events, so it cannot measure this knob.  This arm runs the 25
# events of the 3067-event sample that have 2+ T_tagger rows -- the only events
# the knob can possibly touch -- so both arms are minutes, not hours.
#
# Usage: scripts/d109r3_dedup_arm.sh <label> <libsnap> [VAR=value ...]
set -u
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
LABEL=${1:?usage: d109r3_dedup_arm.sh <label> <libsnap> [VAR=value ...]}
PIN=${2:?usage: d109r3_dedup_arm.sh <label> <libsnap> [VAR=value ...]}
shift 2
cd "$SX" || exit 1
for kv in "$@"; do export "$kv"; done
export LD_LIBRARY_PATH=$PIN:${LD_LIBRARY_PATH:-}
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES-pr_display}
export PR_JOBS=${PR_JOBS:-8}
LOGD=$HOME/tmp/d109r3
mkdir -p "$LOGD"

# The 25 multi-candidate events of work-<s>-d102mpr (scripts/d109r3_dedup_census.py
# header).  Listed literally so the arm is reproducible after those dirs change.
NUECC48=""
NCPI0="18625"
MCP1K="174422 280466 286681 317939 400636 409634 487303 62583"
MCP2K="167612 174661 175094 179054 179369 287638 294174 351564 407798 409624 410698 68748 72275 73038 78032 90751"

echo "=== $LABEL start $(date +%F_%T) pin=$PIN env: $*"
(cd "$PIN" && md5sum libWireCellClus.so libWireCellRoot.so)
for s in ncpi0 mcp1k mcp2k; do
    case $s in
        ncpi0) ids="$NCPI0" ;;
        mcp1k) ids="$MCP1K" ;;
        mcp2k) ids="$MCP2K" ;;
    esac
    [ -n "$ids" ] || continue
    echo "=== $LABEL $s $(date +%T)"
    setarch x86_64 -R ./run_pr_chain_batch.sh "work-$s-d102m" "work-$s-$LABEL" data $ids \
        > "$LOGD/$LABEL-$s.log" 2>&1
    rc=$?
    echo "=== $LABEL $s rc=$rc dirs=$(ls -d work-$s-$LABEL/pr_evt* 2>/dev/null | wc -l)" \
         "not_rc0=$(grep -L 'rc=0' work-$s-$LABEL/pr_evt*/rc.txt 2>/dev/null | wc -l)"
done
(cd "$PIN" && md5sum libWireCellClus.so libWireCellRoot.so)
echo "=== $LABEL done $(date +%F_%T)"
