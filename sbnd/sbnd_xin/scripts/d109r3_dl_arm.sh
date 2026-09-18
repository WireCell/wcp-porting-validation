#!/bin/bash
# sbnd_xin/docs/109 rev 3 sec 8.8.7: the DL-vertex arm.
#
# The byte-gate arms run the GEOMETRIC vertex (SBND_NO_DL=1) because the DL
# vertex is not bit-stable (CLAUDE.md M4).  But doc 109 sec 4.4 measured that
# the geometric vertex produces ZERO vertex_moved_cluster rows on nuecc48 --
# the moved rows are a DL-vertex phenomenon.  Case 1 of the colleague's first
# look IS the moved row, so it cannot be graded on the byte-gate arms at all.
#
# This arm therefore runs nuecc48 with the production (DL) vertex and the
# Group A knob on, and is graded on CONTENT (d109_root_checks.py), not bytes.
# Its "before" is doc 109's own production smoke, work-nuecc48-d109prod, which
# ran the same vertex mode at the pre-rev-3 code.
#
# Usage: scripts/d109r3_dl_arm.sh [label] [libsnap]
set -u
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
LABEL=${1:-d109r3dlon}
PIN=${2:-$HOME/tmp/d109r3-libsnap/new2}
cd "$SX" || exit 1
export LD_LIBRARY_PATH=$PIN:${LD_LIBRARY_PATH:-}
export SBND_ROOT_OUTPUT=1 SBND_ROOT_POINT_IDS=1 SBND_NU_DEDUP_FLASH_GROUP=0
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES-pr_display}
export PR_JOBS=${PR_JOBS:-8}
export PR_CFG_TREE=${PR_CFG_TREE:-$HOME/tmp/d109r3-cfg/new/cfg}
LOGD=$HOME/tmp/d109r3; mkdir -p "$LOGD"
echo "=== $LABEL nuecc48 start $(date +%F_%T) pin=$PIN (DL vertex ON)"
(cd "$PIN" && md5sum libWireCellClus.so libWireCellRoot.so)
setarch x86_64 -R ./run_pr_chain_batch.sh work-nuecc48-d102m "work-nuecc48-$LABEL" data \
    > "$LOGD/$LABEL-nuecc48.log" 2>&1
rc=$?
# A silent fallback to the geometric vertex is THE failure mode here, so it is
# counted per event rather than assumed (same rule as doc 109 sec 4).
echo "=== $LABEL nuecc48 rc=$rc dirs=$(ls -d work-nuecc48-$LABEL/pr_evt* 2>/dev/null | wc -l)" \
     "not_rc0=$(grep -L 'rc=0' work-nuecc48-$LABEL/pr_evt*/rc.txt 2>/dev/null | wc -l)" \
     "dlfail=$(grep -l 'DL vertex failed' work-nuecc48-$LABEL/pr_evt*/wct_pr_evt*.log 2>/dev/null | wc -l)"
