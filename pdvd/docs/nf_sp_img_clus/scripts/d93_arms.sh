#!/usr/bin/env bash
# doc pdvd/93 -- the confirmation arm for the flip of doc 92's wide Bragg-peak read.
#
# ONE arm, p93vprod: the FLIPPED pdvd/wct-pr-perevt.jsonnet with NO TLA at all.  Its whole job is to
# prove that routing the three keys through the file gives exactly what routing them through the TLA
# gave, so the +4/-1 trade measured on p92v13 is the trade production will now run.  It therefore
# MUST run on the same binary p92v13 ran on (libpin_p92, md5-checked before and after in d93_gates.sh)
# -- otherwise a peer's wcbuild, not the config route, could explain any difference.
#
# Expected result (d93_gates.sh sec 1): bit-identical to p92v13 on every branch, every point row,
# every tree, every zip and every calib dump.  Against pre-flip production p90vprod (sec 2) it must
# show exactly 5 is_stm flips: +4 stoppers and 1 THRU.
#
# Fork by duplication (CLAUDE.md M10) of d92_arms.sh, reduced to the single no-TLA arm.
set -uo pipefail
X=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts
PIN=${PIN:-/home/xqian/tmp/p92/libpin_p92}
JOBS=${JOBS:-5}
LOGD=${LOGD:-/home/xqian/tmp/p93}
ARM=${ARM:-p93vprod}

if [ ! -s "$PIN/libWireCellClus.so" ]; then
    echo "REFUSING: no pin at $PIN -- a peer rebuild would void the comparison" >&2
    exit 2
fi
if [ -n "$(ls -d /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work/*_$ARM 2>/dev/null)" ]; then
    echo "REFUSING: pdvd/work/*_$ARM already exists (CLAUDE.md M13: a new run gets a new tag)" >&2
    exit 2
fi

echo "[$(date +%H:%M:%S)] $ARM (pdvd) <no TLA: the flipped file is the config>"
ARM=$ARM DET=pdvd SRC=d16vnu JOBS=$JOBS PIN=$PIN PR_TLA="" \
    LOGD=$LOGD/arm_$ARM "$X/d53_run_arms.sh" > "$LOGD/arm_${ARM}.log" 2>&1
echo "[$(date +%H:%M:%S)] $ARM rc=$?"
touch "$LOGD/DONE_arm"
