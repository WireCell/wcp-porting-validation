#!/bin/bash
# doc sbnd_xin/113 sec 5 -- fix-variant imaging arms (imaging only) on a group list; ARMS = tla basenames.
set -u
SX=/home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin; cd $SX
export JOBS=${JOBS:-6} TO=img KEEP_ICLUSTER=1
LIST=${LIST:-docs/113_figs/113_flagged_groups.txt}
while read -r s groups; do
    [ -n "$s" ] || continue
    for arm in ${ARMS:?set ARMS}; do
        echo "== $(date +%T) $s $arm groups $groups"
        scripts/d113_stageA_arm.sh "$s" "cf$arm" "$groups" docs/113_figs/tla/$arm.tla || echo "ARM FAILED $s $arm"
    done
done < $LIST
echo FIXDONE
