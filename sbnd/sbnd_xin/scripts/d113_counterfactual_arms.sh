#!/bin/bash
# doc sbnd_xin/113 sec 4 -- imaging-only counterfactual arms on the groups that hold a flagged imaging-class
# component: cf0 (no knob = today's binary, the imaging baseline that also separates clustering deletions),
# pdoff / isdoff (deghoster dryruns), dt0 (data-driven zero-summary fallback), th0 (2.5 sigma).  Sequential
# arms, JOBS groups in flight each.  Output: work-<s>-d113cf<arm>/g<K>/icluster-apa*-active.npz
set -u
SX=/home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
cd $SX
export JOBS=${JOBS:-6} TO=img KEEP_ICLUSTER=1
while read -r s groups; do
    [ -n "$s" ] || continue
    for arm in ${ARMS:-cf0 pdoff isdoff dt0 th0}; do
        tla=""; [ "$arm" != cf0 ] && tla=docs/113_figs/tla/$arm.tla
        echo "== $(date +%T) $s $arm groups $groups"
        scripts/d113_stageA_arm.sh "$s" "cf$arm" "$groups" $tla || echo "ARM FAILED $s $arm"
    done
done < docs/113_figs/113_flagged_groups.txt
echo CFDONE
