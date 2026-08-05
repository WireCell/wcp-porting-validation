#!/bin/bash
# doc pr/2 sec 2e(iv) SBND arm: run the 48-event nueCC sample through both arms
# of the detector-extent knobs with ONE binary (concurrent-session-safe, see
# CLAUDE.md / feedback_concurrent_sessions_same_tree) and compare Bee-zip
# member hashes per event.
#   on  = SBND defaults (cosmic_y 183/185/163/133 cm, vertex_z_prior 100 cm)
#   off = uBooNE literals (pr_y_top=117, vertex_z_prior_scale=200)
set -u
SC=$(cd "$(dirname "$0")" && pwd)
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
ROOT=${ROOT:-/home/xqian/tmp/geomab}
JOBS=${JOBS:-6}
mkdir -p "$ROOT"
EVTS=$(ls -d $SX/work-nuecc48-cb0805/ql_evt* | sed 's#.*/ql_evt##' | sort -n)
run_one() {
    local evt=$1 arm=$2
    PROUT="$ROOT/$evt/$arm" ARM=$arm "$SC/run_pr_geom_arm.sh" "$evt" > /dev/null 2>&1
}
n=0
for evt in $EVTS; do
    for arm in off on; do
        run_one "$evt" "$arm" &
        n=$((n+1))
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
    done
done
wait
echo "ALLDONE n=$n"
