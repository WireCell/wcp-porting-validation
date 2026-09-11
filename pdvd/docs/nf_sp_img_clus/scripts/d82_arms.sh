#!/usr/bin/env bash
# doc pdvd/82 -- the arms for doc 78 action item 2 (the peak-then-drop stop).
#
# Fork by duplication (CLAUDE.md M10) of d80_arms.sh.  The ONLY differences are
# the arm names, the TLAs and the pin.
#
# The pin is a FULL local/lib snapshot, not a Clus-only one: doc pdvd/80's wave 1
# died with rc=139 at the ROOT-writing stage because a Clus-only pin was mixed
# with a rebuilt libWireCellRoot (feedback_shared_tree_binary_pin).
#
# WAVE 1 = the OFF gate + the three feature arms + the existing-knob sweep.
#
# Usage:  ./d82_arms.sh            (runs every arm, sequential waves, JOBS 6)
#         WAVE=1 ./d82_arms.sh
set -uo pipefail
X=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts
PIN=${PIN:-/home/xqian/tmp/p82/libpin_p82}
JOBS=${JOBS:-5}
LOGD=${LOGD:-/home/xqian/tmp/p82}
# WAVE 1 (p82v*) carried the d53 SURVEY bag and is VOID as a gate: the OFF
# baselines on disk (p80boff / p80bhoff / p79vprod) are BARE PRODUCTION -- no
# survey -- so the comparison read the survey's own effect (it fits extra
# clusters, so T_rec_charge and T_proj_data move, and doc 53 measured a 20-25%
# mover rate on the muon's own profile branches) as if it were this round's.
# Its dirs are kept as a record (CLAUDE.md M13).  WAVE 2 (p82b*) is bare
# production plus this round's keys and nothing else -- like for like, and the
# path production actually runs.

run () {  # run <arm> <det> <src> <extra-keys-or-empty>
    local arm=$1 det=$2 src=$3 extra=$4 tla=""
    [ -n "$extra" ] && tla="-S stm_michel_extra={$extra}" || tla=""
    echo "[$(date +%H:%M:%S)] $arm ($det) $extra"
    ARM=$arm DET=$det SRC=$src JOBS=$JOBS PIN=$PIN PR_TLA="$tla" \
        LOGD=$LOGD/arm_$arm "$X/d53_run_arms.sh" > "$LOGD/arm_${arm}.log" 2>&1
    echo "[$(date +%H:%M:%S)] $arm rc=$?"
}

# The six arms run CONCURRENTLY, each with its own JOBS: 6 x 5 = 30 wire-cell
# jobs, inside the owner's 32-CPU licence (feedback_parallelism_32_cpus).  Each
# arm writes its own work dirs and its own log dir, so they cannot collide; the
# only shared resource is the pin, which is read-only.
WAVE=${WAVE:-2}
if [ "$WAVE" = 2 ]; then
    run p82boff  pdvd d16vnu "" &
    run p82bhoff pdhd d16hnu "" &
    run p82bcs   pdvd d16vnu "michel_collinear_split:true" &
    run p82btp   pdvd d16vnu "michel_collinear_split:true,stop_tail_peak_frac:0.5,stop_tail_peak_kink_min_deg:25.0" &
    run p82btp0  pdvd d16vnu "stop_tail_peak_frac:0.5,stop_tail_peak_kink_min_deg:25.0" &
    run p82bk10  pdvd d16vnu "split_kink_min_deg:10.0" &
    wait
fi
# WAVE 3 = the flip confirmation: the FLIPPED wct-pr-perevt.jsonnet, no TLA at
# all.  It must be bit-identical to p82bk10, which got the same value through
# stm_michel_extra (CLAUDE.md sec 4: the flipped file is what production runs).
if [ "$WAVE" = 3 ]; then
    run p82vprod pdvd d16vnu ""
fi
touch "$LOGD/DONE_wave$WAVE"
echo "[$(date +%H:%M:%S)] wave $WAVE done"
