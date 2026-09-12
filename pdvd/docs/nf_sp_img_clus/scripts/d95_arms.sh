#!/usr/bin/env bash
# doc pdvd/95 -- the arms for the region-based Michel charge (doc 78 action item 9).
#
# Four arms in three waves.  The point of the split is that the OFF wave must PASS before the ON
# wave is worth reading: if the knob-off path is not byte-identical to today's production, the ON
# arm's numbers are measuring my bug, not the owner's definition.
#
#   WAVE=off      p95voff (pdvd) + p95hoff (pdhd)  -- no q2d key at all.
#                 Gate: p95voff == p93vprod, p95hoff == p85hoff, on every branch, point row,
#                 zip member, calib json and all 8 trees.
#   WAVE=on       p95vq2d (pdvd) -- the measurement arm.
#                 michel_q2d + cells + region 40 cm + the body control at 35 cm.
#                 R_max 40 is DIAGNOSTIC, not the production radius: the owner's constraint is
#                 that the Michel sits near the stop, so the headline R comes from the offline
#                 sweep at the TIGHT end (d95_region.py reads d_stop_cm and re-cuts at any R).
#                 One arm answers every radius -- the docs 91/92 twin rule, no arm per R.
#   WAVE=confirm  p95vprod -- the FLIPPED file with NO TLA, to prove the config route gives
#                 exactly what the TLA route measured.  Must run on the SAME pin.
#
# The control at 35 cm is the same body stretch docs 86/90/94 used (rr 30-40 cm), where no Michel
# can be.  It is what prices the muon subtraction at a real Bragg-peak candidate; a control that
# reads far from 0 means the estimator is biased and the definition is not ready to flip.
#
# Fork by duplication (CLAUDE.md M10) of d93_arms.sh / d81_arms.sh.  Nothing here writes into an
# existing work/ tag (M13): every arm refuses if its own dirs already exist.
set -uo pipefail
X=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts
WORK=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
PIN=${PIN:-/home/xqian/tmp/p95/libpin_p95}
JOBS=${JOBS:-5}
LOGD=${LOGD:-/home/xqian/tmp/p95}
WAVE=${WAVE:-off}

# the ON bag.  Every key is written explicitly, including the ones a later flip may leave at a
# C++ default, so that "the flipped file runs what this arm ran" stays a 0-line claim in
# d95_proofs.sh proof A (doc pdvd/93's lesson: match the file to the TLA that was MEASURED).
ON_TLA=${ON_TLA:-"michel_q2d:true,michel_q2d_cells:true,michel_q2d_region_cm:40.0,michel_q2d_region_ctl_cm:35.0"}

if [ ! -s "$PIN/libWireCellClus.so" ]; then
    echo "REFUSING: no pin at $PIN -- a peer's wcbuild would void every comparison" >&2
    exit 2
fi

# SUF appends to every arm name so a re-run gets FRESH tags and never writes into an existing
# one (CLAUDE.md M13).  Empty for the first wave, "b" for the second, and so on.
SUF=${SUF:-}

run_arm() {   # arm det tla
    local arm=$1$SUF det=$2 tla=$3
    # the source arm whose staged event dirs are reused differs per detector -- PDVD events are
    # 039xxx from d16vnu, PDHD 028xxx from d16hnu.  Passing the PDVD source for a PDHD arm stages
    # 0 events and the run then exits 0 having done nothing: a dead net that LOOKS like a pass
    # (see feedback_runner_rc1_dead_check -- verify by event count, never by the exit code).
    local src=d16vnu
    [ "$det" = pdhd ] && src=d16hnu
    if [ -n "$(ls -d $WORK/$det/work/*_$arm 2>/dev/null)" ]; then
        echo "REFUSING: $det/work/*_$arm already exists (CLAUDE.md M13: a new run gets a new tag)" >&2
        return 2
    fi
    echo "[$(date +%H:%M:%S)] $arm ($det, src=$src) TLA=<${tla:-none}>"
    local pr_tla=""
    [ -n "$tla" ] && pr_tla="-S stm_michel_extra={$tla}"
    ARM=$arm DET=$det SRC=$src JOBS=$JOBS PIN=$PIN PR_TLA="$pr_tla" \
        LOGD=$LOGD/arm_$arm "$X/d53_run_arms.sh" > "$LOGD/arm_${arm}.log" 2>&1
    local rc=$?
    local n=$(ls $WORK/$det/work/*_$arm/tracking-pr.root 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] $arm rc=$rc events_with_output=$n"
    [ "$n" -eq 0 ] && echo "[$(date +%H:%M:%S)] *** $arm PRODUCED NOTHING -- rc=$rc is a dead net, not a pass ***"
}

# Clear this wave's completion marker at the START.  A marker left by an earlier wave is a
# booby trap: it makes a run that is a quarter done look finished, and a gate taken on it would
# be a gate on partial data.  Progress is ALWAYS judged by event count, never by a marker or an
# exit code (feedback_runner_rc1_dead_check).
rm -f "$LOGD/DONE_$WAVE"

case "$WAVE" in
    off)
        run_arm p95voff pdvd ""
        run_arm p95hoff pdhd ""
        ;;
    on)
        run_arm p95vq2d pdvd "$ON_TLA"
        ;;
    confirm)
        # no TLA: the flipped pdvd/wct-pr-perevt.jsonnet IS the config
        run_arm p95vprod pdvd ""
        ;;
    *)
        echo "unknown WAVE=$WAVE (want off|on|confirm)" >&2; exit 2 ;;
esac
touch "$LOGD/DONE_$WAVE"
echo "ARMS_DONE $WAVE"
