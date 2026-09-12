#!/usr/bin/env bash
# doc pdvd/96 -- the arms for the region's SCOPE (doc pdvd/95 sec 9 item 1).
#
# Five arms in four waves.  As in doc 95 the OFF wave must PASS before anything else is worth
# reading: if the knob-off path is not byte-identical to today's production, every later number
# is measuring my bug rather than the owner's definition.
#
#   WAVE=off     p96voff (pdvd) + p96hoff (pdhd) -- no q2d key at all.
#                Gate: p96voff == p95vprodb (TODAY'S PRODUCTION, which already carries doc 95's
#                four keys), p96hoff == p95hoffb.  Note the PDVD baseline is NOT p93vprod any
#                more: doc 95 flipped four keys into wct-pr-perevt.jsonnet, so production moved.
#   WAVE=base    p96vbase -- doc 95's measurement config with michel_q2d_region_scope written
#                EXPLICITLY at its C++ default 0.  Two jobs at once: it is the same-binary
#                partner for the scope comparison, and against p95vq2db it proves the new key at
#                0 is inert across binaries (doc 95's pin was deleted, so drift cannot be
#                assumed away).  That second check is a NICE-TO-HAVE -- see the fallback below.
#   WAVE=scope   p96vscope -- the measurement arm: the same config at scope 1.
#   WAVE=confirm p96vprod -- the FLIPPED file with NO TLA, to prove the config route gives
#                exactly what the TLA route measured.  Must run on the SAME pin.
#
# WHY R STAYS 40 HERE.  40 cm is the DIAGNOSTIC radius, as in doc 95: the cell table is emitted
# for every cell in radius regardless of the scope filter, so ONE arm at scope 1 answers scope 0
# and scope 2 offline as well (d96_scope.py re-cuts on own_blob).  Production keeps doc 95's
# tight R=10 -- the owner's constraint is that the Michel sits near the stop, and this round
# changes the scope, not the radius.
#
# THE FALLBACK, registered in pred.txt as X2.  If p96vbase != p95vq2db, the p95 pin no longer
# exists so compiler/env drift cannot be excluded a priori.  In that case scope 0 is derived
# OFFLINE from p96vscope's own unfiltered cell table and the scope comparison rests on one arm
# and one binary -- which is stronger than a cross-binary baseline anyway.  Reported, not chased.
#
# Fork by duplication (CLAUDE.md M10) of d95_arms.sh.  Nothing here writes into an existing
# work/ tag (M13): every arm refuses if its own dirs already exist.
set -uo pipefail
X=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts
WORK=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
PIN=${PIN:-/home/xqian/tmp/p96/libpin_p96}
JOBS=${JOBS:-5}
LOGD=${LOGD:-/home/xqian/tmp/p96}
WAVE=${WAVE:-off}

# Every key is written explicitly, INCLUDING michel_q2d_region_scope at its C++ default in the
# base arm.  Doc 95's lesson (feedback_flip_proof_matches_measured_arm): write the inert key too,
# so "the flipped file runs what this arm ran" stays a 0-line claim -- and so that a key sitting
# at its default is PROVEN inert rather than assumed so.
BASE_TLA=${BASE_TLA:-"michel_q2d:true,michel_q2d_cells:true,michel_q2d_region_cm:40.0,michel_q2d_region_ctl_cm:35.0,michel_q2d_region_scope:0"}
SCOPE_TLA=${SCOPE_TLA:-"michel_q2d:true,michel_q2d_cells:true,michel_q2d_region_cm:40.0,michel_q2d_region_ctl_cm:35.0,michel_q2d_region_scope:1"}

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
    # (feedback_runner_rc1_dead_check -- verify by event count, never by the exit code).
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

# Clear this wave's completion marker at the START.  A marker left by an earlier wave is a booby
# trap: it makes a run that is a quarter done look finished, and a gate taken on it would be a
# gate on partial data.  Progress is ALWAYS judged by event count, never by a marker or an exit
# code (feedback_runner_rc1_dead_check).
rm -f "$LOGD/DONE_$WAVE"

case "$WAVE" in
    off)
        run_arm p96voff pdvd ""
        run_arm p96hoff pdhd ""
        ;;
    base)
        run_arm p96vbase pdvd "$BASE_TLA"
        ;;
    scope)
        run_arm p96vscope pdvd "$SCOPE_TLA"
        ;;
    confirm)
        # no TLA: the flipped pdvd/wct-pr-perevt.jsonnet IS the config
        run_arm p96vprod pdvd ""
        ;;
    *)
        echo "unknown WAVE=$WAVE (want off|base|scope|confirm)" >&2; exit 2 ;;
esac
touch "$LOGD/DONE_$WAVE"
echo "ARMS_DONE $WAVE"
