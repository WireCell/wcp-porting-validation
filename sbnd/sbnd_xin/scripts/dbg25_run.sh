#!/bin/bash
# doc 95 -- the colleague's 25-event SBND *MC* debug sample through our
# production chain at the pinned operating point ref/prod-2026-09-03
# (stm_entry_rise_guard ON).  Fork of scripts/stmfb8_run.sh (doc 93).
#
#   input_files_reco1/stm_tagger_feedback/debug-25evt-reco1.root   (25 entries)
#   -> staged per entry  (scripts/dbg25_stage.sh)
#   -> two collision-free sample dirs (scripts/dbg25_groups.sh):
#        extracted-dbg25a  20 events   extracted-dbg25b  5 events
#      See dbg25_groups.sh for WHY two: 25 distinct RSE, only 20 distinct
#      bare event ids, and every work dir is keyed on the bare id.
#
# MC, not data (branch scan: recob::Wires_simtpc2d_dnnsp_DetSim., no
# FrameShiftInfo/PTB/TDC):
#   - the dump ran -mc -caf none
#   - reality=sim throughout; run_ql_evt.sh maps mc -> sim.  This gates the
#     switch_scope pos_offset transverse correction (doc pr/38 round 3), so
#     running these as `data` would shift every point by ~6.8 cm in y-z.
#
# Binary pinned to ~/tmp/doc94r3b-libsnap (md5-verified equal to local/lib at
# launch); PR cfg tree pinned to ~/tmp/dbg25-cfgsnap (a copy of toolkit/cfg
# taken at launch, prod_cfg_gate.py PASS 21/21 vs ref/prod-2026-09-03).
# Stages A1/A2 read toolkit/cfg directly -- run_img_evt.sh and run_ql_evt.sh
# hardcode it ahead of $WIRECELL_PATH, so they cannot be pinned without
# forking them; the live tree is hash-fenced before/after instead.
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
export LD_LIBRARY_PATH=$HOME/tmp/doc94r3b-libsnap:${LD_LIBRARY_PATH:-}
export PR_CFG_TREE=$HOME/tmp/dbg25-cfgsnap
export SBND_MAX_JOBS=${SBND_MAX_JOBS:-6}
export PR_JOBS=${PR_JOBS:-6}
LOGD=/home/xqian/tmp/dbg25; mkdir -p "$LOGD"

STAGE=${1:-all}
# NB: NOT named GROUPS -- that is a bash special variable (the
# caller's group list); assignments to it are silently ignored and
# $GROUPS then expands to the primary gid, so the loop runs once on
# a nonexistent group.
DBG_GROUPS=${2:-a b}
rc_all=0

for grp in $DBG_GROUPS; do
    IN=$BASE/input_files_reco1/extracted-dbg25$grp
    QL=$BASE/work-dbg25$grp-ql
    PR=$BASE/work-dbg25$grp-pr
    [ -d "$IN" ] || { echo "ERROR: no $IN" >&2; exit 1; }
    nexp=$(tar tjf "$IN/frames-dnn.tar.bz2" | grep -c '^frame_dnnsp_')
    echo "=== group $grp  expect $nexp events  $(date -Is)"
    export SBND_INPUT_DIR=$IN
    export SBND_WORK_ROOT=$QL
    mkdir -p "$QL"

    if [ "$STAGE" = all ] || [ "$STAGE" = img ]; then
        echo "--- imaging  $(date -Is)"
        ./run_img_evt.sh mc all > "$LOGD/img-$grp.log" 2>&1
        echo "img rc=$?  evt dirs=$(find "$QL" -maxdepth 1 -type d -name 'evt*' | wc -l) / $nexp"
    fi

    if [ "$STAGE" = all ] || [ "$STAGE" = ql ]; then
        echo "--- Q/L  $(date -Is)"
        ./run_ql_evt.sh mc -save-pctree all > "$LOGD/ql-$grp.log" 2>&1
        echo "ql rc=$?  ql dirs=$(find "$QL" -maxdepth 1 -type d -name 'ql_evt*' | wc -l) / $nexp"
        # run_ql_evt.sh does not stamp lineage; run_pr_chain_batch.sh SILENTLY
        # skips its reality check when the stamp and the .batch_* markers are
        # both absent, so write it by hand (doc pr/38 round 3, doc 93).
        echo sim > "$QL/.lineage_reality"
    fi

    if [ "$STAGE" = all ] || [ "$STAGE" = pr ]; then
        echo "--- PR  $(date -Is)"
        export PR_EXTRA_STAGES=pr_display     # keep the calib dump (doc 92 sec 2.3)
        ./run_pr_chain_batch.sh "work-dbg25$grp-ql" "work-dbg25$grp-pr" sim \
            > "$LOGD/pr-$grp.log" 2>&1
        echo "pr rc=$?  pr dirs=$(find "$PR" -maxdepth 1 -type d -name 'pr_evt*' | wc -l) / $nexp"
    fi
done
echo "=== done $(date -Is) rc_all=$rc_all"
