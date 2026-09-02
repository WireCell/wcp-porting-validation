#!/bin/bash
# Colleague STM/TGM-tagger feedback sample: 8 SBND *MC* reco1 events through
# our production chain (epoch prod0901b).  Doc 93.
#
#   input_files_reco1/stm_tagger_feedback/type2-8evt-reco1.root  (8 entries)
#   RSE: 827/27/4  304/6/28  707/18/12  146/60/31  36/77/17  966/2/22
#        921/29/10  658/38/25
#
# MC, not data:
#   - reco1 products live under simtpc2d/DetSim, so the dump needs -mc
#   - no FrameShiftInfo/PTB/TDC products, so -caf none (key omitted)
#   - reality=sim throughout (run_ql_evt.sh maps mc -> sim); this gates the
#     switch_scope pos_offset transverse correction (doc pr/38 round 3)
# => run_chain_group.sh cannot be used (it hardcodes caf_offset_mode=product
#    and the DATA product names), so this is the per-event path of doc 67,
#    which doc 92 sec 3.2 records as byte-equivalent to group mode.
#
# Binary pinned to ~/tmp/prod0901b-libsnap (the prod0901b arm's snapshot);
# cfg tree is toolkit/cfg, prod_cfg_gate.py PASS 21/21 at 2026-09-01 19:26.
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
export LD_LIBRARY_PATH=$HOME/tmp/prod0901b-libsnap:${LD_LIBRARY_PATH:-}
export SBND_INPUT_DIR=$BASE/input_files_reco1/extracted-stmfb8
export SBND_WORK_ROOT=$BASE/work-stmfb8-ql
export SBND_MAX_JOBS=${SBND_MAX_JOBS:-6}
LOGD=/home/xqian/tmp/stmfb8; mkdir -p "$LOGD" "$SBND_WORK_ROOT"

STAGE=${1:-all}

if [ "$STAGE" = all ] || [ "$STAGE" = img ]; then
    echo "=== imaging  $(date -Is)"
    ./run_img_evt.sh mc all > "$LOGD/img.log" 2>&1
    echo "img rc=$?  npz dirs=$(find "$SBND_WORK_ROOT" -maxdepth 1 -type d -name 'evt*' | wc -l)"
fi

if [ "$STAGE" = all ] || [ "$STAGE" = ql ]; then
    echo "=== Q/L  $(date -Is)"
    ./run_ql_evt.sh mc -save-pctree all > "$LOGD/ql.log" 2>&1
    echo "ql rc=$?  ql dirs=$(find "$SBND_WORK_ROOT" -maxdepth 1 -type d -name 'ql_evt*' | wc -l)"
    # run_ql_evt.sh does not stamp lineage; run_pr_chain_batch.sh SILENTLY
    # skips its reality check when the stamp and the .batch_* markers are both
    # absent, so write it by hand (doc pr/38 round 3).
    echo sim > "$SBND_WORK_ROOT/.lineage_reality"
fi

if [ "$STAGE" = all ] || [ "$STAGE" = pr ]; then
    echo "=== PR  $(date -Is)"
    export PR_JOBS=${PR_JOBS:-6}
    export PR_EXTRA_STAGES=pr_display        # keep the calib dump (doc 92 sec 2.3)
    ./run_pr_chain_batch.sh work-stmfb8-ql work-stmfb8-pr sim > "$LOGD/pr.log" 2>&1
    echo "pr rc=$?  pr dirs=$(find "$BASE/work-stmfb8-pr" -maxdepth 1 -type d -name 'pr_evt*' | wc -l)"
fi
echo "=== done $(date -Is)"
