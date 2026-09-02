#!/bin/bash
# doc 97 -- the sep_fv_point knob (the C++ fv_inset_yz + the three far-point /
# dec1-guard knobs) on the doc-95 25-event MC debug sample.
#
# This is the load-bearing control for the SECOND separation case: doc 96 sec
# 8.3 reached 105-23-21 by insetting the SHARED DetectorVolumes FV, which also
# moves clustering_neutrino and the containment taggers, so its rescue could
# not be attributed to separation.  sep_fv_point insets only inside
# clustering_separate, so if 105-23-21 still becomes a nu-candidate here the
# rescue IS the separation.
#
# Separate file from d97_dbg25_arm.sh, which is the record of the
# track_recarve control and was already run.
#
# Usage: [D97_JOBS=6] ./scripts/d97_dbg25_fv.sh [grp ...]
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
export LD_LIBRARY_PATH=$HOME/tmp/d97b-libsnap:${LD_LIBRARY_PATH:-}
export PR_CFG_TREE=$HOME/tmp/dbg25-cfgsnap
export SBND_MAX_JOBS=${D97_JOBS:-6}
export PR_JOBS=${D97_JOBS:-6}
LOGD=/home/xqian/tmp/d97; mkdir -p "$LOGD"
DBG_GROUPS=${*:-a b}       # never GROUPS -- bash special variable
rc_all=0
for grp in $DBG_GROUPS; do
    IN=$BASE/input_files_reco1/extracted-dbg25$grp
    SRC=$BASE/work-dbg25$grp-ql
    QL=$BASE/work-dbg25$grp-d97fv
    PR=$BASE/work-dbg25$grp-d97fvpr3
    [ -d "$SRC" ] || { echo "ERROR: no $SRC" >&2; exit 1; }
    if [ -e "$QL" ]; then echo "SKIP $QL exists (M13)"; else
        mkdir -p "$QL"
        for d in "$SRC"/evt*; do ln -sfn "$d" "$QL/$(basename "$d")"; done
        echo "=== dbg25$grp sep_fv_point  $(date -Is)"
        ( export SBND_INPUT_DIR=$IN SBND_WORK_ROOT=$QL
          setarch x86_64 -R ./run_ql_evt.sh mc -sep-fv-point -save-pctree all
        ) > "$LOGD/dbg25$grp-fv-ql.log" 2>&1
        echo "    ql rc=$?  ql_evt=$(find "$QL" -maxdepth 1 -type d -name 'ql_evt*' | wc -l)"
    fi
    [ -e "$PR" ] && { echo "SKIP $PR exists (M13)"; continue; }
    PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh "$QL" "$PR" sim \
        > "$LOGD/dbg25$grp-fv-pr.log" 2>&1
    rc=$?
    echo "    pr rc=$rc  pr_evt=$(find "$PR" -maxdepth 1 -type d -name 'pr_evt*' | wc -l)  $(date -Is)"
    [ "$rc" -ne 0 ] && rc_all=1
done
echo "=== DBG25 FV DONE rc_all=$rc_all $(date -Is)"
exit $rc_all
