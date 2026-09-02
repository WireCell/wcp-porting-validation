#!/bin/bash
# doc 97 -- positive control: the doc-95 25-event MC debug sample through a
# fresh Q/L stage with sep_track_recarve OFF and ON, then the PR chain on both.
#
# The two owner symptom events live here, not in the data samples:
#   272-2-30  = group a, evt 30   (doc 96 sec 2: 423 cm main = 412 cm cosmic
#                                  touching a 343 cm track at 0.35 cm)
#   105-23-21 = group a, evt 21   (doc 96 sec 3)
#   105-23-5  = group a, evt 5    (doc 96 sec 4, out of scope for separation)
#
# MC: reality=sim throughout (doc 95 -- running these as `data` shifts every
# point by ~6.8 cm in y-z through switch_scope's pos_offset).
#
# Usage: [D97_JOBS=6] ./scripts/d97_dbg25_arm.sh <off|on|both> [grp ...]
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
export LD_LIBRARY_PATH=$HOME/tmp/doc94r3b-libsnap:${LD_LIBRARY_PATH:-}
export PR_CFG_TREE=$HOME/tmp/dbg25-cfgsnap
export SBND_MAX_JOBS=${D97_JOBS:-6}
export PR_JOBS=${D97_JOBS:-6}
LOGD=/home/xqian/tmp/d97; mkdir -p "$LOGD"
ARMS=${1:-both}; shift || true
DBG_GROUPS=${*:-a b}       # never GROUPS -- bash special variable
[ "$ARMS" = both ] && ARMS="off on"
rc_all=0

for arm in $ARMS; do
    case $arm in
        off) FLAG="" ;;
        on)  FLAG="-sep-recarve" ;;
        *) echo "arm must be off|on|both" >&2; exit 2 ;;
    esac
    for grp in $DBG_GROUPS; do
        IN=$BASE/input_files_reco1/extracted-dbg25$grp
        SRC=$BASE/work-dbg25$grp-ql
        QL=$BASE/work-dbg25$grp-d97$arm
        PR=$BASE/work-dbg25$grp-d97${arm}pr
        [ -d "$SRC" ] || { echo "ERROR: no source Q/L root $SRC" >&2; exit 1; }
        [ -e "$QL" ] && { echo "SKIP $QL exists (M13)"; continue; }
        mkdir -p "$QL"
        # imaging is symlinked, never regenerated (M11)
        for d in "$SRC"/evt*; do ln -sfn "$d" "$QL/$(basename "$d")"; done
        echo "=== dbg25$grp arm=$arm  flag='$FLAG'  $(date -Is)"
        ( export SBND_INPUT_DIR=$IN SBND_WORK_ROOT=$QL
          setarch x86_64 -R ./run_ql_evt.sh mc $FLAG -save-pctree all
        ) > "$LOGD/dbg25$grp-$arm-ql.log" 2>&1
        echo "    ql rc=$?  ql_evt dirs=$(find "$QL" -maxdepth 1 -type d -name 'ql_evt*' | wc -l)"
        echo "    track_recarve fires: $(grep -l 'Separate track_recarve' "$QL"/ql_evt*/wct_ql_evt*.log 2>/dev/null | wc -l) events"
        [ -e "$PR" ] && { echo "SKIP $PR exists (M13)"; continue; }
        PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh "$QL" "$PR" sim \
            > "$LOGD/dbg25$grp-$arm-pr.log" 2>&1
        rc=$?
        echo "    pr rc=$rc  pr_evt dirs=$(find "$PR" -maxdepth 1 -type d -name 'pr_evt*' | wc -l)"
        [ "$rc" -ne 0 ] && rc_all=1
    done
done
echo "=== DBG25 ARMS DONE rc_all=$rc_all $(date -Is)"
exit $rc_all
