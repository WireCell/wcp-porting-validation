#!/bin/bash
# doc 99 -- containment gate for the Grouping::flash_at() range guard.
#
# THE CLAIM UNDER TEST.  The guard changes exactly three columns of the
# diagnostic tree T_cluster -- flash_id, flash_time_us, flash_pe -- and only on
# the rows whose cluster carried a matched-flash index past the end of the
# grouping's "flash" point cloud.  Everything else in both stages is unchanged.
#
# WHY A FULL TWO-STAGE RE-RUN.  Grouping::flash_at() is also on the Q/L path
# (QLMatching iterates Grouping::flashes()).  The guard is bounded on the very
# array flashes() sizes its loop with, so that path cannot move by construction
# -- but "by construction" is an argument, not a gate, and stage A is where it
# would hurt.  So both stages run.
#
# NOTE ON BYTE-IDENTITY.  This one is NOT a byte-identical gate and cannot be.
# The cells the fix changes held raw memory read past the end of an array; they
# were never reproducible in the first place (the doc-92 epoch gate found 48 of
# them differing between two otherwise byte-identical arms).  The claim is
# containment -- everything OUTSIDE those three columns is byte-identical -- plus
# the causal check that the rows which moved are exactly the ones the
# independent d99_flash_index_census.py predicted from the Q/L archives.
#
#   ./scripts/d99_flash_gate.sh
#
# Fresh labels (M13): work-<s>-d99fix (stage A Q/L), work-<s>-d99fixpr (stage B).
# Baseline for the comparison: work-<s>-d92gate / -d92gatepr, the pre-fix arms
# this same manifest produced at the same commit (e88f364d) on 2026-09-02.
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
export LIBSNAP=${LIBSNAP:-$HOME/tmp/d99-libsnap}   # post-fix binary, pinned
export LD_LIBRARY_PATH=$LIBSNAP:${LD_LIBRARY_PATH:-}
export D97_JOBS=${D97_JOBS:-8}          # CLAUDE.md M5 cap
export PR_JOBS=${PR_JOBS:-8}            # CLAUDE.md M5 cap
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES:-pr_display}
REF=$BASE/ref/prod-2026-09-04
LOGD=/home/xqian/tmp/d99gate; mkdir -p "$LOGD"
rc_all=0

echo "libsnap: $LIBSNAP  libWireCellClus.so $(date -r "$LIBSNAP/libWireCellClus.so" '+%F %T')"

for smp in ncpi0 nuecc48 mcp1k; do
    EV=$REF/gate308-$smp.txt
    [ -s "$EV" ] || { echo "MISSING manifest $EV"; exit 2; }
    QL=$BASE/work-$smp-d99fix
    PR=$BASE/work-$smp-d99fixpr
    n=$(grep -c '[0-9]' "$EV")
    if [ -e "$QL" ]; then echo "SKIP $QL exists (M13)"; else
        echo "=== $smp Q/L ($n events)  start $(date -Is)"
        QL_EXTRA="-save-pctree" ROOT=$QL SRC=$BASE/work-$smp-grp0825 \
            ./scripts/d97_ql_arm.sh "$smp" -f "$EV" > "$LOGD/$smp-ql.log" 2>&1
        rc=$?; echo "=== $smp Q/L rc=$rc  ql_evt=$(find "$QL" -maxdepth 1 -type d -name 'ql_evt*' | wc -l)  $(date -Is)"
        [ "$rc" -ne 0 ] && { rc_all=1; continue; }
    fi
    if [ -e "$PR" ]; then echo "SKIP $PR exists (M13)"; else
        echo "=== $smp PR  start $(date -Is)"
        ./run_pr_chain_batch.sh "$QL" "$PR" data $(cat "$EV") > "$LOGD/$smp-pr.log" 2>&1
        rc=$?; echo "=== $smp PR rc=$rc  pr_evt=$(find "$PR" -maxdepth 1 -type d -name 'pr_evt*' | wc -l)  $(date -Is)"
        [ "$rc" -ne 0 ] && rc_all=1
    fi
done
echo "=== D99 FLASH GATE ARMS DONE rc_all=$rc_all $(date -Is)"
echo "loadavg: $(cat /proc/loadavg)"
exit $rc_all
