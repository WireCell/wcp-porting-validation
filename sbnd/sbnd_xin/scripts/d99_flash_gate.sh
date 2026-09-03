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
[ "$rc_all" -eq 0 ] || { echo "arms failed; not gating"; exit $rc_all; }

# ---------------------------------------------------------------------------
# The gate itself.  This lives HERE, not in the doc, on purpose: the expected-
# diff column list is the whole subtlety of this gate, and an instruction that
# only exists in prose has to be found and retyped correctly by whoever re-runs
# it.  REFARM and NEWARM are parameters so this doubles as the post-master-merge
# gate: point NEWARM at a fresh arm and REFARM at the last good one.
#
# DELIBERATELY NOT named BASE/ARM.  This script already uses BASE for $PWD
# (inherited from d92_epoch_gate.sh), so `BASE=${BASE:-d92gate}` silently kept
# the PATH: every leg looked for work-<smp>-/home/.../sbnd_xin, three failed
# loudly and two reported PASS on zero events.  Both bugs are fixed (the tools
# now refuse an empty comparison); the names stay distinct so it cannot recur.
REFARM=${REFARM:-d92gate}
NEWARM=${NEWARM:-d99fix}
EXPECT=T_cluster:flash_id,T_cluster:flash_time_us,T_cluster:flash_pe
gate_rc=0
run() { echo; echo "--- $1"; shift; "$@"; rc=$?; echo "    rc=$rc"; [ $rc -eq 0 ] || gate_rc=1; }

run "stage A, Q/L member content" \
    python3 scripts/d97_ql_gate.py "$NEWARM" "$REFARM" ncpi0 nuecc48 mcp1k
for smp in ncpi0 nuecc48 mcp1k; do
    run "stage B archives, $smp" \
        python3 scripts/pr85_hash_gate.py "work-$smp-${REFARM}pr" "work-$smp-${NEWARM}pr"
done
# EXHAUSTIVE, unlike pr87_root_tree_diff.py, which breaks at the first differing
# branch of a tree and so cannot support a "nothing moved except these" claim.
run "every ROOT branch (expected diffs: $EXPECT)" \
    python3 scripts/analysis/d99_root_branch_census.py "${REFARM}pr" "${NEWARM}pr" \
        --samples ncpi0,nuecc48,mcp1k --expect "$EXPECT"
# The causal leg.  --stage pr is required: T_cluster is written from the PR-stage
# grouping, and the PR chain re-clusters (evt 59685: 10 clusters in Q/L, 22 in PR).
run "predict which rows may move (PR-stage census)" \
    python3 scripts/analysis/d99_flash_index_census.py --arm "work-{s}-${NEWARM}pr" \
        --stage pr --samples ncpi0,nuecc48,mcp1k \
        --out "$LOGD/census-pr.tsv" --detail "$LOGD/detail-pr.tsv" --jobs "$PR_JOBS"
run "the moved rows are exactly those, and read the sentinel" \
    python3 scripts/analysis/d99_flash_ab.py --a "${REFARM}pr" --b "${NEWARM}pr" \
        --samples ncpi0,nuecc48,mcp1k --detail "$LOGD/detail-pr.tsv"

echo
echo "=== D99 FLASH GATE VERDICT: $([ $gate_rc -eq 0 ] && echo PASS || echo FAIL)  $(date -Is)"
exit $gate_rc
