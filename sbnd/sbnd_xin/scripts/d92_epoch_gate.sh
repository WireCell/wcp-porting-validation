#!/bin/bash
# doc 92 refresh -- does the CURRENT toolkit tree still reproduce the prod0902
# reference arms?
#
# WHY THIS EXISTS.  The prod0902 arms (work-*-d97fv / -d97fvpr2) were produced
# by a binary pinned at ~/tmp/d97b-libsnap, whose libWireCellClus.so is dated
# 2026-09-02 10:59.  NINE toolkit commits landed after that, and five of them
# touch clus/ sources that SBND shares with PDVD:
#   fb0579c5  CreateSteinerGraph, DynamicPointCloud, Facade_Cluster, SteinerGrapher,
#             TaggerCheckSTM, TrackFitting, clustering_flag_matched_mains, improvecluster_2
#   54172df8  DynamicPointCloud, NeutrinoPatternBase
#   ee1a0d21  DynamicPointCloud
#   03f7645b  TaggerCheckSTM
#   e88f364d  NeutrinoTaggerNuE
# Every one is a PDVD crash-path fix and is *expected* to be a no-op on SBND --
# but "expected" is not a gate, and doc 92 tells a colleague the epoch is at
# e88f364d.  prod_cfg_gate.py 21/21 proves the CONFIG did not move; it says
# nothing about the binary.  This closes that.
#
# Both stages are re-run: the changed sources run in the Q/L job as well as the
# PR job, so a PR-only gate would be blind to half of them.
#
#   ./scripts/d92_epoch_gate.sh            # 308-event manifest, both stages, then gate
#
# Fresh labels (M13): work-<s>-d92gate (stage A Q/L), work-<s>-d92gatepr (stage B).
# NO -sep-fv-point flag is passed: the knob is the production default now, so a
# bare run is the test of exactly that.
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
export LIBSNAP=${LIBSNAP:-$HOME/tmp/d92gate-libsnap}
export LD_LIBRARY_PATH=$LIBSNAP:${LD_LIBRARY_PATH:-}
export D97_JOBS=${D97_JOBS:-8}          # CLAUDE.md M5 cap, not 12
export PR_JOBS=${PR_JOBS:-8}            # CLAUDE.md M5 cap, not 16
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES:-pr_display}
REF=$BASE/ref/prod-2026-09-04
LOGD=/home/xqian/tmp/d92gate; mkdir -p "$LOGD"
rc_all=0

for smp in ncpi0 nuecc48 mcp1k; do
    EV=$REF/gate308-$smp.txt
    [ -s "$EV" ] || { echo "MISSING manifest $EV"; exit 2; }
    QL=$BASE/work-$smp-d92gate
    PR=$BASE/work-$smp-d92gatepr
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
echo "=== D92 EPOCH GATE ARMS DONE rc_all=$rc_all $(date -Is)"
exit $rc_all
