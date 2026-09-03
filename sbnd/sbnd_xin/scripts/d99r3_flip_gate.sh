#!/bin/bash
# doc 99 round 3 -- the FLIP gate: does production, with the knobs flipped ON in
# the jsonnet defaults, reproduce the arms that round 2 gated at 20156/20156?
#
# WHY THIS IS SMALL ON PURPOSE.  Round 2's arms were produced by overriding the
# knobs through the runner's TLA hatch (QL_EXTRA_TLA / PR_EXTRA_TLA) on a tree
# whose jsonnet defaults were false.  Round 3 moves the value into the defaults
# instead.  The compiled-config proof (doc 99 sec 10) shows the two are
# BYTE-IDENTICAL JSON -- pre-flip tree + explicit TLA == post-flip tree with no
# TLA at all -- so a full 308-event re-run would be testing wcsonnet
# determinism, not the change.  What it CANNOT prove is that the runner reaches
# the flipped default when no hatch is set, because the runner, not wcsonnet,
# assembles the command line.  That is what this arm tests, end to end, on one
# sample.
#
# THE BINARY IS PINNED and deliberately not the live local/lib: a concurrent
# session landed two clus commits after round 2's arms were produced, so an
# unpinned run would compare a config change against a moving binary and the
# byte gates below could fail for a reason that is not the flip (CLAUDE.md M1,
# and the shared-tree pin lesson).
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
export LIBSNAP=${LIBSNAP:-$HOME/tmp/d99r2-libsnap}
export LD_LIBRARY_PATH=$LIBSNAP:${LD_LIBRARY_PATH:-}
export D97_JOBS=${D97_JOBS:-8}
export PR_JOBS=${PR_JOBS:-8}
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES:-pr_display}
REF=$BASE/ref/prod-2026-09-04          # the manifest; the NEW ref is cut after this passes
SAMPLES=${SAMPLES:-ncpi0}
LOGD=/home/xqian/tmp/d99r3gate; mkdir -p "$LOGD"
: > "$LOGD/empty.txt"                  # no TLA hatch -- the whole point
rc_all=0

echo "libsnap: $LIBSNAP  libWireCellClus.so $(date -r "$LIBSNAP/libWireCellClus.so" '+%F %T')"
echo "         libWireCellMatch.so $(date -r "$LIBSNAP/libWireCellMatch.so" '+%F %T')"

for smp in $SAMPLES; do
    EV=$REF/gate308-$smp.txt
    [ -s "$EV" ] || { echo "MISSING manifest $EV"; exit 2; }
    OUT=$BASE/work-$smp-d99r3prod
    if [ -e "$OUT" ]; then echo "SKIP $OUT exists (M13)"; else
        echo "=== $smp Q/L d99r3prod ($(grep -c '[0-9]' "$EV") events)  start $(date -Is)"
        QL_EXTRA_TLA=$LOGD/empty.txt QL_EXTRA="-save-pctree" ROOT=$OUT SRC=$BASE/work-$smp-grp0825 \
            ./scripts/d97_ql_arm.sh "$smp" -f "$EV" > "$LOGD/$smp-ql.log" 2>&1
        rc=$?; echo "=== $smp Q/L rc=$rc ql_evt=$(find "$OUT" -maxdepth 1 -type d -name 'ql_evt*' | wc -l) $(date -Is)"
        [ "$rc" -ne 0 ] && rc_all=1
    fi
    OUTPR=$BASE/work-$smp-d99r3prodpr
    if [ -e "$OUTPR" ]; then echo "SKIP $OUTPR exists (M13)"; else
        echo "=== $smp PR d99r3prodpr  start $(date -Is)"
        PR_EXTRA_TLA=$LOGD/empty.txt ./run_pr_chain_batch.sh "$OUT" "$OUTPR" data $(cat "$EV") \
            > "$LOGD/$smp-pr.log" 2>&1
        rc=$?; echo "=== $smp PR rc=$rc pr_evt=$(find "$OUTPR" -maxdepth 1 -type d -name 'pr_evt*' | wc -l) $(date -Is)"
        [ "$rc" -ne 0 ] && rc_all=1
    fi
done
echo "=== ARMS DONE rc_all=$rc_all  loadavg: $(cat /proc/loadavg)"
[ "$rc_all" -eq 0 ] || { echo "arms failed; not gating"; exit $rc_all; }

SMPCSV=$(echo $SAMPLES | tr ' ' ',')
gate_rc=0
run() { echo; echo "--- $1"; shift; "$@"; rc=$?; echo "    rc=$rc"; [ $rc -eq 0 ] || gate_rc=1; }
report() { echo; echo "--- $1 (report only)"; shift; "$@"; echo "    rc=$? (not gated)"; }

echo; echo "########## THE FLIP REPRODUCES THE GATED ARMS, BYTE FOR BYTE"
run "stage A: production defaults == the WRITE-on arm (d99r2wr)" \
    python3 scripts/d97_ql_gate.py d99r3prod d99r2wr $SAMPLES
for smp in $SAMPLES; do
    run "stage B archives, $smp: d99r3prodpr == d99r2bothpr" \
        python3 scripts/pr85_hash_gate.py "work-$smp-d99r2bothpr" "work-$smp-d99r3prodpr"
done
# EXPECT empty: the claim is that NOTHING moves, not that only X moves.
run "every ROOT branch: d99r3prodpr == d99r2bothpr (expect: NOTHING)" \
    python3 scripts/analysis/d99_root_branch_census.py d99r2bothpr d99r3prodpr \
        --samples "$SMPCSV" --expect ""

echo; echo "########## AND IT IS THE CORRECT ONE"
run "every matched row resolves ITS OWN flash" \
    python3 scripts/analysis/d99_tcluster_flash_check.py d99r3prodpr --samples "$SMPCSV" \
        --require-correct --out "$LOGD/tc-prodpr.tsv"
# The instrument's causal negative control: the SAME check on the pre-flip
# production arm must NOT be clean.  A checker that cannot fail is not a gate.
report "pre-flip production (d99r2offpr) on the same events -- must NOT be 100%" \
    python3 scripts/analysis/d99_tcluster_flash_check.py d99r2offpr --samples "$SMPCSV"

echo
echo "=== D99 ROUND 3 FLIP GATE: $([ $gate_rc -eq 0 ] && echo PASS || echo FAIL)  $(date -Is)"
exit $gate_rc
