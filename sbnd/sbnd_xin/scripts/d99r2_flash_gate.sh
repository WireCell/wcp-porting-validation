#!/bin/bash
# doc 99 round 2 -- the two flash-resolution fixes: arms + every gate leg.
#
# WHAT IS UNDER TEST.  Two default-OFF knobs, one per side of the same defect:
#
#   READ  SbndPrMagnifyTrackingVisitor.flash_by_gid  (PR stage, C++ default false)
#         T_cluster's flash columns come from Cluster::get_matched_flash() --
#         matched_flash_gid resolved against the merge-safe "opflash" PC --
#         instead of Cluster::get_flash(), the per-input row index.
#   WRITE QLMatching.merge_flash_pcs                 (Q/L stage, C++ default false)
#         The multi-input merge carries EVERY input's canonical flash/light/
#         flashlight/flashcov PCs (re-basing each input's flash-row references)
#         instead of keeping only the primary input's and dropping the rest.
#
# THE PRIMARY INSTRUMENT is not an A/B at all: T_cluster carries cluster_t0_us
# (written from the flash the cluster ACTUALLY matched) and flash_time_us
# (written from whatever the reader resolved) on the same row, so
# flash_time_us == cluster_t0_us is a within-file identity that grades either
# fix with no baseline arm and no cross-stage join.  Measured on the pre-fix arm
# work-*-d99fixpr: 10219/20175 = 50.7% CORRECT.
#
#   ./scripts/d99r2_flash_gate.sh
#
# Fresh labels (M13), 308-event manifest, ref/prod-2026-09-04/gate308-*.txt:
#   work-<s>-d99r2off    stage A, both knobs OFF   -> must equal work-<s>-d99fix
#   work-<s>-d99r2offpr  stage B, OFF              -> must equal work-<s>-d99fixpr
#   work-<s>-d99r2rdpr   stage B, READ on          (from d99r2off)
#   work-<s>-d99r2wr     stage A, WRITE on         -- NOT byte-identical, by design
#   work-<s>-d99r2wrpr   stage B, read OFF         (from d99r2wr)
#   work-<s>-d99r2bothpr stage B, READ on          (from d99r2wr) -- cross-check
#
# THE OFF LEG RUNS WITH AN EMPTY EXPECT LIST ON PURPOSE.  Round 1's gate
# defaulted EXPECT to the three flash columns because that round's claim was
# containment of an unavoidable change.  This round's OFF claim is strictly
# stronger -- nothing moves at all -- and reusing that default would silently
# permit the very change under test.
set -u
cd -P "$(dirname "$0")/.." || exit 1
BASE=$PWD
export LIBSNAP=${LIBSNAP:-$HOME/tmp/d99r2-libsnap}   # both fixes, pinned
export LD_LIBRARY_PATH=$LIBSNAP:${LD_LIBRARY_PATH:-}
export D97_JOBS=${D97_JOBS:-8}          # CLAUDE.md M5 cap
export PR_JOBS=${PR_JOBS:-8}            # CLAUDE.md M5 cap
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES:-pr_display}
REF=$BASE/ref/prod-2026-09-04
SAMPLES=${SAMPLES:-ncpi0 nuecc48 mcp1k}
LOGD=/home/xqian/tmp/d99r2gate; mkdir -p "$LOGD"
TLAD=$LOGD/tla; mkdir -p "$TLAD"
printf 'merge_flash=true\n'  > "$TLAD/ql-write-on.txt"
printf 'flash_by_gid=true\n' > "$TLAD/pr-read-on.txt"
: > "$TLAD/empty.txt"
rc_all=0

echo "libsnap: $LIBSNAP  libWireCellClus.so $(date -r "$LIBSNAP/libWireCellClus.so" '+%F %T')"
echo "         libWireCellMatch.so $(date -r "$LIBSNAP/libWireCellMatch.so" '+%F %T')"
echo "         libWireCellRoot.so  $(date -r "$LIBSNAP/libWireCellRoot.so" '+%F %T')"

# ---------------------------------------------------------------------------
# Arms.  Existing labels are skipped (M13), so a re-run just re-gates.
ql_arm() {   # ql_arm <tag> <tlafile>
    local tag=$1 tla=$2 smp EV n
    for smp in $SAMPLES; do
        EV=$REF/gate308-$smp.txt
        [ -s "$EV" ] || { echo "MISSING manifest $EV"; return 2; }
        local OUT=$BASE/work-$smp-$tag
        if [ -e "$OUT" ]; then echo "SKIP $OUT exists (M13)"; continue; fi
        n=$(grep -c '[0-9]' "$EV")
        echo "=== $smp Q/L $tag ($n events)  start $(date -Is)"
        QL_EXTRA_TLA=$tla QL_EXTRA="-save-pctree" ROOT=$OUT SRC=$BASE/work-$smp-grp0825 \
            ./scripts/d97_ql_arm.sh "$smp" -f "$EV" > "$LOGD/$smp-$tag-ql.log" 2>&1
        local rc=$?
        echo "=== $smp Q/L $tag rc=$rc  ql_evt=$(find "$OUT" -maxdepth 1 -type d -name 'ql_evt*' | wc -l)  $(date -Is)"
        [ "$rc" -ne 0 ] && rc_all=1
    done
}

pr_arm() {   # pr_arm <qltag> <prtag> <tlafile>
    local qltag=$1 prtag=$2 tla=$3 smp EV
    for smp in $SAMPLES; do
        EV=$REF/gate308-$smp.txt
        local QL=$BASE/work-$smp-$qltag OUT=$BASE/work-$smp-$prtag
        if [ -e "$OUT" ]; then echo "SKIP $OUT exists (M13)"; continue; fi
        [ -d "$QL" ] || { echo "MISSING Q/L root $QL"; rc_all=1; continue; }
        echo "=== $smp PR $prtag (from $qltag)  start $(date -Is)"
        PR_EXTRA_TLA=$tla ./run_pr_chain_batch.sh "$QL" "$OUT" data $(cat "$EV") \
            > "$LOGD/$smp-$prtag-pr.log" 2>&1
        local rc=$?
        echo "=== $smp PR $prtag rc=$rc  pr_evt=$(find "$OUT" -maxdepth 1 -type d -name 'pr_evt*' | wc -l)  $(date -Is)"
        [ "$rc" -ne 0 ] && rc_all=1
    done
}

ql_arm d99r2off "$TLAD/empty.txt"
ql_arm d99r2wr  "$TLAD/ql-write-on.txt"
pr_arm d99r2off d99r2offpr  "$TLAD/empty.txt"
pr_arm d99r2off d99r2rdpr   "$TLAD/pr-read-on.txt"
pr_arm d99r2wr  d99r2wrpr   "$TLAD/empty.txt"
pr_arm d99r2wr  d99r2bothpr "$TLAD/pr-read-on.txt"

echo "=== D99R2 ARMS DONE rc_all=$rc_all $(date -Is)"
echo "loadavg: $(cat /proc/loadavg)"
[ "$rc_all" -eq 0 ] || { echo "arms failed; not gating"; exit $rc_all; }

# ---------------------------------------------------------------------------
# The gate.  Every leg lives here, not in the doc: an instruction that exists
# only in prose is untested (doc 99 round 1 learned that twice in one run).
SMPCSV=$(echo $SAMPLES | tr ' ' ',')
FLASHCOLS=T_cluster:flash_id,T_cluster:flash_time_us,T_cluster:flash_pe
gate_rc=0
run() { echo; echo "--- $1"; shift; "$@"; rc=$?; echo "    rc=$rc"; [ $rc -eq 0 ] || gate_rc=1; }
# report(): same, but its rc never fails the gate -- for the legs whose whole
# point is that something DID change.  Kept distinct from run() so a reader can
# see at a glance which legs are load-bearing.
report() { echo; echo "--- $1 (report only)"; shift; "$@"; echo "    rc=$? (not gated)"; }

echo; echo "########## A. BOTH KNOBS OFF IS INERT (vs the round-1 arms, older binary)"
run "stage A, Q/L member content: d99r2off == d99fix" \
    python3 scripts/d97_ql_gate.py d99r2off d99fix $SAMPLES
for smp in $SAMPLES; do
    run "stage B archives, $smp: d99r2offpr == d99fixpr" \
        python3 scripts/pr85_hash_gate.py "work-$smp-d99fixpr" "work-$smp-d99r2offpr"
done
# EXPECT deliberately EMPTY: the OFF claim is "nothing moved", not "only X moved".
run "every ROOT branch, OFF vs round 1 (expect: NOTHING)" \
    python3 scripts/analysis/d99_root_branch_census.py d99fixpr d99r2offpr \
        --samples "$SMPCSV" --expect ""

echo; echo "########## B. THE READ FIX (PR stage only; same Q/L root, same binary)"
run "baseline: OFF arm resolves its own flash on ~half the rows" \
    python3 scripts/analysis/d99_tcluster_flash_check.py d99r2offpr --samples "$SMPCSV" \
        --out "$LOGD/tc-offpr.tsv"
run "READ on: every matched row resolves ITS OWN flash" \
    python3 scripts/analysis/d99_tcluster_flash_check.py d99r2rdpr --samples "$SMPCSV" \
        --require-correct --out "$LOGD/tc-rdpr.tsv"
run "containment: the read knob moves the three flash columns and nothing else" \
    python3 scripts/analysis/d99_root_branch_census.py d99r2offpr d99r2rdpr \
        --samples "$SMPCSV" --expect "$FLASHCOLS"
for smp in $SAMPLES; do
    run "stage B archives unmoved by the read knob, $smp" \
        python3 scripts/pr85_hash_gate.py "work-$smp-d99r2offpr" "work-$smp-d99r2rdpr"
done

echo; echo "########## C. THE WRITE FIX (Q/L stage; archives MOVE by design)"
report "stage A archives DIFFER -- this is the point, not a regression" \
    python3 scripts/d97_ql_gate.py d99r2wr d99r2off $SAMPLES
run "WRITE on, read OFF: get_flash() resolves ITS OWN flash" \
    python3 scripts/analysis/d99_tcluster_flash_check.py d99r2wrpr --samples "$SMPCSV" \
        --require-correct --out "$LOGD/tc-wrpr.tsv"

echo; echo "########## D. THE TWO PATHS AGREE (the strongest single check)"
# Merged row index (write fix) and matched_flash_gid (read fix) are independent
# resolutions of the same question.  Agreement row-for-row is what says the two
# fixes are both right rather than compensating for each other.
run "WRITE-on arm read by index vs by gid: identical flash_time_us / flash_pe" \
    python3 scripts/analysis/d99_tcluster_flash_check.py d99r2wrpr \
        --compare d99r2bothpr --samples "$SMPCSV" --require-correct

echo
echo "=== D99 ROUND 2 GATE VERDICT: $([ $gate_rc -eq 0 ] && echo PASS || echo FAIL)  $(date -Is)"
exit $gate_rc
