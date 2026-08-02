#!/bin/bash
# valfast: run the SBND PR chain over the 629-event "yields PR results" subset
# (valfast/README.md, derived from docs/pr/11_scores-table.tsv nu_evaluated=1)
# instead of the full 1071-event population, for fast A/B validation arms.
#
# Usage: valfast/run_valfast.sh <tag> [-full] [-j N] [sample ...]
#   <tag>     arm suffix; outputs land in work-vf<sample>-<tag> (PR out roots)
#             and, with -full, the nusel roots listed below
#   -full     ALSO regenerate the nusel (Q/L + tagger-tail) stage on the subset
#             before the PR chain, instead of reading the pinned ql_roots.
#             REQUIRED whenever the change under test can touch clustering /
#             Q-L products (the pinned ql_roots are frozen inputs).
#   -j N      parallelism (default 8; CLAUDE.md caps routine batches ~6-8)
#   sample    any of: mcp1k nuecc48 r1qlmc r2mc  (default: all four)
#
# PR-tail mode (default) -- ql_root is a pinned KEEP hub, both arms of an A/B
# share it byte-for-byte, so archive diffs are attributable to the change:
#   mcp1k   ql_root work-mcp1kall-d59k   (data, 572 events)
#   nuecc48 ql_root work-nuecc48-nuf     (data,  47 events)
#   r1qlmc  ql_root work-r1ql-first10    (sim,    5 events)
#   r2mc    ql_root work-r2patrec-f1     (sim,    5 events)
# CAVEAT: those pctrees predate the pr/14..pr/20 clustering defaults. Valid
# for A/B (same input both arms); NOT a reproduction of today's production
# clustering -- use -full for that.
#
# -full mode nusel roots (imaging is NEVER regenerated -- evt dirs are
# symlinks into the BASE hubs, M11/M13):
#   mcp1k   TAG=vf<tag> ENTRIES=... ./run_full1k_nusel.sh   -> work-mcp1kall-vf<tag>
#   nuecc48 s4_nuecc48.sh pattern (imaging links into work/) -> work-nuecc48-vf<tag>
#   r1qlmc  doc 67 recipe (f1 'all' + f2 idx 1 2)            -> work-r1qlmc-vf<tag>
#   r2mc    doc 67 recipe (f1 'all')                         -> work-r2mc-vf<tag>
# Per-sample nusel flags follow each sample's reference recipe: run_full1k
# carries -stm-fit internally, nuecc48 runs bare (s4_nuecc48.sh precedent),
# the MC samples pass -stm-fit (doc 67's $NUF).
#
# Knob env vars (SBND_*) pass through untouched to run_pr_chain_batch.sh /
# run_nusel_evt.sh. Existing output roots are REFUSED (M13) -- new run, new tag.
# valfast arms are TRANSIENT: record the valfast_compare.sh summary in the
# round doc, then delete the work-vf* / work-*-vf<tag> roots.
set -u
SBND_DIR=$(cd "$(dirname "$0")/.." && pwd -P)
VF=$SBND_DIR/valfast
cd "$SBND_DIR" || exit 1

TAG=${1:?usage: run_valfast.sh <tag> [-full] [-j N] [sample ...]}
shift
FULL=no; JOBS=8; SAMPLES=""
while [ $# -gt 0 ]; do
    case "$1" in
        -full) FULL=yes;;
        -j) JOBS=$2; shift;;
        mcp1k|nuecc48|r1qlmc|r2mc) SAMPLES="$SAMPLES $1";;
        *) echo "unknown arg: $1"; exit 1;;
    esac
    shift
done
[ -n "$SAMPLES" ] || SAMPLES="mcp1k nuecc48 r1qlmc r2mc"

reality() { case "$1" in mcp1k|nuecc48) echo data;; *) echo sim;; esac; }
pinned_qlroot() {
    case "$1" in
        mcp1k)   echo work-mcp1kall-d59k;;
        nuecc48) echo work-nuecc48-nuf;;
        r1qlmc)  echo work-r1ql-first10;;
        r2mc)    echo work-r2patrec-f1;;
    esac
}
nusel_root() {
    case "$1" in
        mcp1k)   echo work-mcp1kall-vf$TAG;;
        nuecc48) echo work-nuecc48-vf$TAG;;
        r1qlmc)  echo work-r1qlmc-vf$TAG;;
        r2mc)    echo work-r2mc-vf$TAG;;
    esac
}

refuse_existing() {
    [ -e "$1" ] && { echo "REFUSE: $1 already exists (M13: new run => new tag)"; exit 1; }
}

fail=0

# ---------- optional nusel stage (-full) -----------------------------------
run_nusel() {
    s=$1; NR=$(nusel_root "$s")
    refuse_existing "$NR"
    echo "== [$s] nusel stage -> $NR =="
    case "$s" in
    mcp1k)
        TAG=vf$TAG ENTRIES="$(tr '\n' ' ' < "$VF/entries-mcp1k.txt")" \
            ./run_full1k_nusel.sh 1000 "$JOBS"
        rc=$?
        n=$(ls "$NR/.status" 2>/dev/null | wc -l)
        bad=$(grep -L '^rc=0' "$NR"/.status/* 2>/dev/null | wc -l)
        echo "[$s] nusel rc=$rc status-files=$n (expect 572) non-rc0=$bad"
        { [ "$n" -eq 572 ] && [ "$bad" -eq 0 ]; } || fail=1
        ;;
    nuecc48)
        mkdir -p "$NR"
        nev=0
        for d in work-nuecc48-nuf/evt*; do
            b=$(basename "$d")
            ln -sfn "$SBND_DIR/work/$b" "$NR/$b"; nev=$((nev+1))
        done
        echo "[$s] seeded $nev imaging links from work/"
        seq 1 48 | xargs -P "$JOBS" -I{} env SBND_WORK_ROOT="$NR" \
            SBND_INPUT_DIR="$SBND_DIR/input_files_reco1/extracted-2025fall-48evt-fsprod" \
            ./run_nusel_evt.sh data {}
        rc=$?
        n=$(ls -d "$NR"/nusel_evt* 2>/dev/null | wc -l)
        echo "[$s] nusel rc=$rc nusel_evt dirs=$n (expect 48)"
        { [ $rc -eq 0 ] && [ "$n" -eq 48 ]; } || fail=1
        ;;
    r1qlmc)
        mkdir -p "$NR"
        for d in work-r1ql-f1/evt* work-r1ql-f2/evt*; do
            ln -sfn "$SBND_DIR/$(basename "$(dirname "$d")")/$(basename "$d")" "$NR/$(basename "$d")"
        done
        env SBND_WORK_ROOT="$NR" SBND_INPUT_DIR="$SBND_DIR/input_files_reco1/extracted-r1ql-f1" \
            ./run_nusel_evt.sh mc -stm-fit all || fail=1
        for i in 1 2; do
            env SBND_WORK_ROOT="$NR" SBND_INPUT_DIR="$SBND_DIR/input_files_reco1/extracted-r1ql-f2" \
                ./run_nusel_evt.sh mc -stm-fit $i || fail=1
        done
        n=$(ls -d "$NR"/nusel_evt* 2>/dev/null | wc -l)
        echo "[$s] nusel nusel_evt dirs=$n (expect 10)"
        [ "$n" -eq 10 ] || fail=1
        ;;
    r2mc)
        mkdir -p "$NR"
        for d in work-r2patrec-f1/evt*; do
            ln -sfn "$SBND_DIR/work-r2patrec-f1/$(basename "$d")" "$NR/$(basename "$d")"
        done
        env SBND_WORK_ROOT="$NR" SBND_INPUT_DIR="$SBND_DIR/input_files_reco1/extracted-r2patrec-f1" \
            ./run_nusel_evt.sh mc -stm-fit all || fail=1
        n=$(ls -d "$NR"/nusel_evt* 2>/dev/null | wc -l)
        echo "[$s] nusel nusel_evt dirs=$n (expect 13)"
        [ "$n" -eq 13 ] || fail=1
        ;;
    esac
}

# ---------- PR chain over the subset ---------------------------------------
run_pr() {
    s=$1
    if [ "$FULL" = yes ]; then QL=$(nusel_root "$s"); else QL=$(pinned_qlroot "$s"); fi
    OUT=work-vf$s-$TAG
    refuse_existing "$OUT"
    [ -d "$QL" ] || { echo "REFUSE: ql_root $QL missing"; fail=1; return; }
    echo "== [$s] PR chain: ql_root=$QL -> $OUT =="
    # PR-tail mode runs the PINNED (pre-pr/20, no-wasmain) hubs on purpose --
    # declare that to the legacy-tree guard (doc pr/23 sec 4.2) or the job
    # aborts.  -full mode regenerates fresh trees and inherits cfg TRUE.
    REQWM=()
    [ "$FULL" = yes ] || REQWM=(SBND_REQUIRE_WASMAIN=0)
    [ "$FULL" = yes ] || echo "[$s] NOTE: pinned legacy hub (P2-inert), SBND_REQUIRE_WASMAIN=0 declared"
    # shellcheck disable=SC2046
    env "${REQWM[@]}" PR_JOBS=$JOBS ./run_pr_chain_batch.sh "$QL" "$OUT" "$(reality "$s")" \
        $(tr '\n' ' ' < "$VF/events-$s.txt")
    rc=$?
    nexp=$(wc -l < "$VF/events-$s.txt")
    ndone=$(ls -d "$OUT"/pr_evt* 2>/dev/null | wc -l)
    nbad=$(grep -L '^rc=0$' "$OUT"/pr_evt*/rc.txt 2>/dev/null | wc -l)
    echo "[$s] PR rc=$rc pr_evt dirs=$ndone/$nexp non-rc0=$nbad"
    { [ $rc -eq 0 ] && [ "$ndone" -eq "$nexp" ] && [ "$nbad" -eq 0 ]; } || fail=1
    python3 pr_scores_table.py --root "$OUT" --sample "$s" --out "$OUT/vf-scores.tsv" \
        || { echo "[$s] pr_scores_table FAILED"; fail=1; }
}

for s in $SAMPLES; do
    [ "$FULL" = yes ] && run_nusel "$s"
    run_pr "$s"
done

echo
echo "valfast tag=$TAG full=$FULL samples=[$SAMPLES] overall_fail=$fail"
echo "load: $(cut -d' ' -f1-3 /proc/loadavg)"
exit $fail
