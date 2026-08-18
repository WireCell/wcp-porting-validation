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
#   sample    any of: mcp1k nuecc48 r1qlmc r2mc ncpi0  (default: all five)
#
# PR-tail mode (default) -- ql_root is a pinned KEEP hub, both arms of an A/B
# share it byte-for-byte, so archive diffs are attributable to the change:
#   mcp1k   ql_root work-mcp1k-cb0805    (data, up to 1000 events)
#   nuecc48 ql_root work-nuecc48-cb0805  (data,   48 events)
#   r1qlmc  ql_root work-r1qlmc-cb0805   (sim,    10 events)
#   r2mc    ql_root work-r2mc-cb0805     (sim,    13 events)
#   ncpi0   ql_root work-ncpi0-cb0805    (data,   19 events)
# CORRECTED 2026-08-05 (doc 71 campaign): the previous pinned hubs
# (work-mcp1kall-d59k, work-nuecc48-nuf, work-r1ql-first10, work-r2patrec-f1)
# were deleted in the 2026-08-05 clean-slate retire round, making PR-tail
# mode mechanically dead (run_valfast.sh's own [ -d "$QL" ] check refused
# every sample). The -cb0805 roots are the fresh campaign production Q/L
# roots -- this is now closer to a rolling "current production" pin than a
# frozen pre-pr/14..pr/20 one; if a future A/B needs a FROZEN pctree
# generation, pin a fresh tag deliberately rather than reusing -cb0805 (which
# may itself be superseded by a later campaign). ncpi0 is a NEW fifth sample
# (doc 71): no pre-campaign pin exists for it, so it has no "legacy hub"
# history the way the other four do.
#
# events-<sample>-cb0805.txt (CORRECTED 2026-08-05) replace the old
# events-<sample>.txt manifests (629 events total, derived from
# docs/pr/11_scores-table.tsv against DELETED arms).  Re-derived from doc 71's
# own campaign via pr_scores_table.py --summary on the fresh -cb0805 roots
# (docs/pr/71_scores-table-cb0805.tsv, nu_evaluated=1): mcp1k 445/1000,
# nuecc48 47/48, r1qlmc 4/10, r2mc 6/13, ncpi0 19/19 -- 521 total.  Old
# manifests are UNTOUCHED (M13): they are a record of the deleted arms.
#
# -full mode nusel roots (imaging is NEVER regenerated -- evt dirs are
# symlinks into the BASE hubs, M11/M13):
#   mcp1k   TAG=vf<tag> ENTRIES=... ./run_full1k_nusel.sh   -> work-mcp1kall-vf<tag>
#   nuecc48 imaging links into work-img-nuecc48/             -> work-nuecc48-vf<tag>
#   r1qlmc  doc 67 recipe (f1 'all' + f2 idx 1 2)            -> work-r1qlmc-vf<tag>
#   r2mc    doc 67 recipe (f1 'all')                         -> work-r2mc-vf<tag>
#   ncpi0   imaging links into work-img-ncpi0/ (all 19)      -> work-ncpi0-vf<tag>
# Per-sample nusel flags follow each sample's reference recipe: run_full1k
# carries -stm-fit internally, nuecc48 runs bare (s4_nuecc48.sh precedent),
# the MC samples pass -stm-fit (doc 67's $NUF); ncpi0 pass -stm-fit like
# nuecc48's data siblings elsewhere in the campaign (doc 71 step 4).
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
        mcp1k|nuecc48|r1qlmc|r2mc|ncpi0) SAMPLES="$SAMPLES $1";;
        *) echo "unknown arg: $1"; exit 1;;
    esac
    shift
done
[ -n "$SAMPLES" ] || SAMPLES="mcp1k nuecc48 r1qlmc r2mc ncpi0"

reality() { case "$1" in mcp1k|nuecc48|ncpi0) echo data;; *) echo sim;; esac; }
# VF_QLROOT_TAG=<t>: PR-tail mode reads the FRESH hubs a previous
# `run_valfast.sh -full <t>` built (work-*-vf<t>) instead of the pinned
# legacy hubs -- so an A/B pair can share one production-operating-point
# Q/L regeneration (doc pr/23 sec 6: regenerating per arm would let Q/L-side
# nondeterminism pollute a PR-stage A/B).  Fresh hubs carry was_main, so the
# SBND_REQUIRE_WASMAIN=0 legacy declaration is NOT applied.
pinned_qlroot() {
    if [ -n "${VF_QLROOT_TAG:-}" ]; then
        case "$1" in
            mcp1k)   echo work-mcp1kall-vf$VF_QLROOT_TAG;;
            nuecc48) echo work-nuecc48-vf$VF_QLROOT_TAG;;
            r1qlmc)  echo work-r1qlmc-vf$VF_QLROOT_TAG;;
            r2mc)    echo work-r2mc-vf$VF_QLROOT_TAG;;
            ncpi0)   echo work-ncpi0-vf$VF_QLROOT_TAG;;
        esac
        return
    fi
    case "$1" in
        mcp1k)   echo work-mcp1k-cb0805;;
        nuecc48) echo work-nuecc48-cb0805;;
        r1qlmc)  echo work-r1qlmc-cb0805;;
        r2mc)    echo work-r2mc-cb0805;;
        ncpi0)   echo work-ncpi0-cb0805;;
    esac
}
nusel_root() {
    # ABSOLUTE on purpose (docs/73 sec 12, first -full exercise of the four
    # small samples): run_ql_evt.sh's opflash split does
    # `( cd "$stage" && tar czf "$QLDIR/opflash_apa<N>.tar.gz" ... )`, and a
    # relative SBND_WORK_ROOT makes that czf target unresolvable from inside
    # the stage dir -- every Q/L step then fails with "Cannot open".  mcp1k
    # never hit this because run_full1k_nusel.sh absolutizes its own root.
    case "$1" in
        mcp1k)   echo "$SBND_DIR/work-mcp1kall-vf$TAG";;
        nuecc48) echo "$SBND_DIR/work-nuecc48-vf$TAG";;
        r1qlmc)  echo "$SBND_DIR/work-r1qlmc-vf$TAG";;
        r2mc)    echo "$SBND_DIR/work-r2mc-vf$TAG";;
        ncpi0)   echo "$SBND_DIR/work-ncpi0-vf$TAG";;
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
        # entries-mcp1k-cb0805.txt (CORRECTED 2026-08-05): 445 art entries,
        # re-derived from events-mcp1k-cb0805.txt against entry_event_map.tsv
        # -- the old entries-mcp1k.txt (572) mapped the deleted d59k manifest.
        TAG=vf$TAG ENTRIES="$(tr '\n' ' ' < "$VF/entries-mcp1k-cb0805.txt")" \
            IMGBASE=$SBND_DIR/work-img-mcp1k \
            ./run_full1k_nusel.sh 1000 "$JOBS"
        rc=$?
        n=$(ls "$NR/.status" 2>/dev/null | wc -l)
        bad=$(grep -L '^rc=0' "$NR"/.status/* 2>/dev/null | wc -l)
        echo "[$s] nusel rc=$rc status-files=$n (expect 445) non-rc0=$bad"
        { [ "$n" -eq 445 ] && [ "$bad" -eq 0 ]; } || fail=1
        ;;
    nuecc48)
        mkdir -p "$NR"
        nev=0
        # CORRECTED 2026-08-05: the event-id source used to be the pinned
        # QL hub work-nuecc48-nuf (deleted); the imaging source used to be
        # the permanent hub work/ (also deleted).  Both roles are now
        # work-img-nuecc48/ -- it both lists the 48 event ids (via its own
        # evt* dirs) and is the imaging source to link from.
        for d in work-img-nuecc48/evt*; do
            b=$(basename "$d")
            ln -sfn "$SBND_DIR/work-img-nuecc48/$b" "$NR/$b"; nev=$((nev+1))
        done
        echo "[$s] seeded $nev imaging links from work-img-nuecc48/"
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
        # CORRECTED 2026-08-05: work-r1ql-f1/f2 (per-extraction imaging roots)
        # were deleted; the campaign's work-img-r1qlmc/ holds both f1's full
        # 8 events and f2's idx 1,2 (evts 5,12) merged into one root (doc 71
        # step 3 -- imaged directly there rather than symlinked in, to avoid
        # a second symlink hop).
        for d in work-img-r1qlmc/evt*; do
            ln -sfn "$SBND_DIR/work-img-r1qlmc/$(basename "$d")" "$NR/$(basename "$d")"
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
        # CORRECTED 2026-08-05: work-r2patrec-f1 (imaging root) was deleted;
        # source is now the campaign's work-img-r2mc/.
        for d in work-img-r2mc/evt*; do
            ln -sfn "$SBND_DIR/work-img-r2mc/$(basename "$d")" "$NR/$(basename "$d")"
        done
        env SBND_WORK_ROOT="$NR" SBND_INPUT_DIR="$SBND_DIR/input_files_reco1/extracted-r2patrec-f1" \
            ./run_nusel_evt.sh mc -stm-fit all || fail=1
        n=$(ls -d "$NR"/nusel_evt* 2>/dev/null | wc -l)
        echo "[$s] nusel nusel_evt dirs=$n (expect 13)"
        [ "$n" -eq 13 ] || fail=1
        ;;
    ncpi0)
        mkdir -p "$NR"
        nev=0
        for d in work-img-ncpi0/evt*; do
            b=$(basename "$d")
            ln -sfn "$SBND_DIR/work-img-ncpi0/$b" "$NR/$b"; nev=$((nev+1))
        done
        echo "[$s] seeded $nev imaging links from work-img-ncpi0/"
        env SBND_WORK_ROOT="$NR" SBND_INPUT_DIR="$SBND_DIR/input_files_reco1/extracted-ncpi0" \
            ./run_nusel_evt.sh data -stm-fit all || fail=1
        n=$(ls -d "$NR"/nusel_evt* 2>/dev/null | wc -l)
        echo "[$s] nusel nusel_evt dirs=$n (expect 19)"
        [ "$n" -eq 19 ] || fail=1
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
    if [ "$FULL" != yes ] && [ -z "${VF_QLROOT_TAG:-}" ]; then
        REQWM=(SBND_REQUIRE_WASMAIN=0)
        echo "[$s] NOTE: pinned legacy hub (P2-inert), SBND_REQUIRE_WASMAIN=0 declared"
    elif [ -n "${VF_QLROOT_TAG:-}" ]; then
        echo "[$s] NOTE: fresh hub override VF_QLROOT_TAG=$VF_QLROOT_TAG (was_main present, guard active)"
    fi
    # events-<sample>-cb0805.txt (CORRECTED 2026-08-05): re-derived manifest,
    # see header comment. The old events-<sample>.txt files are untouched
    # (M13) but no longer match any surviving root.
    # shellcheck disable=SC2046
    env "${REQWM[@]}" PR_JOBS=$JOBS ./run_pr_chain_batch.sh "$QL" "$OUT" "$(reality "$s")" \
        $(tr '\n' ' ' < "$VF/events-$s-cb0805.txt")
    rc=$?
    nexp=$(wc -l < "$VF/events-$s-cb0805.txt")
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
