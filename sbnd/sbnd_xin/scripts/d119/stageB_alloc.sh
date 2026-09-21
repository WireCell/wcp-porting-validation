#!/bin/bash
# doc sbnd_xin/119 round 5 (the allocator round): measure FOUR allocator arms through the SAME
# invocation, on the SAME events, read by the SAME instrument.
#
# Fork by duplication (M10) of scripts/d119/stageB_tail.sh, which stays untouched as round 4's
# record; the pin guard, the 7-event tail list and the M13 arm guard are carried across verbatim.
# What is new is the ALLOC set, the cv/nuecc/off samples (so round 5 can run the 62-event byte
# gate through one script), and the reason the script exists at all:
#
#   WHY THIS IS NOT A ONE-LINER.  Round 4 sec 10.4 reported jemalloc peaking at 1.343 GiB on the
#   tail event where production tcmalloc peaked at 2.084.  Those two numbers came from DIFFERENT
#   INSTRUMENTS: 1.343 is the in-job MEM ladder (VmRSS sampled at stage boundaries) taken under
#   scripts/perf/profile_pr119r4.sh with heap SAMPLING ACTIVE, and 2.084 is .time.meta maxrss_kb
#   (getrusage CHILDREN high-water) from a full run_pr_chain_batch.sh run.  On these seven events
#   getrusage reads 0.15-0.19 GiB above the ladder, and the sampler perturbs allocation.  Read
#   ladder against ladder the p50 comparison actually REVERSES sign (jemalloc 1.065 vs tcmalloc
#   1.045).  So the round-4 number is not evidence for an allocator flip, and this script exists
#   to replace it with four arms that differ in the allocator and in nothing else.
#
# ALLOC:
#   tcm    TODAY'S PRODUCTION.  SBND_PR_TCMALLOC and SBND_TCMALLOC_LIB both unset, so the runner's
#          own default applies -- which is ON, libtcmalloc_minimal (round 1, doc 119 sec 5).
#   glibc  SBND_PR_TCMALLOC=0, the pre-round-1 process environment.  The control doc 116's tail
#          numbers were measured under.
#   jem    jemalloc 5.3.0 via the runner's OWN env hook (SBND_TCMALLOC_LIB is `${...:-default}`),
#          so NO runner edit is needed to measure it.  MALLOC_CONF is explicitly unset: round 4's
#          jemalloc runs had prof:true and a sampling interval, and that is exactly the
#          perturbation this round removes.
#   rel    production tcmalloc + TCMALLOC_RELEASE_RATE=10.  THE MECHANISM CONTROL, and the reason
#          it is not optional: jemalloc 5.x returns dirty pages on a decay timer
#          (opt.dirty_decay_ms, 10 s by default) while tcmalloc's release rate is 1 by default and
#          it holds freed spans in its page heap.  If that is what the tail gap is, the finding is
#          "page-return policy", not "jemalloc", and production already has the lever -- one env
#          var, no flip, no byte gate.  An arm that cannot tell those two apart is not worth
#          running.
#
# Samples:
#   tail            the 7 doc-116 peak-RSS events (round 4's set), arms work-r3nue-d119t<tag>
#   cv|nuecc|off    the 62-event d118 gate manifest, arms work-r3<s>-d119<tag>, for the BYTE GATE
#                   (scripts/d119/lever_gate.py --vs r3 <tag>) and a like-for-like TICK ladder.
#
# The arms are meant to be run BACK TO BACK on an idle box at one fan-out.  A CPU number compared
# across arms run under different machine load is not a measurement (doc 119 sec 6, the
# d119ctl2contended arm is the record of that mistake).
#
# Usage: ALLOC=<tcm|glibc|jem|rel> [JOBS=n] stageB_alloc.sh <tail|cv|nuecc|off> [sub ...]
set -u
cd -P "$(dirname "$0")/../.." || exit 1
SX=$PWD
S=${1:?usage: ALLOC=<tcm|glibc|jem|rel> stageB_alloc.sh <tail|cv|nuecc|off> [sub ...]}
shift

R3_CLUS_MD5=4ff75274e43e766eaec4eef8b6107ae7   # round 3, toolkit c590ae36 == 253f1845 (comment-only)
export LIBSNAP=${LIBSNAP:-$HOME/tmp/d119r3-libpin}
export LD_LIBRARY_PATH=$LIBSNAP:${LD_LIBRARY_PATH:-}
[ -e "$LIBSNAP/libWireCellClus.so" ] || { echo "ERROR: no pin at $LIBSNAP" >&2; exit 1; }
CLUS_MD5=$(md5sum "$LIBSNAP/libWireCellClus.so" | cut -d' ' -f1)
[ "$CLUS_MD5" = "$R3_CLUS_MD5" ] \
    || { echo "REFUSING: pin is $CLUS_MD5, round 3 was measured on $R3_CLUS_MD5 (M1)." >&2; exit 2; }

# Production exactly as it runs today: no fit-JSON override, no cell TLAs.
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES:-pr_display}
unset PR_EXTRA_TLA SBND_TRACKFIT_JSON PR_GROUP_SIZE SBND_NO_DL PR_CFG_TREE
# Inherited MALLOC_CONF would silently re-enable round 4's heap sampler in the jem arm and change
# nothing visible in any log.  Clear it for EVERY arm, so the four differ only in the preload.
unset MALLOC_CONF TCMALLOC_RELEASE_RATE

JEMALLOC=${JEMALLOC:-/usr/lib/x86_64-linux-gnu/libjemalloc.so.2}
ALLOC=${ALLOC:?set ALLOC to tcm, glibc, jem or rel}
case "$ALLOC" in
    tcm)   unset SBND_PR_TCMALLOC SBND_TCMALLOC_LIB; TAG=tcm ;;
    glibc) export SBND_PR_TCMALLOC=0;                TAG=glb ;;
    jem)   [ -e "$JEMALLOC" ] || { echo "ERROR: no jemalloc at $JEMALLOC" >&2; exit 1; }
           export SBND_PR_TCMALLOC=1 SBND_TCMALLOC_LIB="$JEMALLOC"; TAG=jem ;;
    rel)   unset SBND_PR_TCMALLOC SBND_TCMALLOC_LIB
           export TCMALLOC_RELEASE_RATE=10;          TAG=rel ;;
    *) echo "unknown ALLOC=$ALLOC (tcm|glibc|jem|rel)" >&2; exit 2 ;;
esac

# The tail, named from work-r3nue-d116tfull/*/.time.meta (getrusage CHILDREN high-water).
# rank  peak RSS   sub / event
#   1    2.208 GiB  f060/11239      <- the doc-116 maximum
#   2    2.203      f049/12202
#   3    2.050      f188/11811
#   4    2.021      f100/5265
#   5    2.006      f196/6248
#   p99  1.602      f075/9393       <- the shoulder
#   p50  1.165      f054/7582       <- the CONTROL: what normal composition looks like
TAILSET=(f060/11239 f049/12202 f188/11811 f100/5265 f196/6248 f075/9393 f054/7582)

# Arm tag for the 62-event samples.  `tcm` alone would be work-r3cv-d119tcm -- which ALREADY
# EXISTS and is ROUND 1's arm, produced on the PRE-ROUND-3 binary.  Reading round 5's jemalloc
# against it would fold round 3's -14.9 % CPU into the allocator delta; the M13 marker guard below
# would have refused the write, but nothing stops a reader.  So round-5 arms carry `r5`.  `jem`
# is the exception and must stay bare: lever_gate.py addresses arms as work-r3<s>-d119<lever>.
case "$TAG" in
    jem) GTAG=jem ;;
    *)   GTAG=r5$TAG ;;
esac
case "$S" in
    # `t` prefix, so a tail arm can never collide with the 62-event nuecc arm of the same
    # allocator -- both live under work-r3nue-* and both use f* sub-roots.
    tail)  A=$SX/work-r3nue-d115; B=$SX/work-r3nue-d119t$TAG; REALITY=sim; J=${JOBS:-7} ;;
    cv)    A=$SX/work-r3cv-d115;  B=$SX/work-r3cv-d119$GTAG;  REALITY=sim; J=${JOBS:-8} ;;
    nuecc) A=$SX/work-r3nue-d115; B=$SX/work-r3nue-d119$GTAG; REALITY=sim; J=${JOBS:-8} ;;
    off)   A=$SX/work-r3off-d115; B=$SX/work-r3off-d119$GTAG; REALITY=data; J=${JOBS:-8} ;;
    *) echo "unknown sample: $S (tail|cv|nuecc|off)" >&2; exit 2 ;;
esac
[ -d "$A" ] || { echo "ERROR: no stage-A arm $A" >&2; exit 1; }

# M13: never write into an arm this script did not create.
MARK="r5-$S-$ALLOC"
if [ -e "$B" ]; then
    [ "$(cat "$B/.d119_arm" 2>/dev/null)" = "$MARK" ] \
        || { echo "REFUSING: $B exists and is not a d119 $MARK arm (M13)" >&2; exit 2; }
    echo "=== resuming existing $B"
else
    mkdir -p "$B" && echo "$MARK" > "$B/.d119_arm"
fi
LOGD=$HOME/tmp/d119-stageB-r5-$S-$TAG; mkdir -p "$LOGD"

echo "=== d119 round 5: sample=$S alloc=$ALLOC  $(date +%F_%H:%M:%S)"
echo "=== pin: $LIBSNAP  clus md5 $CLUS_MD5"
echo "=== toolkit HEAD: $(git -C /nfs/data/1/xqian/toolkit-dev/toolkit rev-parse HEAD)"
echo "=== SBND_PR_TCMALLOC=${SBND_PR_TCMALLOC:-<unset: runner default>}  SBND_TCMALLOC_LIB=${SBND_TCMALLOC_LIB:-<unset: runner default>}"
echo "=== MALLOC_CONF=${MALLOC_CONF:-<unset>}  TCMALLOC_RELEASE_RATE=${TCMALLOC_RELEASE_RATE:-<unset>}"
echo "=== arm: $B  JOBS=$J"
t0=$(date +%s)

if [ "$S" = tail ]; then
    # One event per run_pr_chain_batch.sh call, so its own PR_JOBS cannot parallelise anything;
    # fan out across the SPECS instead.  Identical to round 4's stageB_tail.sh, so the arms are
    # directly comparable with work-r3nue-d119tail{,g}.
    if [ $# -gt 0 ]; then SEL=("$@"); else SEL=("${TAILSET[@]}"); fi
    for spec in "${SEL[@]}"; do
        sub=${spec%%/*}; evt=${spec##*/}
        qa=$A/$sub; qb=$B/$sub
        [ -d "$qa/evt$evt" ] || { echo "[$spec] ERROR: no stage-A input $qa/evt$evt" >&2; continue; }
        mkdir -p "$qb"
        while [ "$(jobs -rp | wc -l)" -ge "$J" ]; do wait -n; done
        ( PR_JOBS=1 "$SX/run_pr_chain_batch.sh" "$qa" "$qb" "$REALITY" "$evt" \
              > "$LOGD/$sub-$evt.log" 2>&1
          echo "$?" > "$LOGD/$sub-$evt.rc" ) &
    done
    wait
    for spec in "${SEL[@]}"; do
        sub=${spec%%/*}; evt=${spec##*/}
        meta="$B/$sub/pr_evt$evt/.time.meta"
        rss=$(tr ' ' '\n' < "$meta" 2>/dev/null | sed -n 's/^maxrss_kb=//p' | head -1)
        echo "[$spec] rc=$(cat "$LOGD/$sub-$evt.rc" 2>/dev/null) peak_rss=$(awk -v k="${rss:-0}" 'BEGIN{printf "%.3f", k/1048576}') GiB"
    done
else
    # Exactly the d118 gate manifest -- the same 62 events every d119 arm was measured on.
    case "$S" in
        cv)    D118=$SX/work-r3cv-d118flip ;;
        nuecc) D118=$SX/work-r3nue-d118flip ;;
        off)   D118=$SX/work-r3off-d118flip ;;
    esac
    [ -d "$D118" ] || { echo "ERROR: no d118 arm $D118 to take the event set from" >&2; exit 1; }
    if [ "$S" = off ]; then
        SUBS=("")
    elif [ $# -gt 0 ]; then
        SUBS=("$@")
    else
        mapfile -t SUBS < <(ls -d "$D118"/f* 2>/dev/null | xargs -n1 basename | sort)
    fi
    echo "=== sub-roots: ${#SUBS[@]}"
    for sub in "${SUBS[@]}"; do
        qa=$A${sub:+/$sub}; qb=$B${sub:+/$sub}
        [ -d "$qa" ] || { echo "[${sub:-.}] ERROR: no stage-A sub-root $qa" >&2; continue; }
        mkdir -p "$qb"
        mapfile -t EVTS < <(ls -d "$D118${sub:+/$sub}"/pr_evt* 2>/dev/null \
                            | xargs -n1 basename | sed 's/^pr_evt//' | sort -n)
        [ "${#EVTS[@]}" -gt 0 ] || { echo "[${sub:-.}] no events in $D118" >&2; continue; }
        echo "[${sub:-.}] ${#EVTS[@]} events"
        PR_JOBS=$J "$SX/run_pr_chain_batch.sh" "$qa" "$qb" "$REALITY" "${EVTS[@]}" \
            > "$LOGD/${sub:-all}.log" 2>&1
        rc=$?; echo "$rc" > "$LOGD/${sub:-all}.rc"
        echo "[${sub:-.}] rc=$rc pr_evt=$(ls -d "$qb"/pr_evt* 2>/dev/null | wc -l)/${#EVTS[@]}"
    done
fi

t1=$(date +%s)
echo "=== pin end: $(md5sum "$LIBSNAP/libWireCellClus.so" | cut -d' ' -f1)"
echo "=== finished in $((t1-t0)) s"
echo "=== pr_evt dirs : $(ls -d "$B"/*/pr_evt* "$B"/pr_evt* 2>/dev/null | wc -l)"
echo "=== rc!=0 events: $(grep -L 'rc=0' "$B"/*/pr_evt*/rc.txt "$B"/pr_evt*/rc.txt 2>/dev/null | wc -l)"
# The allocator each job ACTUALLY loaded, from the runner's round-5 log line -- not inferred from
# the arm's name.  The runner falls back to glibc when the requested library is absent, and until
# round 5 it did so silently; round 4 had to read /proc/<pid>/maps to find this out.
# It lands in the EVENT's own stdout.log, not in this driver's log: run_pr_chain_batch.sh runs
# process_event in a subshell whose stdout is the per-event dir.  That is the better place -- the
# allocator record sits next to the products it produced, permanently, and lever_gate.py's SKIP
# already excludes stdout.log so it is not mistaken for a product.
echo "=== allocator, as logged by the runner:"
grep -h 'alloc: LD_PRELOAD=' "$B"/*/pr_evt*/stdout.log "$B"/pr_evt*/stdout.log 2>/dev/null \
    | sed 's/.*LD_PRELOAD=//' | sort | uniq -c | sed 's/^/    /'
grep -h 'WARNING: .* allocator preload' "$B"/*/pr_evt*/stdout.log "$B"/pr_evt*/stdout.log 2>/dev/null \
    | sort -u | sed 's/^/    /'
exit 0
