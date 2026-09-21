#!/bin/bash
# doc sbnd_xin/119 rounds 1-2: run stage B at the production operating point with ONE lever
# changed, so each lever can be gated and costed on its own.
#
# Fork by duplication (M10) of scripts/d118/stageB_flip.sh, which stays untouched.  What differs
# is only the lever selection and the arm name; the invocation is otherwise byte-for-byte doc
# 118's flipped-production invocation, which is the point -- the baseline these arms are compared
# against must be what production actually runs today.
#
#   LEVER=ctl   production exactly as it runs today (glibc, no proj_pad).  THE CONTROL.
#               Run it fresh rather than reusing work-r3*-d118flip, because that arm was produced
#               at PR_JOBS=10 under a different machine load: wall_s from it is not comparable.
#               The in-job TICK ladder survives contention, but the control costs 2 minutes and
#               removes the argument entirely.
#   LEVER=tcm   round 1 -- SBND_PR_TCMALLOC=1, i.e. libtcmalloc_minimal JOINED to the PR chain's
#               LD_PRELOAD.  Allocator only; no reconstruction input changes.
#   LEVER=pad   round 2 -- SBND_TRACKFIT_JSON points at docs/119_figs/119_tf_sbnd_pad.json, the
#               production fit JSON plus proj_pad_wire 3 / proj_pad_time 3.  Read at RUNTIME, so
#               no compiled config moves and no jsonnet is touched for the measurement.
#
# ctl and pad both run with SBND_PR_TCMALLOC=0 so that round 2's cost delta is the pad alone and
# not the pad plus the allocator.  Round 1 and round 2 are deliberately NOT measured together.
#
#   LEVER=ctl2  the null pair -- ctl run a second time, to size run-to-run variation.
#
#   LEVER=r3    round 3 -- the projection-constant hoist and the BFS container change, i.e. the
#               SAME configuration as `flip` run against a REBUILT libWireCellClus.  This is the
#               first d119 lever where the arms differ by the BINARY and not by the environment,
#               which makes M1 (the stale-library trap) the failure mode that would silently void
#               the round: an r3 arm run against the round-0..2 pin compares the old binary to
#               itself and PASSes vacuously.  So r3/r3b default to their OWN pin and the guard
#               below REFUSES the wrong one in either direction.
#   LEVER=r3b   the null pair for round 3 -- r3 run a second time, so the cost delta has a noise
#               floor measured on the NEW binary rather than inherited from the old one.
#
# Usage: LEVER=<ctl|ctl2|tcm|pad|flip|r3|r3b> [JOBS=n] [OFF_N=k] stageB_lever.sh <cv|nuecc|off> [f000 ...]
set -u
cd -P "$(dirname "$0")/../.." || exit 1
SX=$PWD
S=${1:?usage: LEVER=<ctl|ctl2|tcm|pad> stageB_lever.sh <cv|nuecc|off> [f000 ...]}
shift
LEVER=${LEVER:?set LEVER to ctl, ctl2, tcm or pad}
J=${JOBS:-8}
# doc sbnd_xin/119 round 3: rounds 0-2 all ran on ONE binary, so one pin was enough.  Round 3
# changes C++, so the pin is part of the lever and picking it by hand is exactly how M1 fires.
PRE_R3_CLUS_MD5=71ebd5aeb3868cbc0803f2fa2654b46b   # the round-0..2 binary, toolkit d7d4da83
case "$LEVER" in
    r3|r3b) export LIBSNAP=${LIBSNAP:-$HOME/tmp/d119r3-libpin} ;;
    *)      export LIBSNAP=${LIBSNAP:-$HOME/tmp/d119-libpin} ;;
esac
export LD_LIBRARY_PATH=$LIBSNAP:${LD_LIBRARY_PATH:-}
[ -e "$LIBSNAP/libWireCellClus.so" ] || { echo "ERROR: no pin at $LIBSNAP" >&2; exit 1; }
CLUS_MD5=$(md5sum "$LIBSNAP/libWireCellClus.so" | cut -d' ' -f1)
case "$LEVER" in
    # The round-3 arms MUST NOT be the pre-round-3 binary -- that is the vacuous PASS.
    r3|r3b) [ "$CLUS_MD5" != "$PRE_R3_CLUS_MD5" ] \
                || { echo "REFUSING: LEVER=$LEVER on the PRE-round-3 clus ($CLUS_MD5) -- this would" >&2
                     echo "          compare the old binary against itself and PASS for nothing (M1)." >&2; exit 2; } ;;
    # ...and the round-0..2 arms MUST be it, or re-running one of them would silently re-measure
    # rounds 1-2 on a binary they were never measured on.
    *)      [ "$CLUS_MD5"  = "$PRE_R3_CLUS_MD5" ] \
                || { echo "REFUSING: LEVER=$LEVER on clus $CLUS_MD5, but rounds 0-2 were measured on" >&2
                     echo "          $PRE_R3_CLUS_MD5.  Use LEVER=r3 for the new binary." >&2; exit 2; } ;;
esac
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES:-pr_display}
# Same as d118: production supplies the trajectory from jsonnet + the in-tree JSON.
unset PR_EXTRA_TLA SBND_TRACKFIT_JSON PR_GROUP_SIZE SBND_NO_DL PR_CFG_TREE

case "$LEVER" in
    # ctl2 is the NULL PAIR: byte-identically the ctl configuration, run a second time.  Without
    # it a per-stage cost delta cannot be told from run-to-run variation -- round 2's first reading
    # showed a uniform +2 % spread across stages proj_pad cannot touch (the BDT scorers,
    # CreateSteinerGraph), which is the signature of noise, not of a knob.
    ctl|ctl2) export SBND_PR_TCMALLOC=0 ;;
    tcm) export SBND_PR_TCMALLOC=1 ;;
    # round 2's FLIP arm: no SBND_TRACKFIT_JSON at all, so the job reads the in-tree
    # cfg/pgrapher/experiment/sbnd/sbnd_track_fitting.json -- i.e. production after the flip.
    # Gated against the `pad` arm to prove the flipped DEFAULT reproduces the MEASURED arm; a key
    # list alone (scripts/d119/tf_key_gate.py) does not prove the path that reads it.
    # tcmalloc stays OFF here so this arm differs from `pad` in nothing but where the knobs came
    # from -- the same discipline doc 118's stageB_flip.sh used.
    flip) export SBND_PR_TCMALLOC=0 ;;
    # Round 3's arms are `flip`'s configuration on a rebuilt binary, so they must carry `flip`'s
    # ENVIRONMENT too -- tcmalloc OFF.  Turning the round-1 allocator on here would mix a 10-16 %
    # allocator effect into a CPU measurement and make the round unreadable.
    r3|r3b) export SBND_PR_TCMALLOC=0 ;;
    pad) export SBND_PR_TCMALLOC=0
         export SBND_TRACKFIT_JSON="$SX/docs/119_figs/119_tf_sbnd_pad.json"
         [ -s "$SBND_TRACKFIT_JSON" ] || { echo "ERROR: no padded fit JSON at $SBND_TRACKFIT_JSON" >&2; exit 1; } ;;
    *) echo "unknown LEVER=$LEVER (ctl|ctl2|tcm|pad|flip|r3|r3b)" >&2; exit 2 ;;
esac

case "$S" in
    cv)    A=$SX/work-r3cv-d115;  B=$SX/work-r3cv-d119$LEVER;  REALITY=sim ;;
    nuecc) A=$SX/work-r3nue-d115; B=$SX/work-r3nue-d119$LEVER; REALITY=sim ;;
    off)   A=$SX/work-r3off-d115; B=$SX/work-r3off-d119$LEVER; REALITY=data ;;
    *) echo "unknown sample: $S" >&2; exit 2 ;;
esac
[ -d "$A" ] || { echo "ERROR: no stage-A arm $A" >&2; exit 1; }

# M13: never write into an arm this script did not create.
if [ -e "$B" ]; then
    [ "$(cat "$B/.d119_arm" 2>/dev/null)" = "$LEVER" ] \
        || { echo "REFUSING: $B exists and is not a d119 $LEVER arm (M13)" >&2; exit 2; }
    echo "=== resuming existing $B"
else
    mkdir -p "$B" && echo "$LEVER" > "$B/.d119_arm"
fi
LOGD=$HOME/tmp/d119-stageB-$S-$LEVER; mkdir -p "$LOGD"

echo "=== d119 stage B: sample=$S lever=$LEVER reality=$REALITY PR_JOBS=$J  $(date +%F_%H:%M:%S)"
echo "=== pin: $LIBSNAP  clus md5 $(md5sum "$LIBSNAP/libWireCellClus.so" | cut -c1-12)"
echo "=== toolkit HEAD: $(git -C /nfs/data/1/xqian/toolkit-dev/toolkit rev-parse HEAD)"
echo "=== SBND_PR_TCMALLOC=$SBND_PR_TCMALLOC  SBND_TRACKFIT_JSON=${SBND_TRACKFIT_JSON:-<none>}"
t0=$(date +%s)

# The d118 gate arms are the event set: 18 cv + 24 nuecc + 20 off = the 62 events doc 118 gated.
# Taking the SAME events keeps every d119 claim comparable to the flip's own gate.
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
    # Exactly the events the d118 arm ran in this sub-root -- not the whole stage-A population.
    mapfile -t EVTS < <(ls -d "$D118${sub:+/$sub}"/pr_evt* 2>/dev/null \
                        | sed -E 's#.*/pr_evt([0-9]+)/?$#\1#' | sort -n)
    [ "${#EVTS[@]}" -gt 0 ] || { echo "[${sub:-.}] ERROR: no events in $D118${sub:+/$sub}" >&2; continue; }
    PR_JOBS=$J "$SX/run_pr_chain_batch.sh" "$qa" "$qb" "$REALITY" "${EVTS[@]}" \
        > "$LOGD/${sub:-all}.log" 2>&1
    rc=$?
    echo "[${sub:-.}] rc=$rc pr_evt=$(ls -d "$qb"/pr_evt* 2>/dev/null | wc -l)/${#EVTS[@]}"
done

t1=$(date +%s)
echo "=== pin end: $(md5sum "$LIBSNAP/libWireCellClus.so" | cut -c1-12)"
echo "=== finished in $((t1-t0)) s"
echo "=== pr_evt dirs : $(ls -d "$B"/*/pr_evt* "$B"/pr_evt* 2>/dev/null | wc -l)"
echo "=== rc!=0 events: $(grep -L 'rc=0' "$B"/*/pr_evt*/rc.txt "$B"/pr_evt*/rc.txt 2>/dev/null | wc -l)"
exit 0
