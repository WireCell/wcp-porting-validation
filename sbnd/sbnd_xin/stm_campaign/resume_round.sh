#!/usr/bin/env bash
# Doc 63: resume an interrupted campaign arm -- re-run only the events of the
# baseline/full list that do not yet have rc=0 in <root>/.status, at the given
# parallelism.  Partial outputs of unfinished events are removed first (the
# work-stmcamp roots are campaign scratch, not scientific record).
# Usage: resume_round.sh <round> <events-file> <njobs> [extra flags...]
set -u
cd "$(dirname "$0")/.."
SBND_DIR=$PWD
ROUND=${1:?}; EVFILE=${2:?}; NJOBS=${3:?}; shift 3
ROOT=$SBND_DIR/work-stmcamp-$ROUND
D59K=$SBND_DIR/work-mcp1kall-d59k
STAGE=$SBND_DIR/input_files_reco1/staged-mcp2025c-1000evt
MAP=$STAGE/entry_event_map.tsv
NUF="-chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair-real -fvx 2.5 -fvy 3 -stm-fit -mip 56000 -unmerge-assoc $*"
[ -d "$ROOT" ] || { echo "no root $ROOT"; exit 1; }
mkdir -p "$ROOT/.status"

todo=""
while read -r evt; do
    [ -n "$evt" ] || continue
    if [ -f "$ROOT/.status/$evt" ] && grep -q '^rc=0' "$ROOT/.status/$evt"; then continue; fi
    todo="$todo $evt"
    rm -rf "$ROOT/nusel_evt$evt" "$ROOT/.status/$evt"
done < "$EVFILE"

echo "resume $ROUND: $(echo "$todo" | wc -w) events to (re)run at $NJOBS-way, flags: $NUF"
entry_of() { awk -F'\t' -v e="$1" '$4==e {print $1; exit}' "$MAP"; }
run_one() {
    local evt=$1
    local entry; entry=$(entry_of "$evt")
    [ -n "$entry" ] || { echo "rc=90 evt=$evt no-entry" > "$ROOT/.status/$evt"; return; }
    ln -sfn "$(readlink -f "$D59K/evt$evt")" "$ROOT/evt$evt"
    ln -sfn "$D59K/ql_evt$evt" "$ROOT/ql_evt$evt"
    local cwd=$ROOT/.cwd/$evt
    mkdir -p "$cwd"
    (
        cd "$cwd" || exit 92
        SBND_INPUT_DIR=$STAGE/e$entry SBND_WORK_ROOT=$ROOT \
            setarch x86_64 -R "$SBND_DIR/run_nusel_evt.sh" data $NUF 1
    ) > "$ROOT/.log_$evt.log" 2>&1
    echo "rc=$? evt=$evt" > "$ROOT/.status/$evt"
}
n=0
for evt in $todo; do
    run_one "$evt" &
    n=$((n+1))
    if [ $((n % NJOBS)) -eq 0 ]; then wait; fi
done
wait
ok=$(grep -l '^rc=0' "$ROOT"/.status/* 2>/dev/null | wc -l)
echo "resume $ROUND done: $ok/$(ls "$ROOT"/.status/ | wc -l) rc=0"
