#!/usr/bin/env bash
# Doc 63: run one STM-campaign evaluation arm over the 72 doc-62 baseline
# events (or a subset), with the d59k production flag set + optional round
# knobs, into a FRESH work root work-stmcamp-<round>/.
#
# Imaging and the Q/L pctree are symlinked per event from work-mcp1kall-d59k
# (read-only, M13); only the PR tagger tail re-executes (seconds per event).
#
# Usage: ./run_round.sh <round> [extra run_nusel_evt.sh flags...]
#   round   label, e.g. r0, r1 -> work root work-stmcamp-<round>
# Env:
#   STM_EVENTS   space-separated event list override (default: all baseline)
#   NJOBS        parallel jobs (default 6, CLAUDE.md cap)
# A work root that already exists is REFUSED (new arm => new root; M13).
set -u
cd "$(dirname "$0")/.."
SBND_DIR=$PWD
ROUND=${1:?usage: run_round.sh <round> [flags...]}; shift
ROOT=$SBND_DIR/work-stmcamp-$ROUND
D59K=$SBND_DIR/work-mcp1kall-d59k
STAGE=$SBND_DIR/input_files_reco1/staged-mcp2025c-1000evt
MAP=$STAGE/entry_event_map.tsv
NJOBS=${NJOBS:-6}

# The d59k production flag set (run_full1k_nusel.sh NUF verbatim) + round knobs.
NUF="-chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair-real -fvx 2.5 -fvy 3 -stm-fit -mip 56000 -unmerge-assoc"
NUF="$NUF $*"

if [ -e "$ROOT" ]; then
    echo "REFUSING: $ROOT exists (new arm => new round label)" >&2
    exit 1
fi
mkdir -p "$ROOT/.status"

EVENTS=${STM_EVENTS:-$(grep -v '^#' "$SBND_DIR/scan-d59k/stm-baseline.tsv" | awk -F'\t' 'NR>1{print $2}' | sort -un)}

# event -> art entry (col 1 keyed by col 4)
entry_of() { awk -F'\t' -v e="$1" '$4==e {print $1; exit}' "$MAP"; }

run_one() {
    local evt=$1
    local entry; entry=$(entry_of "$evt")
    if [ -z "$entry" ]; then echo "rc=90 evt=$evt no-entry" > "$ROOT/.status/$evt"; return; fi
    if [ ! -d "$D59K/evt$evt" ] || [ ! -d "$D59K/ql_evt$evt" ]; then
        echo "rc=91 evt=$evt missing-d59k-input" > "$ROOT/.status/$evt"; return
    fi
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

echo "round $ROUND: $(echo "$EVENTS" | wc -w) events, flags: $NUF"
n=0
for evt in $EVENTS; do
    run_one "$evt" &
    n=$((n+1))
    if [ $((n % NJOBS)) -eq 0 ]; then wait; fi
done
wait

ok=$(grep -l '^rc=0' "$ROOT"/.status/* 2>/dev/null | wc -l)
tot=$(ls "$ROOT"/.status/ | wc -l)
echo "round $ROUND done: $ok/$tot rc=0"
grep -L '^rc=0' "$ROOT"/.status/* 2>/dev/null | while read -r f; do cat "$f"; done
