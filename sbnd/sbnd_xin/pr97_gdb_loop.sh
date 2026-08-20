#!/usr/bin/env bash
# doc pr/97 sec.5 -- name the line behind the SBND 18255-178410 SIGSEGV.
#
# The crash is intermittent at ~4 % per run (4 in 108 runs, doc pr/97), and
# kernel core dumps cannot be captured here (/proc/sys/kernel/core_pattern is
# the bare "core", so a 2.3 GB core lands in the dying process's CWD).  So we
# just run the Q/L job over and over under gdb -batch and keep the backtrace of
# whichever run dies: run_ql_evt_pr97gdb.sh is run_ql_evt.sh with the wire-cell
# step wrapped in gdb (M10 -- the production runner is byte-untouched).
#
# This is the run_ql_batch.sh worker for one event, inlined: the batch driver
# hardcodes run_ql_evt.sh, and forking it too would duplicate more than this.
#
# NOTE the two detectors, which do NOT agree: under gdb a caught SIGSEGV leaves
# the runner with rc=0 (gdb reaps the inferior), so a crash in THIS arm is found
# by grepping "received signal SIG", never by rc=139.  Arms that run without gdb
# are the opposite: rc=139 and no such line.  Tally each arm with its own.
#
# Usage:
#   ./pr97_gdb_loop.sh <root-prefix> <n-runs> [jobs]
#   e.g. ./pr97_gdb_loop.sh work-pr97g-r 40 4
# Every run gets a FRESH root <prefix><n> and refuses to reuse one (M13).
# Concurrency is capped (M5); the crash is not concurrency-dependent (doc pr/97),
# so running 4 at a time only buys wall-clock.
set -u
cd "$(dirname "$0")"
SBND_DIR=$PWD
PREFIX=${1:?usage: pr97_gdb_loop.sh <root-prefix> <n-runs> [jobs]}
N=${2:?}
J=${3:-4}

EVT=178410
ENTRY=1774
STAGE=$SBND_DIR/input_files_reco1/staged-mcp2025c-2nd-2000evt/e$ENTRY
IMG=$SBND_DIR/work-img-mcp2k/evt$EVT
TC=$(cd ../../abtest && pwd)/timecmd.py
[ -d "$IMG" ]   || { echo "REFUSE: no imaging $IMG" >&2; exit 2; }
[ -d "$STAGE" ] || { echo "REFUSE: no stage $STAGE" >&2; exit 2; }

one() {
    local n=$1 R="$SBND_DIR/${PREFIX}$1"
    if [ -e "$R" ]; then echo "r=$n SKIP exists (M13)"; return 0; fi
    mkdir -p "$R/.status" "$R/.cwd"
    ln -sfn "$IMG" "$R/evt$EVT"
    ( cd "$R/.cwd" || exit 92
      SBND_INPUT_DIR=$STAGE SBND_WORK_ROOT=$R \
          setarch x86_64 -R python3 "$TC" "$R/.time.meta" \
          "$SBND_DIR/run_ql_evt_pr97gdb.sh" data 1
    ) > "$R/.log.log" 2>&1
    local rc=$?
    printf 'rc=%s evt=%s %s\n' "$rc" "$EVT" "$(tr '\n' ' ' < "$R/.time.meta" 2>/dev/null)" \
        > "$R/.status/s"
    if grep -q "received signal SIG" "$R/.log.log"; then
        echo "r=$n rc=$rc **BACKTRACE** $R/.log.log"
    else
        echo "r=$n rc=$rc $(cat "$R/.status/s")"
    fi
}
export -f one 2>/dev/null || true

echo "=== pr97 gdb loop: prefix=$PREFIX runs=$N jobs=$J ==="
echo "  toolkit HEAD=$(git -C "$SBND_DIR/../../../toolkit" rev-parse --short HEAD)"
echo "  libWireCellClus.so mtime=$(stat -c %y /nfs/data/1/xqian/toolkit-dev/local/lib/libWireCellClus.so)"
echo "  started $(date -Is)"
i=1
while [ "$i" -le "$N" ]; do
    running=0
    while [ "$running" -lt "$J" ] && [ "$i" -le "$N" ]; do
        one "$i" & i=$((i+1)); running=$((running+1))
    done
    wait
    # stop as soon as any run has produced a stack
    if grep -l "received signal SIG" "$SBND_DIR/${PREFIX}"*/.log.log 2>/dev/null | head -1 | grep -q .; then
        echo "  HIT -- stopping the loop"
        break
    fi
done
echo "  finished $(date -Is)"
grep -c . /dev/null >/dev/null
for d in "$SBND_DIR/${PREFIX}"*; do
    [ -d "$d" ] || continue
    printf '%s %s\n' "$(basename "$d")" "$(cat "$d"/.status/s 2>/dev/null)"
done
echo "GDBLOOPDONE"
