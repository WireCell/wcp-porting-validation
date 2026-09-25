#!/bin/bash
# Library A/B arm for the PDHD/PDVD clustering stage WITHOUT touching any
# existing work dir (M13) and WITHOUT installing into the shared local/lib
# (doc pdvd/119: the shared tree carried peers' uncommitted edits and live
# campaigns, so both arms were built privately from one worktree).
#
# For every manifest event: a FRESH tagged work dir
#   <det>/work/<run6>_<evt>_<label>/
# gets symlinks to the production imaging archives of <det>/work/<run6>_<evt>/
# (read-only inputs), then <det>/run_clus_evt.sh -s <label> <run> <evt> runs
# with the arm's install prefix pinned first on PATH and LD_LIBRARY_PATH,
# under setarch -R (M4).  Compare two arms with libab_clus_compare.sh.
#
# Usage: ./libab_clus_run.sh <label> <install-prefix> [manifest]
#   manifest default events.txt (lines "<det> <run> <evt>"; pdhd|pdvd only)
#   CLUS_ARGS  extra run_clus_evt.sh args (default none = the harness's plain call)
#   MAXJ       parallel events (default 3; wire-cell is multi-threaded)
set -u
AB_DIR=$(cd "$(dirname "$0")" && pwd)
BASE_DIR=$(dirname "$AB_DIR")
LABEL=${1:?usage: libab_clus_run.sh <label> <install-prefix> [manifest]}
PREFIX=${2:?usage: libab_clus_run.sh <label> <install-prefix> [manifest]}
MANIFEST=${3:-$AB_DIR/events.txt}
MAXJ=${MAXJ:-3}
[ -x "$PREFIX/bin/wire-cell" ] || { echo "no $PREFIX/bin/wire-cell" >&2; exit 2; }

run_one() {
    local det=$1 run=$2 evt=$3 rp src dst rc
    rp=$(printf '%06d' "$((10#$run))")
    src="$BASE_DIR/$det/work/${rp}_${evt}"
    dst="$BASE_DIR/$det/work/${rp}_${evt}_${LABEL}"
    if [ -e "$dst" ]; then echo "[$det $run $evt] $dst exists -- refusing (M13)" >&2; return 3; fi
    mkdir -p "$dst"
    for f in "$src"/clusters-apa-*.tar.gz "$src"/img-provenance.txt; do
        [ -e "$f" ] && ln -s "$f" "$dst/"
    done
    PATH="$PREFIX/bin:$PATH" LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH" \
        setarch x86_64 -R "$BASE_DIR/$det/run_clus_evt.sh" -s "$LABEL" ${CLUS_ARGS:-} "$run" "$evt" \
        > "$dst/libab_stdout.log" 2>&1
    rc=$?
    echo "[$det $run $evt] rc=$rc -> $dst"
    return $rc
}

fail=0
while read -r det run evt; do
    case "$det" in ''|\#*) continue ;; esac
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n || fail=1; done
    run_one "$det" "$run" "$evt" &
done < "$MANIFEST"
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || fail=1; done
echo "=== libab arm $LABEL ($PREFIX) complete (fail=$fail) ==="
exit $fail
