#!/usr/bin/env bash
# Doc 59: run the DEFAULT SBND production nusel chain over the FULL 1000-event
# MCP2025C data sample (input_files_reco1/staged-mcp2025c-1000evt/e0..e999),
# one output root, N-way parallel, per-event rc + wall + peak-RSS recorded.
#
# Imaging is NEVER regenerated: work-mcp1kall-<tag>/evt<ID> is a symlink into
# work-mcp1000/evt<ID> (M11/M13).  Everything downstream of imaging IS produced
# fresh in this root: the Q/L step (run_nusel_evt.sh launches it because the
# pctree is missing) and the PR tagger tail + label table.
#
# Flags: since doc 68 the SBND production operating point IS the config default
# (cfg/pgrapher/experiment/sbnd/wct-{pr,clus-matching}-perevt.jsonnet), so the
# twelve-flag d55ton/d56bw string this script used to carry is gone -- passing
# nothing now reproduces it exactly.  What remains is -stm-fit, a pure
# diagnostic output (STM fit PCs + the Bee stm_fit layer + tracking-stm.root)
# that the doc-59 scan consumes and that the doc-64 line deliberately keeps OFF
# in the config.
#
# The Q/L step's lm / save_rcid / save_assoc are likewise config defaults now,
# so SBND_SAVE_ASSOC no longer has to be exported here.  Every event's status
# line still carries assoc=<mains>/<parts>, read back from the PR log's
# `<ClusteringUnmergeBundle:prassoc> unmerged ...` line: 0/0 on every event of a
# batch means the arrays were absent and the inner un-merge was a no-op (0/0 on
# a single event is legitimate -- nothing to undo).
#
# Usage: ./run_full1k_nusel.sh [nentries] [njobs]
#   nentries  how many art entries to process, from entry 0 (default 1000)
#   njobs     concurrent wire-cell chains (default 8; process_all.sh used 16 on
#             this same sample, CLAUDE.md caps routine batches at ~6-8 -- check
#             `cut -d' ' -f1 /proc/loadavg` after launch)
# Env:
#   TAG           output root suffix (default d59k) -> work-mcp1kall-<TAG>
#   ENTRIES       explicit entry list, e.g. ENTRIES="10 12 15" for a smoke batch
#                 (overrides nentries)
# Internal:
#   ./run_full1k_nusel.sh --worker <entry>    one event (used via xargs)
set -u
cd "$(dirname "$0")"
SBND_DIR=$PWD
TAG=${TAG:-d59k}
ROOT=$SBND_DIR/work-mcp1kall-$TAG
STAGE=$SBND_DIR/input_files_reco1/staged-mcp2025c-1000evt
IMGBASE=$SBND_DIR/work-mcp1000
MAP=$STAGE/entry_event_map.tsv
TC=$(cd ../../abtest && pwd)/timecmd.py

# The production flag set: everything else is now a config default (doc 68).
NUF="-stm-fit"

evt_of() { awk -F'\t' -v e="$1" '$1==e {print $4; exit}' "$MAP"; }

# --- one event -------------------------------------------------------------
if [ "${1:-}" = "--worker" ]; then
    i=$2
    evt=$(evt_of "$i")
    [ -n "$evt" ] || { echo "rc=90 entry=$i no-event-map-row" > "$ROOT/.status/$i"; exit 0; }
    st=$ROOT/.status/$i

    # Imaging comes from work-mcp1000, by absolute symlink (relink_tags.py-style
    # convention).  A missing imaging dir is a hard skip, not a silent 0-cluster
    # run.
    if [ ! -d "$IMGBASE/evt$evt" ]; then
        echo "rc=91 entry=$i evt=$evt no-imaging" > "$st"; exit 0
    fi
    ln -sfn "$IMGBASE/evt$evt" "$ROOT/evt$evt"

    # Per-entry cwd: the chain's wire-cell steps cd into their own output dirs,
    # but anything that does write to $PWD must not collide across 8 workers.
    cwd=$ROOT/.cwd/$i
    mkdir -p "$cwd"
    (
        cd "$cwd" || exit 92
        SBND_INPUT_DIR=$STAGE/e$i SBND_WORK_ROOT=$ROOT \
            setarch x86_64 -R python3 "$TC" "$ROOT/.time_e$i.meta" \
            "$SBND_DIR/run_nusel_evt.sh" data $NUF 1
    ) > "$ROOT/.log_e$i.log" 2>&1
    rc=$?
    # Did the inner (isolated-grouping) un-merge actually have its arrays?
    assoc=$(sed -n 's/.*<ClusteringUnmergeBundle:prassoc> unmerged \([0-9]*\) main cluster(s) into \([0-9]*\) .*/\1\/\2/p' \
        "$ROOT/nusel_evt$evt/wct_nusel_evt$evt.log" 2>/dev/null | tail -1)
    printf 'rc=%s entry=%s evt=%s %s assoc=%s\n' "$rc" "$i" "$evt" \
        "$(tr '\n' ' ' < "$ROOT/.time_e$i.meta" 2>/dev/null)" "${assoc:-none}" > "$st"
    rmdir "$cwd" 2>/dev/null
    exit 0
fi

# --- batch driver ----------------------------------------------------------
N=${1:-1000}
J=${2:-8}
mkdir -p "$ROOT/.status" "$ROOT/.cwd"

if [ -n "${ENTRIES:-}" ]; then
    list=$ENTRIES
else
    list=$(seq 0 $((N - 1)))
fi
nreq=$(echo "$list" | wc -w)

echo "=== doc59 full-1k nusel production ==="
echo "  tag=$TAG root=$ROOT"
echo "  entries=$nreq jobs=$J"
echo "  flags=$NUF (everything else = the config defaults, doc 68)"
echo "  started $(date -Is)"
t0=$(date +%s)
echo "$list" | tr ' ' '\n' | xargs -P "$J" -I{} "$SBND_DIR/run_full1k_nusel.sh" --worker {}
t1=$(date +%s)

ok=0; bad=0
for i in $list; do
    s=$ROOT/.status/$i
    if [ -s "$s" ] && grep -q '^rc=0 ' "$s"; then ok=$((ok + 1)); else bad=$((bad + 1)); fi
done
echo "=== done in $((t1 - t0)) s: ok=$ok failed=$bad of $nreq ==="
[ "$bad" -eq 0 ] || { echo "--- failures ---"; grep -L '^rc=0 ' "$ROOT"/.status/* 2>/dev/null | head -40; }
exit 0
