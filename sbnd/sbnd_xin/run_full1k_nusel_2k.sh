#!/usr/bin/env bash
# Doc 59: run the DEFAULT SBND production nusel chain over the FULL 1000-event
# MCP2025C data sample (input_files_reco1/staged-mcp2025c-1000evt/e0..e999),
# one output root, N-way parallel, per-event rc + wall + peak-RSS recorded.
#
# Imaging is symlinked from IMGBASE/evt<ID>, never copied (M11/M13) -- ROOT/
# evt<ID> is a symlink into $IMGBASE/evt<ID>.  Everything downstream of imaging
# IS produced fresh in ROOT: the Q/L step (run_nusel_evt.sh launches it because
# the pctree is missing) and the PR tagger tail + label table.
#
# CORRECTED 2026-08-05 (doc 71 campaign): the original comment here said
# "imaging is NEVER regenerated", true only while work-mcp1000 stood as a
# permanent hub.  That hub (and work-mcp1kall-d59k) was deleted in the
# 2026-08-05 clean-slate retire round -- the whole point of this campaign is
# to regenerate imaging fresh into IMGBASE (now an env var, see below) and
# point ROOT at it.  The symlink mechanics below are unchanged; only the
# claim that the source root is permanent is retracted.
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
#                 (ignored if ROOT is set explicitly)
#   ROOT          output root, overrides the TAG-derived work-mcp1kall-<TAG>
#                 name entirely (2026-08-05 campaign: work-mcp1k-cb0805 -- the
#                 old work-mcp1kall-d59k hub was deleted in the 2026-08-05
#                 clean-slate retire round, doc 71)
#   IMGBASE       imaging root to symlink evt<ID> from (default work-mcp1000,
#                 also deleted 2026-08-05 -- MUST be set for any run after
#                 that round, e.g. work-img-mcp1k)
#   ENTRIES       explicit entry list, e.g. ENTRIES="10 12 15" for a smoke batch
#                 (overrides nentries)
# Internal:
#   ./run_full1k_nusel.sh --worker <entry>    one event (used via xargs)
set -u
cd "$(dirname "$0")"
SBND_DIR=$PWD
TAG=${TAG:-d59k}
ROOT=${ROOT:-$SBND_DIR/work-mcp1kall-$TAG}
# doc pr/82 sec 2.3: fork of run_full1k_nusel.sh for the SECOND 2000-event
# MCP2025C data batch.  STAGE (and therefore MAP) is the ONLY difference -- it
# is hardcoded in the original at :58, while ROOT / IMGBASE / TAG / ENTRIES were
# already env-overridable.  Fork rather than parameterise per CLAUDE.md M10: the
# original is the doc-59 / doc-71 first-1k production record and stays
# byte-untouched.
STAGE=${STAGE:-$SBND_DIR/input_files_reco1/staged-mcp2025c-2nd-2000evt}
IMGBASE=${IMGBASE:-$SBND_DIR/work-mcp1000}
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
# MUST be this script, not run_full1k_nusel.sh.  The original hardcodes STAGE
# to the 1000-evt dir, whose map has no rows for entries 1000..1999 -- and a
# missing map row is not an error: evt_of() returns empty, the worker writes
# `rc=90 no-event-map-row` and exits 0.  Dispatching to the original would
# silently skip half this sample and report a completed batch.
echo "$list" | tr ' ' '\n' | xargs -P "$J" -I{} "$SBND_DIR/run_full1k_nusel_2k.sh" --worker {}
t1=$(date +%s)

ok=0; bad=0
for i in $list; do
    s=$ROOT/.status/$i
    if [ -s "$s" ] && grep -q '^rc=0 ' "$s"; then ok=$((ok + 1)); else bad=$((bad + 1)); fi
done
echo "=== done in $((t1 - t0)) s: ok=$ok failed=$bad of $nreq ==="
[ "$bad" -eq 0 ] || { echo "--- failures ---"; grep -L '^rc=0 ' "$ROOT"/.status/* 2>/dev/null | head -40; }
exit 0
