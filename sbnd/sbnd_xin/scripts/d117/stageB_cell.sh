#!/bin/bash
# doc sbnd_xin/117: stage B (the 15-stage PR + tagger + BDT chain) for ONE VERTEX-CHAIN cell on a
# doc-115 stage-A arm.  Fork by duplication (CLAUDE.md M10) of scripts/d116/stageB_cell.sh, which
# stays untouched and keeps writing the doc-116 trajectory arms work-r3<s>-d116<cell>; that one is
# itself a fork of scripts/d115/stageB.sh (the baseline arms work-r3<s>-d115pr).  Same stage A,
# same pin, same output set; only the TLA file and the out_root prefix differ.
#
# A cell is a set of PR-job knobs, given as ONE file of raw jsonnet `key=value` lines
# (docs/117_figs/tla/<cell>.tla -> PR_EXTRA_TLA, appended last by run_pr_chain_batch.sh:1881 so it
# wins over every SBND_* env) plus, when docs/117_figs/tla/<cell>.tfjson exists, the TrackFitting
# JSON it names (-> SBND_TRACKFIT_JSON -> --tla-str trackfitting_config=...).  The fit keys of the
# PDHD/PDVD production (fit_weight_pow 1.5, assoc_cont_center 1) are ONLY reachable that way: they
# are read at runtime, never from the compiled jsonnet (doc pr/150 sec 2, the `tfull` trap).
# Every doc-117 cell HAS a .tla (unlike doc 116's `s0rep`): the noise floor for this round is doc
# 116 sec 4's, measured on this same pin and box, where the whole chain reproduced itself row for
# row (CLAUDE.md M4 did not bite).  A cell without a .tla is refused rather than silently running
# production.
#
# Same pin as the baseline arms (LIBSNAP, default ~/tmp/d115-libsnap = toolkit 0a2807f4), same
# PR_EXTRA_STAGES=pr_display, same per-sub-root layout (MC keeps .lineage_reality per f*/, so the
# runner is pointed at each sub-root, never at the top dir), same nusel merge.  The only difference
# between a cell arm and the baseline arm is the TLA set -- which lands in Trun.op_config_sha256
# (.d109-opcfg.json), the per-arm proof of what ran.
#
# Guards: a TLA file containing '__' is refused (launcher placeholders, feedback memory); an
# existing out_root is refused unless its .d117_cell marker names this cell (resume of a partial
# run of the SAME cell -- M13; the batch runner's own "fresh dir" guard is documented but not
# implemented).  Pin md5 printed at start and end.
#
# Usage: [JOBS=n] stageB_cell.sh <cell> <cv|nuecc|off> [f000 f001 ...]
#   cell   c1e | c1x | t1e | t1x | any later rung with docs/117_figs/tla/<cell>.tla
#          c/t = current / PDHD-PDVD trajectory; 1 = one-step vertex chain (dl_vtx_dual_chain=false);
#          e/x = fit exclusion on / off.  The 2-step corners are the doc-115 baseline (c2) and the
#          doc-116 `tfull` arms (t2); neither is re-run here.
#   f*     optional sub-root subset (the pilot); default = every stage-A sub-root
set -u
cd -P "$(dirname "$0")/../.." || exit 1
SX=$PWD
CELL=${1:?usage: [JOBS=n] stageB_cell.sh <cell> <cv|nuecc|off> [f000 ...]}
S=${2:?usage: [JOBS=n] stageB_cell.sh <cell> <cv|nuecc|off> [f000 ...]}
shift 2
J=${JOBS:-16}
export LIBSNAP=${LIBSNAP:-$HOME/tmp/d115-libsnap}
export LD_LIBRARY_PATH=$LIBSNAP:${LD_LIBRARY_PATH:-}
export PR_EXTRA_STAGES=${PR_EXTRA_STAGES:-pr_display}
unset PR_EXTRA_TLA SBND_TRACKFIT_JSON PR_GROUP_SIZE SBND_NO_DL PR_CFG_TREE

case "$S" in
    cv)    A=$SX/work-r3cv-d115;  B=$SX/work-r3cv-d117$CELL;  REALITY=sim ;;
    nuecc) A=$SX/work-r3nue-d115; B=$SX/work-r3nue-d117$CELL; REALITY=sim ;;
    off)   A=$SX/work-r3off-d115; B=$SX/work-r3off-d117$CELL; REALITY=data ;;
    *) echo "unknown sample: $S" >&2; exit 2 ;;
esac
[ -d "$A" ] || { echo "ERROR: no stage-A arm $A" >&2; exit 1; }

TLA=$SX/docs/117_figs/tla/$CELL.tla
TFJ=$SX/docs/117_figs/tla/$CELL.tfjson
[ -s "$TLA" ] || { echo "ERROR: cell '$CELL' has no $TLA (every doc-117 cell needs one)" >&2; exit 2; }
grep -q '__' "$TLA" && { echo "REFUSING: placeholder '__' in $TLA" >&2; exit 2; }
export PR_EXTRA_TLA=$TLA
if [ -s "$TFJ" ]; then
    _tf=$(head -1 "$TFJ")
    [ -r "$_tf" ] || { echo "ERROR: $TFJ names an unreadable TrackFitting JSON: $_tf" >&2; exit 2; }
    export SBND_TRACKFIT_JSON=$_tf
fi

if [ -e "$B" ]; then
    [ "$(cat "$B/.d117_cell" 2>/dev/null)" = "$CELL" ] \
        || { echo "REFUSING: $B exists and is not a '$CELL' arm of this script (M13)" >&2; exit 2; }
    echo "=== resuming existing $B"
else
    mkdir -p "$B" && echo "$CELL" > "$B/.d117_cell"
fi
LOGD=$HOME/tmp/d117-stageB-$CELL-$S; mkdir -p "$LOGD"

echo "=== d117 stage B: cell=$CELL sample=$S reality=$REALITY PR_JOBS=$J  $(date +%F_%H:%M:%S)"
echo "=== libsnap: $LIBSNAP (toolkit $(cat "$LIBSNAP/TOOLKIT_HEAD" 2>/dev/null))"
echo "=== toolkit HEAD before: $(git -C /nfs/data/1/xqian/toolkit-dev/toolkit rev-parse HEAD)"
echo "=== PR_EXTRA_TLA=${PR_EXTRA_TLA:-<none>}  SBND_TRACKFIT_JSON=${SBND_TRACKFIT_JSON:-<none>}  PR_EXTRA_STAGES=$PR_EXTRA_STAGES"
[ -n "${PR_EXTRA_TLA:-}" ] && grep -v '^\s*#' "$PR_EXTRA_TLA" | sed 's/^/    tla: /'
echo "=== pin start: $(md5sum "$LIBSNAP/libWireCellClus.so")"
t0=$(date +%s)

if [ "$S" = off ]; then
    SUBS=("")                       # one root
elif [ $# -gt 0 ]; then
    SUBS=("$@")
else
    mapfile -t SUBS < <(ls -d "$A"/f* 2>/dev/null | xargs -n1 basename | sort)
fi
echo "=== stage-A sub-roots: ${#SUBS[@]}"

# Several workers may share one cell (the sub-roots are 6-21 events each, so one sequential loop
# leaves the box idle -- doc 115 ran stageB_range.sh beside stageB.sh for the same reason).  A
# sub-root is claimed with an atomic mkdir of $qb/.d117_running and released on completion; a
# worker skips what another holds.  CLEAR_LOCKS=1 drops stale claims left by a killed worker
# (the sub-root is then re-run in full: an event killed mid-write has no rc.txt).
[ "${CLEAR_LOCKS:-0}" = 1 ] && rm -rf "$B"/f*/.d117_running "$B"/.d117_running
complete_sub() {  # qa qb -> 0 when every stage-A event has a pr_evt dir with rc=0
    local nql npr nrc
    nql=$(ls -d "$1"/ql_evt* 2>/dev/null | wc -l)
    npr=$(ls -d "$2"/pr_evt* 2>/dev/null | wc -l)
    nrc=$(grep -ls 'rc=0' "$2"/pr_evt*/rc.txt 2>/dev/null | wc -l)
    [ "$nql" -gt 0 ] && [ "$npr" -eq "$nql" ] && [ "$nrc" -eq "$nql" ]
}
n_skipped=0
for sub in "${SUBS[@]}"; do
    qa=$A${sub:+/$sub}; qb=$B${sub:+/$sub}
    [ -d "$qa" ] || { echo "[${sub:-.}] ERROR: no stage-A sub-root $qa" >&2; continue; }
    nql=$(ls -d "$qa"/ql_evt* 2>/dev/null | wc -l)
    if complete_sub "$qa" "$qb"; then
        echo "[${sub:-.}] already complete ($nql/$nql) -- skipped"; continue
    fi
    mkdir -p "$qb"
    if ! mkdir "$qb/.d117_running" 2>/dev/null; then
        echo "[${sub:-.}] held by another worker -- skipped"; n_skipped=$((n_skipped+1)); continue
    fi
    PR_JOBS=$J "$SX/run_pr_chain_batch.sh" "$qa" "$qb" "$REALITY" \
        > "$LOGD/${sub:-all}.log" 2>&1
    rc=$?
    rmdir "$qb/.d117_running"
    echo "[${sub:-.}] rc=$rc pr_evt=$(ls -d "$qb"/pr_evt* 2>/dev/null | wc -l)/$nql opcfg=$(sha256sum "$qb/.d109-opcfg.json" 2>/dev/null | cut -c1-16)"
done

# One nusel table pair per sample, as stageB.sh does (the truth analysis does not read them).
# Only the worker that sees every sub-root complete (no one else still holds one) merges.
n_incomplete=0
for sub in "${SUBS[@]}"; do
    complete_sub "$A${sub:+/$sub}" "$B${sub:+/$sub}" || n_incomplete=$((n_incomplete+1))
done
if [ $# -eq 0 ] && [ "$n_incomplete" -gt 0 ]; then
    echo "=== $n_incomplete sub-root(s) not complete (held by other workers or failed) -- merge skipped"
elif [ $# -eq 0 ]; then
    echo "=== merging the per-file nusel tables"
    mapfile -t _NS < <(ls "$B"/f*/pr_evt*/nusel-evt*.tsv "$B"/pr_evt*/nusel-evt*.tsv 2>/dev/null)
    if [ "${#_NS[@]}" -gt 0 ]; then
        python3 "$SX/nusel_extract.py" --merge "${_NS[@]}" \
            --out "$B/nusel-table.tsv" --events-out "$B/nusel-events.tsv" \
            > "$LOGD/merge.log" 2>&1
        echo "  merge rc=$? rows=$(( $(wc -l < "$B/nusel-table.tsv" 2>/dev/null || echo 1) - 1 ))" \
             "events=$(( $(wc -l < "$B/nusel-events.tsv" 2>/dev/null || echo 1) - 1 ))"
    else
        echo "  no per-event nusel tables found"
    fi
fi

t1=$(date +%s)
echo "=== pin end  : $(md5sum "$LIBSNAP/libWireCellClus.so")"
echo "=== toolkit HEAD after : $(git -C /nfs/data/1/xqian/toolkit-dev/toolkit rev-parse HEAD)"
echo "=== distinct op_config_sha256 in the arm: $(sha256sum "$B"/f*/.d109-opcfg.json "$B"/.d109-opcfg.json 2>/dev/null | cut -c1-16 | sort -u | tr '\n' ' ')"
echo "=== stage B $CELL/$S finished in $((t1-t0)) s"
echo "=== pr_evt dirs : $(ls -d "$B"/*/pr_evt* "$B"/pr_evt* 2>/dev/null | wc -l)"
echo "=== rc!=0 events: $(grep -L 'rc=0' "$B"/*/pr_evt*/rc.txt "$B"/pr_evt*/rc.txt 2>/dev/null | wc -l)"
exit 0
