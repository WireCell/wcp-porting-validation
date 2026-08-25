#!/bin/bash
# doc 82 -- minimal reproducer for the doc 81 sec 7.1 Q/L non-determinism.
#
# Doc 81 found that when several events run in ONE wire-cell process, a few
# marginal events out of 3067 come out with different Q/L point clouds and
# cluster-identity arrays, and that running the SAME group twice can give
# different answers -- so it is heap-layout dependent, not state carried from a
# particular predecessor.  That 2-event case was assembled by hand and never
# saved, and round 5 then pruned the group scratch it lived in.  This rebuilds
# it from the surviving per-event products (which are a lossless split of the
# group archives) so the bug is reproducible from a clean checkout in ~2 min.
#
#   ./repro_ql_nondet.sh <src_root> <out_dir> <evt> [<evt> ...]
#
# e.g. the doc 81 pair, whose second event is one of the seven that failed:
#   ./repro_ql_nondet.sh ../../work-mcp1k-grp0825 /home/xqian/tmp/d82/pair \
#        285993 286191
#
# Env:
#   DRAWS=N            how many independent runs of the same group (default 3)
#   PRECOMPILE=0|1     compile the config in a separate wcsonnet process (1) or
#                      let wire-cell run gojsonnet IN-PROCESS (0, the default,
#                      which is what run_chain_group.sh does).  This is the
#                      round's trigger test: the in-process path leaves a
#                      64-thread Go runtime alive for the whole job, and the
#                      other two drivers in this tree precompile precisely to
#                      avoid it (doc pr/97 sec 5).
#   REF=<root>         reference root to gate each draw against, in the
#                      per-event layout (default: the ql0819 arm for the
#                      sample the src_root names).
#   REALITY=data|sim   default data.
#
# Output: one line per draw per event saying whether that draw's Q/L products
# are member-identical to REF, plus a draw-to-draw identity matrix.  A run in
# which the draws disagree with EACH OTHER is the bug; a run in which they
# agree with each other but not with REF is a different (binary/config) change.
set -u

SX=$(cd "$(dirname "$0")/../.." && pwd -P)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
TK=$WCT_BASE/toolkit
export WIRECELL_PATH=$TK/cfg:$WCT_BASE/wire-cell-data:$WCT_BASE/wire-cell-data/sbnd/photodet:${WIRECELL_PATH:-}
export PYTHONPATH=$TK/pyutil/python:$WCT_BASE/local/python:$WCT_BASE/wire-cell-python:${PYTHONPATH:-}

[ $# -ge 3 ] || { sed -n '2,30p' "$0"; exit 1; }
SRC=$(cd "$1" && pwd -P); OUT=$2; shift 2
EVTS=("$@")
DRAWS=${DRAWS:-3}
PRECOMPILE=${PRECOMPILE:-0}
REALITY=${REALITY:-data}

# Default reference: work-<sample>-ql0819, derived from the src root's name.
if [ -z "${REF:-}" ]; then
    _s=$(basename "$SRC"); _s=${_s#work-}; _s=${_s%%-*}
    REF=$SX/work-${_s}-ql0819
fi
[ -d "$REF" ] || { echo "ERROR: no reference root: $REF" >&2; exit 1; }

mkdir -p "$OUT"
GDIR=$OUT/group
if [ ! -s "$GDIR/events.txt" ]; then
    python3 "$SX/scripts/multi/merge_group_products.py" "$SRC" "$GDIR" "${EVTS[@]}" \
        || exit 1
fi
# run/subrun per event: a group can span many runs and the job takes one pair.
[ -s "$GDIR/rse.json" ] || python3 "$SX/scripts/multi/reco1_rse_map.py" \
    --opflash "$GDIR/opflash_apa0.tar.gz" --out "$GDIR/rse.json" > "$GDIR/rse.log" 2>&1
RUN0=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); k=sorted(d)[0]; print(d[k][0])" "$GDIR/rse.json")
SUB0=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); k=sorted(d)[0]; print(d[k][1])" "$GDIR/rse.json")

echo "reproducer: ${#EVTS[@]} events, $DRAWS draws, PRECOMPILE=$PRECOMPILE"
echo "  group : $GDIR"
echo "  ref   : $REF"

for d in $(seq 1 "$DRAWS"); do
    DD=$OUT/draw$d
    if [ -f "$DD/.done" ]; then echo "draw $d: already present, reusing"; continue; fi
    mkdir -p "$DD"
    for e in "${EVTS[@]}"; do mkdir -p "$DD/ql_evt$e"; done

    _TLA=(--tla-str  "input=$GDIR"
          --tla-code "anode_indices=[0,1]"
          --tla-str  "output_dir=$DD"
          --tla-str  "evt_subdir=ql_evt%1%"
          --tla-code "run=$RUN0" --tla-code "subrun=$SUB0" --tla-code "event=0"
          --tla-str  "reality=$REALITY"
          --tla-code "multi_event=true"
          --tla-code "rse_map=$(cat "$GDIR/rse.json")"
          --tla-str  "save_tensors=$DD/ql_evt%1%/pctree-evt%1%.tar.gz")
    _CFG=(-c wct-clus-matching-perevt.jsonnet)
    if [ "$PRECOMPILE" = 1 ]; then
        if wcsonnet "${_TLA[@]}" -o "$DD/.wct-cfg.json" \
                "$TK/cfg/pgrapher/experiment/sbnd/wct-clus-matching-perevt.jsonnet" \
                > "$DD/wcsonnet.log" 2>&1; then
            _CFG=(-c "$DD/.wct-cfg.json"); _TLA=()
        else
            echo "draw $d: WARN wcsonnet failed -- in-process jsonnet" >&2
        fi
    fi
    setarch x86_64 -R wire-cell -l stderr -l "$DD/wct_ql.log:debug" -L debug \
        "${_TLA[@]}" "${_CFG[@]}" > "$DD/ql.stdout" 2>&1
    rc=$?
    echo "draw $d: rc=$rc"
    [ $rc -eq 0 ] || { echo "  see $DD/wct_ql.log" >&2; continue; }
    echo ok > "$DD/.done"
done

echo
echo "--- each draw vs the reference ($(basename "$REF")) ---"
for d in $(seq 1 "$DRAWS"); do
    [ -f "$OUT/draw$d/.done" ] || { echo "draw $d: NOT RUN"; continue; }
    printf 'draw %s: ' "$d"
    python3 "$SX/scripts/multi/repro_cmp.py" "$OUT/draw$d" "$REF" "${EVTS[@]}"
done

echo
echo "--- draw vs draw (disagreement here IS the bug) ---"
for a in $(seq 1 "$DRAWS"); do
    for b in $(seq $((a+1)) "$DRAWS"); do
        [ -f "$OUT/draw$a/.done" ] && [ -f "$OUT/draw$b/.done" ] || continue
        printf 'draw %s vs draw %s: ' "$a" "$b"
        python3 "$SX/scripts/multi/repro_cmp.py" "$OUT/draw$a" "$OUT/draw$b" "${EVTS[@]}"
    done
done
