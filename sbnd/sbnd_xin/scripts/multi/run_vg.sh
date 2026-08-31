#!/bin/bash
# doc 82 r2 -- one memcheck run with a named malloc/free fill pair.
#   run_vg.sh <out> <evt> <malloc-fill> <free-fill> [freelist-bytes]
# Pass "-" for either fill to leave valgrind's default (no fill) in place --
# that is how the B0/E rows of the doc-82 sec 2c table were produced, and the
# pair (fills off, freelist 20MB vs 500MB) is what showed the answer tracks
# REUSE rather than either fill byte.  freelist-bytes defaults to 500 MB.
# valgrind replaces the allocator, so MALLOC_PERTURB_ is inert inside it;
# --malloc-fill / --free-fill are its equivalents AND they are independent,
# which glibc's single perturb byte is not.  That independence is the point:
# an outcome tracking --malloc-fill is a read of NEVER-WRITTEN memory, one
# tracking --free-fill is a read of FREED memory.
set -u
SX=$(cd "$(dirname "$0")/../.." && pwd -P)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
TK=$WCT_BASE/toolkit
export WIRECELL_PATH=$TK/cfg:$WCT_BASE/wire-cell-data:$WCT_BASE/wire-cell-data/sbnd/photodet:${WIRECELL_PATH:-}
export PYTHONPATH=$TK/pyutil/python:$WCT_BASE/local/python:$WCT_BASE/wire-cell-python:${PYTHONPATH:-}

OUT=$1; EVT=$2; MF=$3; FF=$4; FL=${5:-500000000}
_FILL=()
[ "$MF" != "-" ] && _FILL+=(--malloc-fill="$MF")
[ "$FF" != "-" ] && _FILL+=(--free-fill="$FF")
SRC=$SX/work-mcp2k-grp0825
case $EVT in 286191|292643) SRC=$SX/work-mcp1k-grp0825 ;; esac
mkdir -p "$OUT"
GDIR=$OUT/group
if [ ! -s "$GDIR/events.txt" ]; then
    python3 "$SX/scripts/multi/merge_group_products.py" "$SRC" "$GDIR" "$EVT" || exit 1
fi
[ -s "$GDIR/rse.json" ] || python3 "$SX/scripts/multi/reco1_rse_map.py" \
    --opflash "$GDIR/opflash_apa0.tar.gz" --out "$GDIR/rse.json" > "$GDIR/rse.log" 2>&1
RUN0=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); k=sorted(d)[0]; print(d[k][0])" "$GDIR/rse.json")
SUB0=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); k=sorted(d)[0]; print(d[k][1])" "$GDIR/rse.json")

DD=$OUT/draw1
mkdir -p "$DD/ql_evt$EVT"
_TLA=(--tla-str  "input=$GDIR" --tla-code "anode_indices=[0,1]"
      --tla-str  "output_dir=$DD" --tla-str "evt_subdir=ql_evt%1%"
      --tla-code "run=$RUN0" --tla-code "subrun=$SUB0" --tla-code "event=0"
      --tla-str  "reality=data" --tla-code "multi_event=true"
      --tla-code "rse_map=$(cat "$GDIR/rse.json")"
      --tla-str  "save_tensors=$DD/ql_evt%1%/pctree-evt%1%.tar.gz")
wcsonnet "${_TLA[@]}" -o "$DD/.wct-cfg.json" \
    "$SX/wct-clus-matching-perevt.jsonnet" > "$DD/wcsonnet.log" 2>&1 || exit 1

valgrind --tool=memcheck --error-limit=no --num-callers=25 --leak-check=no \
         ${_FILL[@]+"${_FILL[@]}"} \
         --freelist-vol="$FL" --freelist-big-blocks=0 \
         --log-file="$DD/memcheck.log" \
    wire-cell -l stderr -l "$DD/wct_ql.log:debug" -L debug \
        -c "$DD/.wct-cfg.json" > "$DD/ql.stdout" 2>&1
echo "rc=$? malloc-fill=$MF free-fill=$FF freelist=$FL"
echo ok > "$DD/.done"
