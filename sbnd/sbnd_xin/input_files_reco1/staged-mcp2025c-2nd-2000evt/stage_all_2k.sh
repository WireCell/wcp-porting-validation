#!/bin/bash
# doc pr/82 sec 2 Phase A: extract every art-file entry of the SECOND MCP2025C
# data batch into its own single-event sample dir.
#
# Fork of ../staged-mcp2025c-1000evt/stage_all.sh (M10: that file is the
# first-1k record and stays byte-untouched).  Two changes, both forced:
#
#   1. TWO input files instead of one.  The flat index is
#          e0    .. e999    <- part1 entry i
#          e1000 .. e1999   <- part2 entry (i - 1000)
#      This mapping has no precedent in the tree, which is why it is written
#      down here and in PROVENANCE.txt rather than only living in the code.
#   2. SP/RUNDIR point at a durable dir.  The first-1k copy hardcodes a 2026-07
#      session scratchpad that no longer exists (its own PROVENANCE.txt:16-18
#      says to edit these before re-running).
#
# Deliberately a barrier before imaging, exactly as the first-1k script says:
# work dirs are keyed by event ID (work/evt<ID>), but event numbers are only
# unique within a (run, subrun).  Staging first lets us prove uniqueness for
# the cost of ~4 min -- and this sample is riskier than the first, because it
# spans the SAME runs (18255/18259), so a collision could be across samples as
# well as within.  The assertions live in check_uniqueness.py, run separately.
#
# Per-entry staging is mandatory, not stylistic: _runlib.sh:113 stages an event
# by wildcard-extracting the WHOLE frames-dnn.tar.bz2, so one combined
# 2000-event archive would make staging quadratic.
#
# Usage: stage_all_2k.sh [nentries] [njobs]
set -u

SBND=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
SP=/home/xqian/tmp/pr82

export SBND
export STAGE=$SBND/input_files_reco1/staged-mcp2025c-2nd-2000evt
export RUNDIR=$SP/stage2k
export SRC=/nfs/data/1/yuhw/production-prep/add-frameshift-data-2nd-2k-2026-08-15
export PART1=$SRC/data_MCP2025C_reco1_frameshift_2nd1k_part1.root
export PART2=$SRC/data_MCP2025C_reco1_frameshift_2nd1k_part2.root
export SPLIT=1000                       # entries in PART1
export WIRECELL_PATH=$WCT_BASE/toolkit/cfg:$WCT_BASE/wire-cell-data:$WCT_BASE/wire-cell-data/sbnd/photodet

for f in "$PART1" "$PART2"; do
    [ -r "$f" ] || { echo "ERROR: missing input: $f" >&2; exit 1; }
done

# The reco1 source factories live in the standalone plugin wire-cell-sbnd-reco1
# (WCT master 5f684887, issue #494).  Build recipe: $SBND/run_reco1_dump.sh:44-51.
SBND_RECO1=${SBND_RECO1:-$WCT_BASE/wire-cell-sbnd-reco1/install}
[ -r "$SBND_RECO1/lib/libWireCellSBNDReco1.so" ] || {
    echo "ERROR: reco1 plugin missing: $SBND_RECO1/lib/libWireCellSBNDReco1.so" >&2; exit 1; }
export LD_LIBRARY_PATH=$SBND_RECO1/lib:${LD_LIBRARY_PATH:-}
export WIRECELL_PATH=$WIRECELL_PATH:$SBND_RECO1/share/wirecell

N=${1:-2000}
J=${2:-32}

mkdir -p "$STAGE" "$RUNDIR"/{dumpstatus,log}

dump_one() {
    local i=$1
    local d="$STAGE/e$i" st="$RUNDIR/dumpstatus/$i"
    if [ -s "$d/frames-dnn.tar.bz2" ] && [ -s "$d/opflash_apa0.tar.gz" ]; then
        echo "OK" > "$st"; return 0
    fi
    # The two-file mapping.  Keep this the ONLY place it is expressed.
    local in e
    if [ "$i" -lt "$SPLIT" ]; then in=$PART1; e=$i; else in=$PART2; e=$((i - SPLIT)); fi
    mkdir -p "$d"
    # caf_offset_mode=product reads the authoritative FrameShiftInfo product.
    # frameshift_product is NOT passed: empty => wct-reco1-dump.jsonnet:133
    # suppresses the key => the C++ default instance
    # sbnd::timing::FrameShiftInfo_frameshift__FRAMESHIFT. applies, which doc
    # pr/82 sec 0b test 2 confirmed by branch-name grep on both part files.
    if ! wire-cell -l "$RUNDIR/log/dump-$i.log:info" -L info \
            --tla-str "input=$in" --tla-str "output_dir=$d" \
            --tla-str "caf_offset_mode=product" \
            --tla-str "caf_offset_override=0" --tla-str "entry=$e" \
            -c "$SBND/wct-reco1-dump.jsonnet" > "$RUNDIR/log/dump-$i.out" 2>&1; then
        echo "FAIL" > "$st"; return 0
    fi
    rm -f "$RUNDIR/log/dump-$i.out" "$RUNDIR/log/dump-$i.log"
    echo "OK" > "$st"
}
export -f dump_one

echo "staging $N entries with $J jobs, started $(date -Is)"
seq 0 $((N - 1)) | xargs -P "$J" -I{} bash -c 'dump_one "$@"' _ {}
echo "staging finished $(date -Is)"
grep -h . "$RUNDIR"/dumpstatus/* | sort | uniq -c
