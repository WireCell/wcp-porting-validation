#!/bin/bash
# Phase A: extract every art-file entry into its own single-event sample dir.
#
# Deliberately a barrier before imaging: work dirs are keyed by event ID
# (work/evt<ID>), but event numbers are only unique within a (run, subrun).
# If this file spans runs, two entries could share an ID and silently clobber
# each other.  Staging first lets us prove uniqueness for the cost of ~2 min.
#
# Usage: stage_all.sh <nentries> <njobs>
set -u

SBND=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
SP=/home/xqian/tmp/claude-25225/-nfs-data-1-xqian-toolkit-dev-wcp-porting-img-sbnd-sbnd-xin/5c05a71b-98ea-4282-95d8-24bf944a98e7/scratchpad

export SBND
export STAGE=$SBND/input_files_reco1/staged-mcp2025c-1000evt
export RUNDIR=$SP/run1000
export INPUT=$(readlink -f "$SBND/input_files_reco1/data_MCP2025C_reco1_frameshift_first1000ev.root")
export WIRECELL_PATH=$WCT_BASE/toolkit/cfg:$WCT_BASE/wire-cell-data:$WCT_BASE/wire-cell-data/sbnd/photodet

# The reco1 source factories moved out of libWireCellRoot.so into the standalone
# plugin wire-cell-sbnd-reco1 (WCT master 5f684887, issue #494).  See the build
# recipe in $SBND/run_reco1_dump.sh.
SBND_RECO1=${SBND_RECO1:-$WCT_BASE/wire-cell-sbnd-reco1/install}
[ -r "$SBND_RECO1/lib/libWireCellSBNDReco1.so" ] || {
    echo "ERROR: reco1 plugin missing: $SBND_RECO1/lib/libWireCellSBNDReco1.so" >&2; exit 1; }
export LD_LIBRARY_PATH=$SBND_RECO1/lib:${LD_LIBRARY_PATH:-}
export WIRECELL_PATH=$WIRECELL_PATH:$SBND_RECO1/share/wirecell

N=${1:-1000}
J=${2:-16}

mkdir -p "$STAGE" "$RUNDIR"/{dumpstatus,log}

dump_one() {
    local i=$1
    local d="$STAGE/e$i" st="$RUNDIR/dumpstatus/$i"
    if [ -s "$d/frames-dnn.tar.bz2" ] && [ -s "$d/opflash_apa0.tar.gz" ]; then
        echo "OK" > "$st"; return 0
    fi
    mkdir -p "$d"
    if ! wire-cell -l "$RUNDIR/log/dump-$i.log:info" -L info \
            --tla-str "input=$INPUT" --tla-str "output_dir=$d" \
            --tla-str "caf_offset_mode=product" \
            --tla-str "caf_offset_override=0" --tla-str "entry=$i" \
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
