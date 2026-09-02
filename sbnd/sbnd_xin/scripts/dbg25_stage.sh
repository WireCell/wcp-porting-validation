#!/bin/bash
# doc 95 phase A -- stage each art-file entry of the colleague's 25-event MC
# debug sample into its OWN single-event sample dir.
#
# WHY PER-ENTRY AND NOT ONE ARCHIVE (this is the whole point of the script):
# a whole-file `run_reco1_dump.sh -t dbg25` was run first and produced 25 frame
# members but only **20 distinct event ids** -- ids 12, 14, 22, 31, 34 each
# appear twice.  Every downstream name (frame_dnnsp_<ID>.npy, work/evt<ID>,
# ql_evt<ID>, pr_evt<ID>) is keyed on the BARE event id, which is unique only
# within a (run, subrun); this sample is 25 debug events drawn from 100 files,
# so it spans many runs.  In the combined archive the second copy silently
# overwrites the first and five events vanish with no error anywhere.
# Precedent + reasoning: staged-mcp2025c-1000evt/stage_all.sh and
# staged-mcp2025c-2nd-2000evt/check_uniqueness.py (doc pr/82 sec 2.1).
#
# MC, like the 8-event set of doc 93: reco1 products under simtpc2d/DetSim, no
# FrameShiftInfo/PTB/TDC product => caf_offset_mode=none, reality=sim downstream.
#
# Usage: dbg25_stage.sh [nentries] [njobs]
set -u

SBND=$(cd -P "$(dirname "$0")/.." && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export STAGE=$SBND/input_files_reco1/staged-dbg25
export RUNDIR=/home/xqian/tmp/dbg25/stage      # durable, NOT a session scratchpad
export INPUT=$(readlink -f "$SBND/input_files_reco1/stm_tagger_feedback/debug-25evt-reco1.root")
export SBNDCFG=$SBND
export WIRECELL_PATH=$WCT_BASE/toolkit/cfg:$WCT_BASE/wire-cell-data:$WCT_BASE/wire-cell-data/sbnd/photodet

SBND_RECO1=${SBND_RECO1:-$WCT_BASE/wire-cell-sbnd-reco1/install}
[ -r "$SBND_RECO1/lib/libWireCellSBNDReco1.so" ] || {
    echo "ERROR: reco1 plugin missing: $SBND_RECO1/lib/libWireCellSBNDReco1.so" >&2; exit 1; }
export LD_LIBRARY_PATH=$SBND_RECO1/lib:${LD_LIBRARY_PATH:-}
export WIRECELL_PATH=$WIRECELL_PATH:$SBND_RECO1/share/wirecell

N=${1:-25}
J=${2:-6}
mkdir -p "$STAGE" "$RUNDIR"/{status,log}

dump_one() {
    local i=$1
    local d="$STAGE/e$i" st="$RUNDIR/status/$i"
    if [ -s "$d/frames-dnn.tar.bz2" ] && [ -s "$d/opflash_apa0.tar.gz" ]; then
        echo "OK" > "$st"; return 0
    fi
    mkdir -p "$d"
    # MC product names (run_reco1_dump.sh -mc) + caf none (-caf none).
    if ! wire-cell -l "$RUNDIR/log/dump-$i.log:info" -L info \
            --tla-str "input=$INPUT" --tla-str "output_dir=$d" \
            --tla-str "entry=$i" \
            --tla-str "caf_offset_mode=none" --tla-str "caf_offset_override=0" \
            --tla-str "wire_product=recob::Wires_simtpc2d_dnnsp_DetSim." \
            --tla-str "badmask_product=ints_simtpc2d_badmasks_DetSim." \
            --tla-str "summary_product=doubles_simtpc2d_wienersummary_DetSim." \
            --tla-str "frameshift_product=" \
            -c "$SBNDCFG/wct-reco1-dump.jsonnet" > "$RUNDIR/log/dump-$i.out" 2>&1; then
        echo "FAIL" > "$st"; return 0
    fi
    echo "OK" > "$st"
}
export -f dump_one

echo "staging $N entries with $J jobs from $INPUT"
echo "started $(date -Is)"
seq 0 $((N - 1)) | xargs -P "$J" -I{} bash -c 'dump_one "$@"' _ {}
echo "finished $(date -Is)"
grep -h . "$RUNDIR"/status/* | sort | uniq -c
