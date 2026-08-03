#!/bin/bash
# Drive the full MCP2025C 1000-event sample through dump -> imaging -> QLMatching.
#
# One worker per art-file entry, pipelined (dump, img, ql chained per event) and
# fanned out with xargs -P.  Each entry keeps its OWN single-event sample dir:
# _runlib.sh stages an event by decompressing the whole frames-dnn.tar.bz2 with
# a wildcard, so one 1000-event archive would cost 1000 full-archive scans
# (quadratic).  Per-entry archives keep that staging at ~1 MB per event.
#
# Usage: process_all.sh <nentries> <njobs>
set -u

SBND=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
SP=/home/xqian/tmp/claude-25225/-nfs-data-1-xqian-toolkit-dev-wcp-porting-img-sbnd-sbnd-xin/5c05a71b-98ea-4282-95d8-24bf944a98e7/scratchpad

export SBND WCT_BASE
export STAGE=$SBND/input_files_reco1/staged-mcp2025c-1000evt
export WORK=$SBND/work-mcp1000
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

mkdir -p "$STAGE" "$WORK" "$RUNDIR"/{status,log,cwd}

worker() {
    local i=$1
    local d="$STAGE/e$i" st="$RUNDIR/status/$i" cwd="$RUNDIR/cwd/$i"
    mkdir -p "$cwd" && cd "$cwd" || { echo "cwd-FAIL" > "$st"; return 0; }

    # 1. extract this entry (idempotent: skip if already staged)
    if [ ! -s "$d/frames-dnn.tar.bz2" ]; then
        mkdir -p "$d"
        if ! wire-cell -l "$RUNDIR/log/dump-$i.log:info" -L info \
                --tla-str "input=$INPUT" --tla-str "output_dir=$d" \
                --tla-str "caf_offset_mode=product" \
                --tla-str "caf_offset_override=0" --tla-str "entry=$i" \
                -c "$SBND/wct-reco1-dump.jsonnet" > "$RUNDIR/log/dump-$i.out" 2>&1; then
            echo "dump-FAIL" > "$st"; return 0
        fi
    fi

    local evt
    evt=$(tar tjf "$d/frames-dnn.tar.bz2" 2>/dev/null \
          | sed -n 's/^frame_dnnsp_\([0-9][0-9]*\)\.npy$/\1/p' | head -1)
    if [ -z "$evt" ]; then echo "noevent-FAIL" > "$st"; return 0; fi

    # 2. imaging
    if ! SBND_INPUT_DIR="$d" SBND_WORK_ROOT="$WORK" \
            "$SBND/run_img_evt.sh" data 1 > "$RUNDIR/log/img-$i.out" 2>&1; then
        echo "img-FAIL $evt" > "$st"; return 0
    fi
    rm -f "$RUNDIR/log/img-$i.out"

    # 3. Q/L matching
    if ! SBND_INPUT_DIR="$d" SBND_WORK_ROOT="$WORK" \
            "$SBND/run_ql_evt.sh" data 1 > "$RUNDIR/log/ql-$i.out" 2>&1; then
        echo "ql-FAIL $evt" > "$st"; return 0
    fi
    rm -f "$RUNDIR/log/ql-$i.out" "$RUNDIR/log/dump-$i.out" "$cwd/trash-all-apa.tar.gz"

    echo "OK $evt" > "$st"
}
export -f worker

echo "entries: $N   jobs: $J   started: $(date -Is)"
seq 0 $((N - 1)) | xargs -P "$J" -I{} bash -c 'worker "$@"' _ {}
echo "finished: $(date -Is)"

echo "=== summary ==="
cat "$RUNDIR"/status/* 2>/dev/null | awk '{print $1}' | sort | uniq -c
