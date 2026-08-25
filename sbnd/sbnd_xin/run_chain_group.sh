#!/bin/bash
# doc 76 round 2 -- STAGE A of the two-stage SBND chain, one process per STAGE
# per GROUP of events instead of one process per stage per event.
#
# reco1 art file --> imaging --> clustering + Q/L matching --> ONE output set
#
#   <out>/g<K>/frames-dnn.tar.bz2            the group's SP frames + opflash
#   <out>/g<K>/opflash_apa{0,1}.tar.gz
#   <out>/g<K>/icluster-apa{0,1}-{active,masked}.npz   all the group's events
#   <out>/g<K>/pctree-ql.tar.gz              the group's post-Q/L trees
#   <out>/g<K>/mabc.zip                      Bee, one layer set per event
#
# Stage B (the pattern recognition) is run_pr_chain_batch.sh with PR_GROUP_SIZE.
#
# Why this is a runner and not a new jsonnet: nothing here needs a new graph.
# wct-img-all.jsonnet never had an event TLA at all, the cluster/tensor/frame
# sources have always streamed a whole archive to EOS, and the Q/L job needed
# only the default-OFF `multi_event` knob so each event's Bee layers are
# labelled from its own tensor ident instead of a per-process constant.  The
# per-event drivers (run_img_evt.sh, run_ql_evt.sh, run_pr_evt.sh) are
# untouched and remain the way to run or re-run a single event or a single
# step.
#
# Usage:
#   ./run_chain_group.sh <reco1.root> <out_root> <data|sim> [--size G] [--group K]
#                        [--groups K1,K2,...] [--from img|ql] [--to img|ql]
#
#   --size G     events per group (default 16; see docs/76 round 2 for why)
#   --entries N  entries in the art file (default: read from the file)
#   --group K    run only group K (0-based).  Default: every group in the file.
#   --groups L   comma-separated group list.
#   --from/--to  run only part of the chain on an existing group dir, e.g.
#                `--from ql` to redo Q/L from the imaging checkpoint.
#
# Env: SBND_MAX_JOBS (concurrent GROUPS, default 4 -- each wire-cell process is
#      itself multi-threaded, CLAUDE.md M5), SBND_RECO1 (plugin install dir).
set -u

SX=$(cd "$(dirname "$0")" && pwd -P)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
TK=$WCT_BASE/toolkit
AB=$SX/../../abtest
export WIRECELL_PATH=$TK/cfg:$WCT_BASE/wire-cell-data:$WCT_BASE/wire-cell-data/sbnd/photodet:${WIRECELL_PATH:-}
export PYTHONPATH=$TK/pyutil/python:$WCT_BASE/local/python:$WCT_BASE/wire-cell-python:${PYTHONPATH:-}

# The reco1 art-file sources live in the standalone WireCellSBNDReco1 plugin
# (WCT master moved them out; see sbnd_xin/wct-reco1-dump.jsonnet's header).
SBND_RECO1=${SBND_RECO1:-${WCT_BASE}/wire-cell-sbnd-reco1/install}
export LD_LIBRARY_PATH=${SBND_RECO1}/lib:${LD_LIBRARY_PATH:-}
export WIRECELL_PATH=${SBND_RECO1}/share/wirecell:${WIRECELL_PATH}

usage() { sed -n '2,30p' "$0"; exit "${1:-1}"; }
[ $# -ge 3 ] || usage

INPUT=$1; OUTROOT=$2; REALITY=$3; shift 3
case "$REALITY" in data|sim) ;; *) echo "ERROR: reality must be data|sim" >&2; exit 1;; esac
[ -r "$INPUT" ] || { echo "ERROR: no such reco1 file: $INPUT" >&2; exit 1; }

GSIZE=16; ONLY=""; FROM=img; TO=ql
while [ $# -gt 0 ]; do
    case "$1" in
        --size)   GSIZE=$2; shift 2;;
        --entries) ENTRIES_OVERRIDE=$2; shift 2;;
        --group)  ONLY=$2; shift 2;;
        --groups) ONLY=$2; shift 2;;
        --from)   FROM=$2; shift 2;;
        --to)     TO=$2; shift 2;;
        -h|--help) usage 0;;
        *) echo "ERROR: unknown argument: $1" >&2; usage;;
    esac
done

# M13: a fresh out_root, or one this script made.
if [ -e "$OUTROOT" ] && [ -n "$(ls -A "$OUTROOT" 2>/dev/null)" ] \
   && [ ! -f "$OUTROOT/.chain_group" ]; then
    echo "ERROR: $OUTROOT is not empty and was not created by this script." >&2
    echo "       Use a fresh out_root (CLAUDE.md M13)." >&2
    exit 1
fi
mkdir -p "$OUTROOT"
touch "$OUTROOT/.chain_group"
echo "$REALITY" > "$OUTROOT/.lineage_reality"

# Total entries in the art file -- read once.  Via the `root` CLI, not python:
# this environment has ROOT but no PyROOT module.  --entries overrides it.
if [ -n "${ENTRIES_OVERRIDE:-}" ]; then
    NENT=$ENTRIES_OVERRIDE
else
    NENT=$(root -l -b -q -e "std::unique_ptr<TFile> f(TFile::Open(\"$INPUT\")); TTree* t=(TTree*)f->Get(\"Events\"); printf(\"NENTRIES=%lld\\n\", t?t->GetEntries():-1);" 2>/dev/null \
           | sed -n 's/^NENTRIES=//p' | head -1)
fi
[ "${NENT:-0}" -gt 0 ] || { echo "ERROR: cannot read entry count from $INPUT (pass --entries N)" >&2; exit 1; }
NGROUP=$(( (NENT + GSIZE - 1) / GSIZE ))
echo "input=$INPUT entries=$NENT group_size=$GSIZE groups=$NGROUP reality=$REALITY"

if [ -n "$ONLY" ]; then
    IFS=, read -r -a GIDS <<< "$ONLY"
else
    GIDS=(); for ((k=0;k<NGROUP;k++)); do GIDS+=("$k"); done
fi

run_group() {
    local K=$1
    local GDIR="$OUTROOT/g$K"
    local BEG=$(( K * GSIZE ))
    mkdir -p "$GDIR"

    # ---- 1. reco1 -> the group's frames + opflash (one process) -------------
    if [ "$FROM" = img ] && [ ! -s "$GDIR/frames-dnn.tar.bz2" ]; then
        wire-cell -l stderr -l "$GDIR/wct_dump.log:info" -L info \
            --tla-str "input=$INPUT" \
            --tla-str "output_dir=$GDIR" \
            --tla-str "caf_offset_mode=product" \
            --tla-str "caf_offset_override=0" \
            --tla-str "entry=-1" \
            --tla-str "entry_begin=$BEG" \
            --tla-str "entry_count=$GSIZE" \
            -c "$SX/wct-reco1-dump.jsonnet" > "$GDIR/dump.stdout" 2>&1 \
            || { echo "[g$K] reco1 dump FAILED (see $GDIR/wct_dump.log)" >&2; return 1; }
    fi

    # The group's event ids, in archive order -- the order every downstream
    # archive must keep (see scripts/multi/make_group_pctree.py).
    tar tjf "$GDIR/frames-dnn.tar.bz2" \
        | sed -n 's/^frame_dnnsp_\([0-9][0-9]*\)\.npy$/\1/p' | awk '!seen[$0]++' \
        > "$GDIR/events.txt"
    local NEV; NEV=$(wc -l < "$GDIR/events.txt")
    echo "[g$K] entries [$BEG,$((BEG+GSIZE))) -> $NEV events"

    # ---- 2. imaging, whole group in one process ----------------------------
    # wct-img-all.jsonnet has no event TLA: hand it the group's frame archive
    # and its ClusterFileSinks key every member by cluster ident, so the four
    # npz hold the whole group.
    if [ "$FROM" != ql ]; then
        setarch x86_64 -R python3 "$AB/timecmd.py" "$GDIR/.img.time.meta" \
        wire-cell -l stderr -l "$GDIR/wct_img.log:debug" -L debug \
            --tla-str  "input=$GDIR/frames-dnn.tar.bz2" \
            --tla-code "anode_indices=[0,1]" \
            --tla-str  "output_dir=$GDIR" \
            -c wct-img-all.jsonnet > "$GDIR/img.stdout" 2>&1 \
            || { echo "[g$K] imaging FAILED (see $GDIR/wct_img.log)" >&2; return 1; }
    fi
    [ "$TO" = img ] && { echo "[g$K] stopped after imaging"; return 0; }

    # ---- 3. clustering + Q/L matching, whole group in one process ----------
    # multi_event: each event's Bee layers take their number from that event's
    # tensor ident.  rse_map: run/subrun per event, because a group can span
    # many runs and the job's run/subrun TLAs are one pair.
    python3 "$SX/scripts/multi/reco1_rse_map.py" \
        --opflash "$GDIR/opflash_apa0.tar.gz" --out "$GDIR/rse.json" \
        > "$GDIR/rse.log" 2>&1 || echo "{}" > "$GDIR/rse.json"
    local RUN0 SUB0
    read -r RUN0 SUB0 < <(python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
v=next(iter(d.values()),[0,0])
print(v[0], v[1])' "$GDIR/rse.json")

    setarch x86_64 -R python3 "$AB/timecmd.py" "$GDIR/.ql.time.meta" \
    wire-cell -l stderr -l "$GDIR/wct_ql.log:debug" -L debug \
        --tla-str  "input=$GDIR" \
        --tla-code "anode_indices=[0,1]" \
        --tla-str  "output_dir=$GDIR" \
        --tla-code "run=$RUN0" --tla-code "subrun=$SUB0" --tla-code "event=0" \
        --tla-str  "reality=$REALITY" \
        --tla-code "multi_event=true" \
        --tla-code "rse_map=$(cat "$GDIR/rse.json")" \
        --tla-str  "save_tensors=$GDIR/pctree-ql.tar.gz" \
        -c wct-clus-matching-perevt.jsonnet > "$GDIR/ql.stdout" 2>&1 \
        || { echo "[g$K] Q/L FAILED (see $GDIR/wct_ql.log)" >&2; return 1; }

    echo "[g$K] ok -> $GDIR"
    return 0
}

NJOBS=${SBND_MAX_JOBS:-4}
pids=(); fail=0
for K in "${GIDS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n 2>/dev/null || true; done
    ( run_group "$K" ) > "$OUTROOT/.g$K.log" 2>&1 &
    echo "  [start] group=$K  log: $OUTROOT/.g$K.log"
done
wait
for K in "${GIDS[@]}"; do
    grep -q "^\[g$K\] ok" "$OUTROOT/.g$K.log" || { echo "FAILED: group $K (see $OUTROOT/.g$K.log)" >&2; fail=1; }
done
echo "loadavg: $(cat /proc/loadavg)"
exit $fail
