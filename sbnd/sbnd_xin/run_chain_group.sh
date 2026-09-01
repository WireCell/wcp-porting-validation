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
#   --fsproduct T  art InputTag of the FrameShiftInfo product for the reco1
#                dump's caf_offset_mode=product.  Default is the standard
#                '...__FRAMESHIFT.'; the NCpi0 sideband file carries
#                '...__FILTERFRAMESHIFT.' instead (doc 71 sec 3), and the
#                dump ABORTS rather than silently falling back.
#   --gbase N    offset the g<K> directory NAMES by N (entry ranges are
#                unaffected).  A sample split across several reco1 files --
#                mcp2k is two 1000-entry parts -- is one invocation per
#                file into ONE out_root; without an offset the second
#                invocation's g0 lands on the first's, and the
#                skip-if-present guard would then reuse the WRONG frames.
#   --layout L   'group' (default) keeps one archive set per group.
#                'perevt' ALSO writes the per-event layout a per-event job
#                writes -- <out>/evt<ID>/icluster-*.npz and
#                <out>/ql_evt<ID>/{pctree-evt<ID>.tar.gz, mabc*.zip,
#                opflash_apa*.tar.gz} -- so the products are a file-for-file
#                drop-in for work-<s>-ql0819 and can be gated with the
#                existing tools.  Uses the Q/L job's evt_subdir TLA plus
#                scripts/multi/split_group_products.py (doc 81 round 1).
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

GSIZE=16; ONLY=""; FROM=img; TO=ql; LAYOUT=group
# Empty => the TLA is not passed at all, so wct-reco1-dump.jsonnet omits the
# key (its own default is '') and the C++ default applies -- i.e. exactly
# what this runner did before --fsproduct existed.
FSPRODUCT=''
GBASE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --size)   GSIZE=$2; shift 2;;
        --entries) ENTRIES_OVERRIDE=$2; shift 2;;
        --group)  ONLY=$2; shift 2;;
        --groups) ONLY=$2; shift 2;;
        --from)   FROM=$2; shift 2;;
        --to)     TO=$2; shift 2;;
        --layout) LAYOUT=$2; shift 2;;
        --gbase)  GBASE=$2; shift 2;;
        --fsproduct) FSPRODUCT=$2; shift 2;;
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

# doc 82 -- compile the config in a SEPARATE, short-lived wcsonnet process and
# hand wire-cell plain JSON.  Passing wire-cell a .jsonnet makes it run
# gojsonnet IN-PROCESS, which leaves a 64-thread Go runtime alive for the whole
# job.  Both other drivers in this tree already precompile for exactly that
# reason (run_ql_evt.sh:552, run_pr_chain_batch.sh:1617/1751) -- doc pr/97
# sec 5, a SIGSEGV at ~120 s of process life -- and THIS driver was the only one
# left that did not, though a group job runs far longer than 120 s.
#
# It also made stage A non-reproducible: with the Go runtime alive the Q/L
# stage's output varies run to run on marginal events, which is where doc 81
# sec 7.1's seven events came from.  Precompiled runs are deterministic and
# reproduce work-<s>-ql0819 exactly (doc 82 part 2).
#
# The SAME path string goes to wcsonnet and to wire-cell, so the two can never
# resolve different files.  The entry points are named by absolute path rather
# than bare, which is what the bare name resolved to anyway -- via the CWD, so
# the runner silently depended on being invoked from sbnd_xin.
#
# DEFAULT ON since doc 81 round 2 (owner decision), matching run_ql_evt.sh and
# run_pr_chain_batch.sh -- this driver is no longer the odd one out.
#
# Doc 82 part 3 shipped it OFF on the ground that precompiling changes the
# process's allocation history and part 2 had shown the Q/L answer on a bistable
# event to be a function of exactly that: its own ON-vs-OFF test differed on
# 286191.  Doc 82 round 3 then removed that dependence at the source
# (QLMatching::rescue_empty_flashes walking flash_bundles_map in heap order,
# toolkit 95c10cd1), so the condition it attached -- "needs its own byte-identity
# gate on the standard manifest" -- is now DISCHARGED rather than waived:
#
#   32 events over 2 samples, all 7 known-bistable ones plus controls, ON vs OFF
#   at 95c10cd1: 128/128 archives member-content IDENTICAL.  Against
#   work-<s>-grp0825 both arms agree exactly -- mcp1k 64/64 identical, mcp2k the
#   same 4 archives of 53793 alone, which is round 3's own documented mover.
#   (doc 81 sec 10.4)
#
# Buys 38 threads -> 7 and closes the doc pr/97 SIGSEGV hazard, which this was
# the only driver still exposed to.  SBND_PRECOMPILE_CFG=0 restores the
# in-process gojsonnet path byte for byte.
#
# NOTE the gate above covers the Q/L stage only (repro_ql_nondet.sh runs
# --from ql); this function is also called for the dump and imaging stages, so
# a campaign should pilot its smallest sample first and check imaging against
# work-img-<s> before the rest of the events run.
# Sets the caller's _CFG from the caller's _TLA, and empties _TLA on success.
precompile_cfg() {           # precompile_cfg <tag> <jsonnet> <outjson>
    local _tag=$1 _jsonnet=$2 _out=$3
    _CFG=(-c "$_jsonnet")
    [ "${SBND_PRECOMPILE_CFG:-1}" = 1 ] || return 0
    rm -f "$_out"
    if wcsonnet "${_TLA[@]}" -o "$_out" "$_jsonnet" > "$_out.log" 2>&1; then
        _CFG=(-c "$_out")
        _TLA=()
    else
        echo "[$_tag] WARN: wcsonnet failed -- falling back to in-process jsonnet" >&2
    fi
}

run_group() {
    local K=$1
    local GDIR="$OUTROOT/g$(( K + GBASE ))"
    local BEG=$(( K * GSIZE ))
    mkdir -p "$GDIR"

    # ---- 1. reco1 -> the group's frames + opflash (one process) -------------
    # doc 81 round 1: a FAILED dump leaves a short, unreadable frames archive
    # behind, and plain `-s` counts it as done -- the retry then skips the dump
    # and every later stage runs on zero events.  Require the archive to
    # actually list frames, and delete it when the dump fails.
    if [ -e "$GDIR/frames-dnn.tar.bz2" ] \
       && ! tar tjf "$GDIR/frames-dnn.tar.bz2" 2>/dev/null | grep -q '^frame_dnnsp_'; then
        echo "[g$K] discarding an unreadable frames-dnn.tar.bz2 from an earlier failed dump" >&2
        rm -f "$GDIR/frames-dnn.tar.bz2"
    fi
    if [ "$FROM" = img ] && [ ! -s "$GDIR/frames-dnn.tar.bz2" ]; then
        local FS_TLA=(); [ -n "$FSPRODUCT" ] && FS_TLA=(--tla-str "frameshift_product=$FSPRODUCT")
        local -a _TLA=(--tla-str "input=$INPUT"
                       --tla-str "output_dir=$GDIR"
                       --tla-str "caf_offset_mode=product"
                       --tla-str "caf_offset_override=0"
                       "${FS_TLA[@]}"
                       --tla-str "entry=-1"
                       --tla-str "entry_begin=$BEG"
                       --tla-str "entry_count=$GSIZE")
        local -a _CFG=()
        precompile_cfg "g$K" "$SX/wct-reco1-dump.jsonnet" "$GDIR/.wct-cfg-dump.json"
        wire-cell -l stderr -l "$GDIR/wct_dump.log:info" -L info \
            "${_TLA[@]}" "${_CFG[@]}" > "$GDIR/dump.stdout" 2>&1 \
            || { rm -f "$GDIR/frames-dnn.tar.bz2"
                 echo "[g$K] reco1 dump FAILED (see $GDIR/wct_dump.log)" >&2; return 1; }
    fi

    # The group's event ids, in archive order -- the order every downstream
    # archive must keep (see scripts/multi/make_group_pctree.py).
    # Derive the group's event ids from the frame archive when we have one.
    # Under --from ql there may be no frames archive at all (an imaging-only
    # checkpoint, or a hand-assembled group): keep an existing non-empty
    # events.txt rather than truncating it to nothing.  doc 76 noted this ran
    # unconditionally; doc 81 makes it conditional.
    if tar tjf "$GDIR/frames-dnn.tar.bz2" 2>/dev/null | grep -q '^frame_dnnsp_'; then
        tar tjf "$GDIR/frames-dnn.tar.bz2" \
            | sed -n 's/^frame_dnnsp_\([0-9][0-9]*\)\.npy$/\1/p' | awk '!seen[$0]++' \
            > "$GDIR/events.txt"
    elif [ ! -s "$GDIR/events.txt" ]; then
        echo "[g$K] no frames archive and no events.txt in $GDIR" >&2
        : > "$GDIR/events.txt"
    fi
    local NEV; NEV=$(wc -l < "$GDIR/events.txt")
    echo "[g$K] entries [$BEG,$((BEG+GSIZE))) -> $NEV events"
    [ "$NEV" -gt 0 ] || { echo "[g$K] no events in the group -- refusing to run downstream stages" >&2; return 1; }

    # ---- 2. imaging, whole group in one process ----------------------------
    # wct-img-all.jsonnet has no event TLA: hand it the group's frame archive
    # and its ClusterFileSinks key every member by cluster ident, so the four
    # npz hold the whole group.
    if [ "$FROM" != ql ]; then
        local -a _TLA=(--tla-str  "input=$GDIR/frames-dnn.tar.bz2"
                       --tla-code "anode_indices=[0,1]"
                       --tla-str  "output_dir=$GDIR")
        local -a _CFG=()
        precompile_cfg "g$K" "$SX/wct-img-all.jsonnet" "$GDIR/.wct-cfg-img.json"
        setarch x86_64 -R python3 "$AB/timecmd.py" "$GDIR/.img.time.meta" \
        wire-cell -l stderr -l "$GDIR/wct_img.log:debug" -L debug \
            "${_TLA[@]}" "${_CFG[@]}" > "$GDIR/img.stdout" 2>&1 \
            || { echo "[g$K] imaging FAILED (see $GDIR/wct_img.log)" >&2; return 1; }
    fi
    [ "$TO" = img ] && { echo "[g$K] ok (stopped after imaging)"; return 0; }

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

    # doc 81 round 1: in perevt layout each event's Q/L products go to
    # <out>/ql_evt<ID>/ via the evt_subdir TLA -- the same names and the same
    # places a per-event job writes.  The sinks do not create directories, so
    # the runner must (same as run_pr_chain_batch.sh).
    # doc 87 -- QL output knobs.  Unset/empty => no TLA => byte-identical; only
    # the literal 0 turns an output off, so a typo fails safe.
    #   SBND_QL_PERFACE_BEE=0  no ql_evt<ID>/mabc-apa{0,1}-face0.zip (~0.24 GB/1k).
    #     Read by NOTHING in either repo -- the PR chain never opens them.
    #   SBND_QL_ALLAPA_BEE=0   no ql_evt<ID>/mabc-all-apa.zip        (~0.47 GB/1k).
    #     Read only by nusel_extract.py --qlbee, an optional cross-check that
    #     group mode already skips (QLBEE="" in run_pr_chain_batch.sh).
    local -a QL_BEE_TLA=()
    [ "${SBND_QL_PERFACE_BEE:-1}" = 0 ] && QL_BEE_TLA+=(--tla-code "perface_bee=false")
    [ "${SBND_QL_ALLAPA_BEE:-1}" = 0 ] && QL_BEE_TLA+=(--tla-code "allapa_bee=false")
    local QL_TLA=() QL_OUT="$GDIR" QL_TENSORS="$GDIR/pctree-ql.tar.gz"
    if [ "$LAYOUT" = perevt ]; then
        QL_OUT="$OUTROOT"
        QL_TENSORS="$OUTROOT/ql_evt%1%/pctree-evt%1%.tar.gz"
        QL_TLA=(--tla-str "evt_subdir=ql_evt%1%")
        local _e
        while read -r _e; do [ -n "$_e" ] && mkdir -p "$OUTROOT/ql_evt$_e"; done < "$GDIR/events.txt"
    fi

    local -a _TLA=(--tla-str  "input=$GDIR"
                   --tla-code "anode_indices=[0,1]"
                   --tla-str  "output_dir=$QL_OUT"
                   "${QL_TLA[@]}"
                   --tla-code "run=$RUN0" --tla-code "subrun=$SUB0" --tla-code "event=0"
                   --tla-str  "reality=$REALITY"
                   --tla-code "multi_event=true"
                   --tla-code "rse_map=$(cat "$GDIR/rse.json")"
                   --tla-str  "save_tensors=$QL_TENSORS"
                   "${QL_BEE_TLA[@]}")
    local -a _CFG=()
    precompile_cfg "g$K" "$SX/wct-clus-matching-perevt.jsonnet" "$GDIR/.wct-cfg-ql.json"
    setarch x86_64 -R python3 "$AB/timecmd.py" "$GDIR/.ql.time.meta" \
    wire-cell -l stderr -l "$GDIR/wct_ql.log:debug" -L debug \
        "${_TLA[@]}" "${_CFG[@]}" > "$GDIR/ql.stdout" 2>&1 \
        || { echo "[g$K] Q/L FAILED (see $GDIR/wct_ql.log)" >&2; return 1; }

    if [ "$LAYOUT" = perevt ]; then
        python3 "$SX/scripts/multi/split_group_products.py" "$GDIR" "$OUTROOT" \
            > "$GDIR/split.log" 2>&1 \
            || { echo "[g$K] product split FAILED (see $GDIR/split.log)" >&2; return 1; }
    fi

    # doc 87: SBND_QL_KEEP_ICLUSTER=0 drops the imaging->Q/L handoff, the single
    # largest class on disk (~4.80 GB/1000 evt).  The Q/L job above is its ONLY
    # consumer -- the PR chain's compiled config contains no icluster/.npz
    # reference at all, its only input key is the pctree -- so once Q/L has
    # succeeded they are dead weight.  Deleted only AFTER that success: this runs
    # past every `return 1` above, so a failed group keeps its inputs for the
    # post-mortem.  Regenerating them means re-running imaging (M11: never borrow
    # someone else's npz).
    if [ "${SBND_QL_KEEP_ICLUSTER:-1}" = 0 ]; then
        # SCOPE: this group's OWN events only.  run_group() runs SBND_MAX_JOBS
        # (default 4) at a time, so a sweep over $OUTROOT at maxdepth 2 would
        # reach into a CONCURRENT group's evt<ID>/ -- deleting its products,
        # possibly while its split_group_products.py is still writing them.
        # events.txt is this group's list and is already read above.
        # ql_evt<ID>/icluster-*.npz are SYMLINKS into evt<ID>/, so the real
        # files go first and the now-broken links (-xtype l) after -- leaving
        # dangling links behind would make a later run look like it has inputs.
        local _n=0 _e _d
        # the group dir's own copies first (one set for the whole group)
        _n=$(find "$GDIR" -maxdepth 1 -name 'icluster-apa*.npz' -type f 2>/dev/null | wc -l)
        find "$GDIR" -maxdepth 1 -name 'icluster-apa*.npz' -type f -delete 2>/dev/null
        while read -r _e; do
            [ -n "$_e" ] || continue
            for _d in "$OUTROOT/evt$_e" "$OUTROOT/ql_evt$_e"; do
                [ -d "$_d" ] || continue
                _n=$((_n + $(find "$_d" -maxdepth 1 -name 'icluster-apa*.npz' -type f 2>/dev/null | wc -l)))
                find "$_d" -maxdepth 1 -name 'icluster-apa*.npz' -type f -delete 2>/dev/null
                find "$_d" -maxdepth 1 -name 'icluster-apa*.npz' -xtype l -delete 2>/dev/null
            done
        done < "$GDIR/events.txt"
        echo "[g$K] doc 87: removed $_n icluster npz + their symlinks (SBND_QL_KEEP_ICLUSTER=0)"
    fi

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
