#!/bin/bash
# Run clustering for one event.
# Usage: ./run_clus_evt.sh [-a anode] [-s sel_tag] <run> <evt|all> [subrun]
#        ./run_clus_evt.sh               # list available runs
#
# EVT may be 'all' to run every discovered event in parallel (capped at nproc,
# override with PDHD_MAX_JOBS=N).  Events with missing inputs are skipped.
#
# Input:  work/<run>_<evt>[_sel<TAG>]/ (from imaging) or input_data event dir as fallback
# Output: work/<run>_<evt>[_sel<TAG>]/mabc-apa{N}.zip, mabc-all-apa.zip

set -e

PDHD_DIR=$(cd "$(dirname "$0")" && pwd)

WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}

# Preload tcmalloc for the clustering wire-cell process (~20% faster on
# busy events).  Unlocked by the get_closest_blob pointer-order fix: with
# it, glibc and tcmalloc outputs are byte-identical on the A/B event set.
# Disable with WCT_TCMALLOC=off.
TCMALLOC_SO=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
WC_PRELOAD=""
if [ "${WCT_TCMALLOC:-on}" = "on" ] && [ -f "$TCMALLOC_SO" ]; then
    WC_PRELOAD="LD_PRELOAD=$TCMALLOC_SO"
fi

. "$PDHD_DIR/_runlib.sh"

ANODE=""
SEL_TAG=""
# Charge-light (Q/L) matching before the final all-TPC clustering: opt-in
# (-q or PDHD_QLMATCH=1).  Default off => historical no-matching chain.
QLMATCH=${PDHD_QLMATCH:-0}
# -calib: also dump the per-drift-side Q/L hand-scan calibration JSONs
# (work/<run6>_<evt>/calib-evt<EVT>-group{02,13}.json) for the pdhd/ql_scan viewer.
# Implies Q/L matching; the matched mabc-*.zip output is byte-identical with/without it.
CALIB=0
OPDUMP=${PDHD_OPDUMP:-0}   # -op: dump the optical "op" bee instance (light + Q/L pred)
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -a) ANODE="$2"; shift 2 ;;
        -a*) ANODE="${1#-a}"; shift ;;
        -s) SEL_TAG="$2"; shift 2 ;;
        -s*) SEL_TAG="${1#-s}"; shift ;;
        -q) QLMATCH=1; shift ;;
        -calib|--calib) CALIB=1; QLMATCH=1; shift ;;
        -op|--op) OPDUMP=1; QLMATCH=1; shift ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"

if [ $# -eq 0 ]; then
    list_runs; exit 0
fi

if [ $# -lt 2 ]; then
    echo "Usage: $0 [-a anode] [-s sel_tag] [-q] [-calib] [-op] <run> <evt|all> [subrun]   (-q: Q/L matching; -calib: + hand-scan dumps; -op: + optical bee instance)" >&2
    exit 1
fi
RUN=$1
EVT=$2
SUBRUN=${3:-0}

RUN_STRIPPED=$(echo "$RUN" | sed 's/^0*//')
[ -z "$RUN_STRIPPED" ] && RUN_STRIPPED=0
RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")

find_evtdir() {
    local base="$PDHD_DIR/input_data"
    for rname in "run${RUN}" "run${RUN_PADDED}" "run${RUN_STRIPPED}"; do
        local rdir="$base/$rname"
        [ -d "$rdir" ] || continue
        for ename in "evt${EVT}" "evt_${EVT}"; do
            local cand="$rdir/$ename"
            if [ -d "$cand" ] && [ -n "$(ls -A "$cand" 2>/dev/null)" ]; then
                echo "$cand"; return 0
            fi
        done
        if ls "$rdir/clusters-apa-apa"*"-ms-active.tar.gz" >/dev/null 2>&1; then
            echo "$rdir"; return 0
        fi
    done
    return 1
}

process_event() {
    local RUN=$1 EVT=$2
    local RUN_STRIPPED RUN_PADDED WORKDIR EVTDIR CLUS_INPUT ANODE_CODE TAG_SUFFIX LOG
    local APA0_CLUS EVENT_NO
    RUN_STRIPPED=$(echo "$RUN" | sed 's/^0*//')
    [ -z "$RUN_STRIPPED" ] && RUN_STRIPPED=0
    RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")

    if [ -n "$SEL_TAG" ]; then
        WORKDIR="$PDHD_DIR/work/${RUN_PADDED}_${EVT}_${SEL_TAG}"
    else
        WORKDIR="$PDHD_DIR/work/${RUN_PADDED}_${EVT}"
    fi
    mkdir -p "$WORKDIR"

    CLUS_INPUT=""
    if ls "$WORKDIR/clusters-apa-apa"*"-ms-active.tar.gz" >/dev/null 2>&1; then
        CLUS_INPUT="$WORKDIR"
    else
        EVTDIR=$(find_evtdir) || EVTDIR=""
        if [ -n "$EVTDIR" ] && ls "$EVTDIR/clusters-apa-apa"*"-ms-active.tar.gz" >/dev/null 2>&1; then
            CLUS_INPUT="$EVTDIR"
        fi
    fi

    if [ -z "$CLUS_INPUT" ]; then
        echo "[skip] run=$RUN evt=$EVT: no cluster tarballs found" >&2
        return 2
    fi
    echo "Cluster input: $CLUS_INPUT"
    echo "Work dir:      $WORKDIR"

    APA0_CLUS=$(ls "$CLUS_INPUT/clusters-apa-apa"*"-ms-active.tar.gz" 2>/dev/null | head -1)
    EVENT_NO=$(tar tzf "$APA0_CLUS" | head -1 | sed -E 's/.*cluster_([0-9]+)_.*/\1/')
    if ! echo "$EVENT_NO" | grep -qE '^[0-9]+$'; then
        echo "ERROR: could not parse event number from $APA0_CLUS (got: '$EVENT_NO')" >&2
        return 1
    fi
    echo "Art event number: $EVENT_NO"

    if [ -n "$ANODE" ]; then
        ANODE_CODE="[$ANODE]"
        TAG_SUFFIX="_a${ANODE}"
    else
        ANODE_CODE="[0,1,2,3]"
        TAG_SUFFIX=""
    fi

    LOG="$WORKDIR/wct_clus_${RUN_PADDED}_${EVT}${TAG_SUFFIX}.log"
    echo "Log:           $LOG"

    cd "$PDHD_DIR"
    rm -f "$LOG"
    # Pre-compile the config with wcsonnet and feed wire-cell pure JSON,
    # with GOGC=off.  Rationale: an intermittent clustering SIGABRT was
    # identified as the embedded gojsonnet Go runtime GC crashing
    # ("traceback did not unwind completely").  libgojsonnet is hard-linked
    # so its runtime threads exist regardless of config format, but with a
    # pure-JSON config wire-cell never evaluates jsonnet (no Go heap) and
    # GOGC=off disables the Go collector entirely, including the 2-minute
    # periodic forced GC that is the crash vector.  Config is identical
    # (same pattern as imaging -P).
    # Per-event trigger offset for Q/L matching: read offset_us (~250) from the
    # converted opflash archive metadata and apply it DOWNSTREAM (matching geometry
    # + T0Correction x_t0cor).  Imaging stays offset-free (time_offset=0).  Stays 0
    # when not matching or when the archive/metadata is absent (bit-identical).
    TRIGGER_OFFSET_US=0
    if [ "$QLMATCH" = 1 ]; then
        OPFLASH_TAR="$CLUS_INPUT/opflash_pdhd-wct.tar.gz"
        if [ -f "$OPFLASH_TAR" ]; then
            TRIGGER_OFFSET_US=$(python3 - "$OPFLASH_TAR" <<'PY' || echo 0)
import sys, json, tarfile
off = 0.0
with tarfile.open(sys.argv[1]) as tf:
    for m in tf.getmembers():
        if m.name.endswith("_metadata.json"):
            md = json.loads(tf.extractfile(m).read())
            if "offset_us" in md:
                off = float(md["offset_us"])
                break
print(off)
PY
            echo "Trigger offset: ${TRIGGER_OFFSET_US} us (from $(basename "$OPFLASH_TAR"))"
        else
            echo "[warn] $OPFLASH_TAR not found; trigger_offset_us=0" >&2
        fi
    fi

    local CFG_JSON="$WORKDIR/.wct-clus${TAG_SUFFIX}.json"
    wcsonnet \
        -A "input=${CLUS_INPUT}" \
        -S "anode_indices=${ANODE_CODE}" \
        -A "output_dir=${WORKDIR}" \
        -S "run=${RUN_STRIPPED}" \
        -S "subrun=${SUBRUN}" \
        -S "event=${EVENT_NO}" \
        -S "do_qlmatch=$([ "$QLMATCH" = 1 ] && echo true || echo false)" \
        -S "calib=$([ "$CALIB" = 1 ] && echo true || echo false)" \
        -S "save_opflash=$([ "$OPDUMP" = 1 ] && echo true || echo false)" \
        -S "trigger_offset_us=${TRIGGER_OFFSET_US}" \
        -o "$CFG_JSON" wct-clustering.jsonnet
    if [ ! -s "$CFG_JSON" ]; then
        echo "ERROR: wcsonnet failed to compile wct-clustering.jsonnet" >&2
        return 1
    fi
    env $WC_PRELOAD GOGC=off wire-cell \
        -l stderr \
        -l "${LOG}:debug" \
        -L debug \
        -c "$CFG_JSON"
    rm -f "$CFG_JSON"

    echo "Clustering done -> $WORKDIR"
}

mkdir -p "$PDHD_DIR/work"
if [ "$EVT" = "all" ]; then
    batch_init
    mapfile -t _events < <(discover_events "$RUN" "$RUN_PADDED")
    if [ ${#_events[@]} -eq 0 ]; then
        echo "no events found for run=$RUN under input_data/ or work/" >&2; exit 1
    fi
    echo "Found ${#_events[@]} event(s) for run=$RUN: ${_events[*]}"
    echo "Parallel jobs: $BATCH_MAX"
    for _e in "${_events[@]}"; do
        _blogfile="$PDHD_DIR/work/.batch_clus_${RUN_PADDED}_${_e}.log"
        batch_wait_slot
        ( process_event "$RUN" "$_e" ) > "$_blogfile" 2>&1 &
        BATCH_PIDS[$!]=$_e
        echo "  [start] evt=$_e  log: $_blogfile"
    done
    batch_drain
    batch_summary
    exit $?
else
    ( process_event "$RUN" "$EVT" )
    exit $?
fi
