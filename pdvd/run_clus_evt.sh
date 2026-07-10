#!/bin/bash
# Run clustering for one event.
# Usage: ./run_clus_evt.sh [-a anode] [-s sel_tag] [-noq] [-calib] [-op] <run> <evt|all> [subrun]
#        ./run_clus_evt.sh               # list available runs
#
# Q/L (charge-light) matching runs by DEFAULT; -noq (or PDVD_QLMATCH=0) disables it.
# An event with no converted light (work/<RUN6>_light<EVENTNO>/opflash_pdvd-wct.tar.gz,
# looked up by the ART EVENT NUMBER parsed from the cluster tarball — the charge work
# dirs are INDEX-named) auto-falls back to the no-matching chain, so 'evt all' never
# fails on a light-less event (e.g. run 039324, no raw light staged).
#   -calib          also dump the hand-scan calib JSON (calib-evt<EVENTNO>.json)
#   -op / -noop     optical "op" bee instance (default ON when matching)
#   PDVD_LIGHT_MODEL=semi        semi-analytical visibility backend (default library)
#   PDVD_TRIGGER_OFFSET_US=<us>  override the light<->charge time-base offset
#   PDVD_QL_DIAG=1               offset-calibration diagnostic mode: containment off,
#                                flash_minPE=100, trigger offset forced 0
# The light<->charge offset = opflash metadata offset_us + the per-run calibrated
# constant from data/ql_trigger_offset.txt ("<run> <offset_us>" lines), unless
# overridden by PDVD_TRIGGER_OFFSET_US.
#
# EVT may be 'all' to run every discovered event in parallel (capped at nproc,
# override with PDVD_MAX_JOBS=N).  Events with missing inputs are skipped.
#
# Input:  work/<run>_<evt>[_sel<TAG>]/ (from imaging) or input_data event dir as fallback
# Output: work/<run>_<evt>[_sel<TAG>]/mabc-anode{N}.zip, mabc-group0123.zip, mabc-group4567.zip, mabc-all-apa.zip

set -e

PDVD_DIR=$(cd "$(dirname "$0")" && pwd)

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

. "$PDVD_DIR/_runlib.sh"

ANODE=""
SEL_TAG=""
# Charge-light (Q/L) matching before the final all-TPC clustering: ON by default
# (-noq or PDVD_QLMATCH=0 forces the historical no-matching chain).  Per-event,
# matching auto-disables when the converted light is absent (see QLMATCH_EVT).
QLMATCH=${PDVD_QLMATCH:-1}
CALIB=0
OPDUMP=${PDVD_OPDUMP:-1}   # optical "op" bee instance; default ON, -noop to disable
_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -a) ANODE="$2"; shift 2 ;;
        -a*) ANODE="${1#-a}"; shift ;;
        -s) SEL_TAG="$2"; shift 2 ;;
        -s*) SEL_TAG="${1#-s}"; shift ;;
        -q) QLMATCH=1; shift ;;
        -noq|--no-qlmatch) QLMATCH=0; shift ;;
        -calib|--calib) CALIB=1; QLMATCH=1; shift ;;
        -op|--op) OPDUMP=1; QLMATCH=1; shift ;;
        -noop|--no-op) OPDUMP=0; shift ;;
        *) _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]}"

if [ $# -eq 0 ]; then
    list_runs; exit 0
fi

if [ $# -lt 2 ]; then
    echo "Usage: $0 [-a anode] [-s sel_tag] [-noq] [-calib] [-noop] <run> <evt|all> [subrun]" >&2
    exit 1
fi
RUN=$1
EVT=$2
SUBRUN=${3:-0}

RUN_STRIPPED=$(echo "$RUN" | sed 's/^0*//')
[ -z "$RUN_STRIPPED" ] && RUN_STRIPPED=0
RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")

find_evtdir() {
    local base="$PDVD_DIR/input_data"
    for rname in "run${RUN}" "run${RUN_PADDED}" "run${RUN_STRIPPED}"; do
        local rdir="$base/$rname"
        [ -d "$rdir" ] || continue
        for ename in "evt${EVT}" "evt_${EVT}"; do
            local cand="$rdir/$ename"
            if [ -d "$cand" ] && [ -n "$(ls -A "$cand" 2>/dev/null)" ]; then
                echo "$cand"; return 0
            fi
        done
        if ls "$rdir/clusters-apa-anode"*"-ms-active.tar.gz" >/dev/null 2>&1; then
            echo "$rdir"; return 0
        fi
    done
    return 1
}

process_event() {
    local RUN=$1 EVT=$2
    local RUN_STRIPPED RUN_PADDED WORKDIR EVTDIR CLUS_INPUT ANODE_CODE TAG_SUFFIX LOG
    local ANODE0_CLUS EVENT_NO
    RUN_STRIPPED=$(echo "$RUN" | sed 's/^0*//')
    [ -z "$RUN_STRIPPED" ] && RUN_STRIPPED=0
    RUN_PADDED=$(printf '%06d' "$RUN_STRIPPED")

    if [ -n "$SEL_TAG" ]; then
        WORKDIR="$PDVD_DIR/work/${RUN_PADDED}_${EVT}_${SEL_TAG}"
    else
        WORKDIR="$PDVD_DIR/work/${RUN_PADDED}_${EVT}"
    fi
    mkdir -p "$WORKDIR"

    CLUS_INPUT=""
    if ls "$WORKDIR/clusters-apa-anode"*"-ms-active.tar.gz" >/dev/null 2>&1; then
        CLUS_INPUT="$WORKDIR"
    else
        EVTDIR=$(find_evtdir) || EVTDIR=""
        if [ -n "$EVTDIR" ] && ls "$EVTDIR/clusters-apa-anode"*"-ms-active.tar.gz" >/dev/null 2>&1; then
            CLUS_INPUT="$EVTDIR"
        fi
    fi

    if [ -z "$CLUS_INPUT" ]; then
        echo "[skip] run=$RUN evt=$EVT: no cluster tarballs found" >&2
        return 2
    fi
    echo "Cluster input: $CLUS_INPUT"
    echo "Work dir:      $WORKDIR"

    ANODE0_CLUS=$(ls "$CLUS_INPUT/clusters-apa-anode"*"-ms-active.tar.gz" 2>/dev/null | head -1)
    EVENT_NO=$(tar tzf "$ANODE0_CLUS" | head -1 | sed -E 's/.*cluster_([0-9]+)_.*/\1/')
    if ! echo "$EVENT_NO" | grep -qE '^[0-9]+$'; then
        echo "ERROR: could not parse event number from $ANODE0_CLUS (got: '$EVENT_NO')" >&2
        return 1
    fi
    echo "Art event number: $EVENT_NO"

    # Q/L matching needs this event's converted light.  The charge work dirs are
    # INDEX-named (work/<RUN6>_<idx>) but the light chain's dirs are keyed by the
    # ART EVENT NUMBER (work/<RUN6>_light<EVENTNO>) — bridge via EVENT_NO above.
    local QLMATCH_EVT=$QLMATCH
    local OPFLASH_TAR="$PDVD_DIR/work/${RUN_PADDED}_light${EVENT_NO}/opflash_pdvd-wct.tar.gz"
    if [ "$QLMATCH_EVT" = 1 ] && [ ! -f "$OPFLASH_TAR" ]; then
        echo "[note] no $OPFLASH_TAR -> skipping Q/L matching for this event" >&2
        QLMATCH_EVT=0
    fi

    # Light<->charge time-base offset: opflash metadata offset_us (0 until the
    # light chain bakes a calibrated value) + the per-run constant from
    # data/ql_trigger_offset.txt; PDVD_TRIGGER_OFFSET_US overrides everything;
    # the diagnostic mode (PDVD_QL_DIAG=1) forces 0.  Also guard that the
    # opflash metadata 'event' matches the charge EVENT_NO.
    TRIGGER_OFFSET_US=0
    READOUT_NTICKS=10000
    if [ "$QLMATCH_EVT" = 1 ]; then
        local META_OFF OPFLASH_EVENT
        META_OFF=$(python3 - "$OPFLASH_TAR" <<'PY' || echo 0
import sys, json, tarfile
off = 0.0
evt = ""
with tarfile.open(sys.argv[1]) as tf:
    for m in tf.getmembers():
        if m.name.endswith("_metadata.json"):
            md = json.loads(tf.extractfile(m).read())
            if "offset_us" in md:
                off = float(md["offset_us"])
            if "event" in md:
                evt = int(md["event"])
            break
print(off, evt)
PY
)
        OPFLASH_EVENT=$(echo "$META_OFF" | awk '{print $2}')
        META_OFF=$(echo "$META_OFF" | awk '{print $1}')
        if [ -n "$OPFLASH_EVENT" ] && [ "$OPFLASH_EVENT" != "$EVENT_NO" ]; then
            echo "ERROR: charge/light event mismatch: charge art_event=$EVENT_NO but opflash event=$OPFLASH_EVENT ($OPFLASH_TAR)." >&2
            return 1
        fi
        local RUN_OFF=0
        if [ -f "$PDVD_DIR/data/ql_trigger_offset.txt" ]; then
            RUN_OFF=$(awk -v r="$RUN_STRIPPED" '$1+0==r+0{print $2}' "$PDVD_DIR/data/ql_trigger_offset.txt" | head -1)
            RUN_OFF=${RUN_OFF:-0}
        fi
        TRIGGER_OFFSET_US=$(python3 -c "print(${META_OFF:-0} + ${RUN_OFF:-0})")
        if [ -n "${PDVD_TRIGGER_OFFSET_US:-}" ]; then
            TRIGGER_OFFSET_US="$PDVD_TRIGGER_OFFSET_US"
        fi
        if [ "${PDVD_QL_DIAG:-0}" = 1 ]; then
            TRIGGER_OFFSET_US=0
        fi
        echo "Trigger offset: ${TRIGGER_OFFSET_US} us (metadata ${META_OFF} + run table ${RUN_OFF})"

        # Real readout window (post-resample SP frame length, 10000 ticks x
        # 0.5 us = 5 ms) for the window-truncation flag.
        local _SPF
        _SPF=$(ls "$CLUS_INPUT"/protodune-sp-dnnroi-frames-anode*.tar.bz2 2>/dev/null | head -1)
        if [ -n "$_SPF" ]; then
            local _NT
            _NT=$(python3 - "$_SPF" <<'PY'
import sys, tarfile, io, numpy as np
with tarfile.open(sys.argv[1]) as tf:
    name = next(m.name for m in tf.getmembers() if m.name.startswith("frame_"))
    print(np.load(io.BytesIO(tf.extractfile(name).read())).shape[1])
PY
)
            READOUT_NTICKS=${_NT:-10000}
            echo "Readout window: ${READOUT_NTICKS} ticks (from $(basename "$_SPF"))"
        fi
    fi

    if [ -n "$ANODE" ]; then
        ANODE_CODE="[$ANODE]"
        TAG_SUFFIX="_a${ANODE}"
    else
        ANODE_CODE="[0,1,2,3,4,5,6,7]"
        TAG_SUFFIX=""
    fi

    LOG="$WORKDIR/wct_clus_${RUN_PADDED}_${EVT}${TAG_SUFFIX}.log"
    echo "Log:           $LOG"

    cd "$PDVD_DIR"
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
    local CFG_JSON="$WORKDIR/.wct-clus${TAG_SUFFIX}.json"
    # Q/L diagnostic mode (offset calibration): containment off (meaningless
    # until the time base is measured) + a higher PE floor to bound the
    # un-pruned bundle count.
    local QL_CONTAIN=true QL_MINPE=25
    if [ "${PDVD_QL_DIAG:-0}" = 1 ]; then
        QL_CONTAIN=false; QL_MINPE=100
    fi
    wcsonnet \
        -A "input=${CLUS_INPUT}" \
        -S "anode_indices=${ANODE_CODE}" \
        -A "output_dir=${WORKDIR}" \
        -S "run=${RUN_STRIPPED}" \
        -S "subrun=${SUBRUN}" \
        -S "event=${EVENT_NO}" \
        -S "stepped_center_fallback=${PDVD_STEPPED_CENTER_FALLBACK:-false}" \
        -S "do_qlmatch=$([ "$QLMATCH_EVT" = 1 ] && echo true || echo false)" \
        -A "opflash_input=${OPFLASH_TAR}" \
        -S "calib=$([ "$CALIB" = 1 ] && echo true || echo false)" \
        -S "save_opflash=$([ "$OPDUMP" = 1 ] && [ "$QLMATCH_EVT" = 1 ] && echo true || echo false)" \
        -S "trigger_offset_us=${TRIGGER_OFFSET_US:-0}" \
        -S "readout_window_ticks=${READOUT_NTICKS:-10000}" \
        -A "light_model=${PDVD_LIGHT_MODEL:-library}" \
        -S "ql_require_containment=${QL_CONTAIN}" \
        -S "ql_flash_minpe=${QL_MINPE}" \
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

mkdir -p "$PDVD_DIR/work"
if [ "$EVT" = "all" ]; then
    batch_init
    mapfile -t _events < <(discover_events "$RUN" "$RUN_PADDED")
    if [ ${#_events[@]} -eq 0 ]; then
        echo "no events found for run=$RUN under input_data/ or work/" >&2; exit 1
    fi
    echo "Found ${#_events[@]} event(s) for run=$RUN: ${_events[*]}"
    echo "Parallel jobs: $BATCH_MAX"
    for _e in "${_events[@]}"; do
        _blogfile="$PDVD_DIR/work/.batch_clus_${RUN_PADDED}_${_e}.log"
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
