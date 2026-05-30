#!/bin/bash
# Shared helper library for sbnd_xin/run_*.sh scripts.
# Source after setting SBND_DIR:
#   . "$SBND_DIR/_runlib.sh"

# Default (mc) event list.  Overridden by load_events <mode>, which derives the
# list from the mode's frames-dnn.tar.bz2 (so data ids 659242… are picked up).
SBND_EVENTS=(2 9 11 12 14 18 31 35 41 42)

# Mode (mc|data) → the per-event SP-frame source archive provided upstream.
sp_frames_archive() {
    echo "$SBND_DIR/input_files/input-10evt-${1}/frames-dnn.tar.bz2"
}

# Populate SBND_EVENTS (in archive order) from the mode's frames-dnn archive.
# Keeps the built-in mc list as a fallback if the archive is absent.
load_events() {
    local mode="${1:-mc}" archive
    archive=$(sp_frames_archive "$mode")
    if [ -s "$archive" ]; then
        mapfile -t SBND_EVENTS < <(tar tjf "$archive" | sed -n 's/^frame_dnnsp_\([0-9][0-9]*\)\.npy$/\1/p')
    fi
    [ "${#SBND_EVENTS[@]}" -gt 0 ] || {
        echo "ERROR: no events found for mode '$mode' ($archive)" >&2; return 1; }
}

# Extract one event's 5-member SP-frame subset from the mode archive into a
# fresh single-event archive at <dest>.  Always (re)extracts so imaging tracks
# the current frames-dnn.tar.bz2 (cheap: ~one event of npy).
ensure_sp_frames() {
    local mode="$1" evt_id="$2" dest="$3" src tmp
    src=$(sp_frames_archive "$mode")
    [ -s "$src" ] || { echo "ERROR: missing frames archive: $src" >&2; return 1; }
    tmp=$(mktemp -d /home/xqian/tmp/sbnd_spf_XXXXXX) || return 1
    tar -xjf "$src" -C "$tmp" --wildcards "*_${evt_id}.npy"
    ( cd "$tmp" && tar -cjf "$dest" *_"${evt_id}".npy )
    rm -rf "$tmp"
}

# Print the idx→EVT_ID mappings.  Called when the invoking script
# receives no positional arguments.
list_events() {
    echo "Available SBND events (1-based index → event ID):"
    for i in "${!SBND_EVENTS[@]}"; do
        printf '  idx %-3d → evt %d\n' "$((i+1))" "${SBND_EVENTS[$i]}"
    done
}

# Resolve a 1-based event index to the real event ID.
# Echoes the ID on stdout; on bad input writes the table to stderr and
# returns 1 (callers should check $? or let set -e handle it).
lookup_evt_id() {
    local idx="$1" n="${#SBND_EVENTS[@]}"
    if ! echo "$idx" | grep -qE '^[0-9]+$' || [ "$idx" -lt 1 ] || [ "$idx" -gt "$n" ]; then
        echo "ERROR: invalid event index '$idx' — must be 1..$n" >&2
        for i in "${!SBND_EVENTS[@]}"; do
            printf '    %d → %d\n' "$((i+1))" "${SBND_EVENTS[$i]}" >&2
        done
        return 1
    fi
    echo "${SBND_EVENTS[$((idx-1))]}"
}

# Prints the full sequence of valid event indices (1..N), one per line.
discover_event_indices() {
    seq 1 "${#SBND_EVENTS[@]}"
}

# ── Batch parallel runner ─────────────────────────────────────────────────────
#
# Usage in each script:
#   batch_init
#   for idx in $(discover_event_indices); do
#       batch_wait_slot
#       ( process_event "$idx" ) > "$logfile" 2>&1 &
#       BATCH_PIDS[$!]=$idx
#   done
#   batch_drain
#   batch_summary; exit $?
#
# Concurrency cap: ${SBND_MAX_JOBS:-$(nproc)}.

batch_init() {
    BATCH_OK=0
    BATCH_FAIL=0
    BATCH_FAIL_LIST=()
    BATCH_MAX=${SBND_MAX_JOBS:-$(nproc)}
    declare -gA BATCH_PIDS=()
}

# Reap one finished background job; update BATCH_OK / BATCH_FAIL.
# Requires bash >= 5.1 for 'wait -n -p VARNAME'.
_batch_reap_one() {
    local _pid _st
    wait -n -p _pid 2>/dev/null && _st=0 || _st=$?
    if [ "$_st" -eq 0 ]; then
        BATCH_OK=$((BATCH_OK + 1))
    else
        BATCH_FAIL=$((BATCH_FAIL + 1))
        BATCH_FAIL_LIST+=("${BATCH_PIDS[$_pid]:-pid:$_pid}")
    fi
    unset "BATCH_PIDS[$_pid]"
}

batch_wait_slot() {
    while [ "${#BATCH_PIDS[@]}" -ge "$BATCH_MAX" ]; do _batch_reap_one; done
}

batch_drain() {
    while [ "${#BATCH_PIDS[@]}" -gt 0 ]; do _batch_reap_one; done
}

# Print summary; returns 0 if at least one event succeeded, 1 otherwise.
batch_summary() {
    echo
    echo "===== batch summary ====="
    echo "  ok:      $BATCH_OK"
    echo "  failed:  $BATCH_FAIL"
    [ "$BATCH_FAIL" -gt 0 ] && echo "  failed events: ${BATCH_FAIL_LIST[*]}"
    [ "$BATCH_OK" -gt 0 ] && return 0 || return 1
}
