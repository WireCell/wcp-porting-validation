#!/bin/bash
# Reprocess the all-PD WCT light reconstruction for every selected event of the
# given PDHD runs.  Each event is run through run_light_allpd_evt.sh, which
# (re)builds the snippet + full-stream decoana from raw when absent and runs the
# all-PD chain with the production config -- i.e. the PDHD defaults, which now
# include the 14-bit ADC-saturation veto (detect_saturation/veto_saturation,
# saturation_pad=1024) and the per-PD fired threshold (min_fired_pe=1.0).  See
# docs/run29107-evt1015-light-anomaly.md.
#
# "Selected events" of a run = its numeric charge work dirs work/<padded>_<N>.
# Output overwrites the production all-PD dirs work/<padded>_allpd<N>/.
#
# Usage:
#   ./run_light_allpd_all.sh [-n] [run ...]
#     -n        dry run: list the events that would be processed, do nothing
#     run ...   runs to process (default: 27305 27980 28084 29107)

set -u
PDHD_DIR=$(cd "$(dirname "$0")" && pwd)

DRY=0
while getopts "n" opt; do case $opt in n) DRY=1;; *) exit 1;; esac; done
shift $((OPTIND-1))

RUNS=("$@"); [ ${#RUNS[@]} -eq 0 ] && RUNS=(27305 27980 28084 29107)

total=0 ok=0 fail=0
for RUN in "${RUNS[@]}"; do
    RUN_PADDED=$(printf "%06d" "$RUN")
    mapfile -t EVENTS < <(ls -d "$PDHD_DIR/work/${RUN_PADDED}_"[0-9]* 2>/dev/null \
        | sed -E "s#.*/${RUN_PADDED}_([0-9]+)\$#\1#" | grep -E '^[0-9]+$' | sort -n)
    echo "=== run $RUN: ${#EVENTS[@]} selected events ==="
    for EV in "${EVENTS[@]}"; do
        total=$((total+1))
        if [ "$DRY" -eq 1 ]; then echo "  would process run $RUN event $EV"; continue; fi
        echo "--- [$total] run $RUN event $EV $(date +%H:%M:%S) ---"
        if "$PDHD_DIR/run_light_allpd_evt.sh" "$RUN" "$EV" >/dev/null 2>&1; then
            ok=$((ok+1))
        else
            fail=$((fail+1)); echo "  WARN: run $RUN event $EV failed"
        fi
    done
done
echo "done: $total events, $ok ok, $fail failed"
