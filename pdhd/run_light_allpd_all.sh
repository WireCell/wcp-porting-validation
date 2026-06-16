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
# The dir suffix is NOT the event number (runs index dirs 0,1,2..); the real
# event number is read from each dir's light-reco.log ("run R event N"), or its
# calib-evt<N>-*.json products.  Dirs with neither (empty/unprocessed stubs) are
# skipped.  Output goes to work/<padded>_allpd<event>/.
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
    # real event number per selected dir: calib-evt<N> if present, else dir suffix
    mapfile -t EVENTS < <(
        for d in "$PDHD_DIR/work/${RUN_PADDED}_"[0-9]*; do
            b=$(basename "$d"); [[ "$b" =~ ^${RUN_PADDED}_[0-9]+$ ]] || continue
            ev=$(sed -nE 's/.*[Ee]vent[ =:]+([0-9]+).*/\1/p' "$d"/light-reco.log 2>/dev/null | head -1)
            [ -z "$ev" ] && ev=$(ls "$d"/calib-evt*-*.json 2>/dev/null | head -1 | sed -nE 's#.*calib-evt([0-9]+)-.*#\1#p')
            [ -z "$ev" ] && continue   # no reliable event number -> skip stub dir
            echo "$ev"
        done | sort -n -u)
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
