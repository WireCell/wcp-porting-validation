#!/bin/bash
# Batch WCT light reconstruction: every event of one PDVD light run.
# Usage: ./run_light_all.sh [-f RAW_FILE] [-s SUFFIX] <run>
#   Events are discovered from the raw_waveform tree (unique `event`
#   values).  Jobs run in parallel capped at PDVD_MAX_JOBS (default 6).
#   Output per event: work/<RUN_PADDED>_light<EVENT><SUFFIX>/

set -e
PDVD_DIR=$(cd "$(dirname "$0")" && pwd)
source "$PDVD_DIR/_runlib.sh"

RAW_FILE=""
SUFFIX=""
FARGS=()
while getopts "f:s:" opt; do
    case $opt in
        f) RAW_FILE="$OPTARG"; FARGS+=(-f "$OPTARG") ;;
        s) SUFFIX="$OPTARG"; FARGS+=(-s "$OPTARG") ;;
        *) echo "usage: $0 [-f RAW_FILE] [-s SUFFIX] <run>" >&2; exit 1 ;;
    esac
done
shift $((OPTIND-1))
if [ $# -lt 1 ]; then echo "usage: $0 [-f RAW_FILE] [-s SUFFIX] <run>" >&2; exit 1; fi

RUN=$1
RUN_PADDED=$(printf "%06d" "$RUN")

if [ -z "$RAW_FILE" ]; then
    mapfile -t CANDS < <(ls "$PDVD_DIR"/input_data_light/np02vd_raw_run${RUN_PADDED}_*_rawwf.root 2>/dev/null)
    if [ ${#CANDS[@]} -eq 0 ]; then
        echo "no input_data_light/np02vd_raw_run${RUN_PADDED}_*_rawwf.root; use -f" >&2
        exit 1
    fi
    if [ ${#CANDS[@]} -gt 1 ]; then
        echo "WARNING: ${#CANDS[@]} stream files match run $RUN, using ${CANDS[0]}" >&2
    fi
    RAW_FILE=${CANDS[0]}
    FARGS+=(-f "$RAW_FILE")
fi

EVENTS=$(python3 -c "
import uproot
t = uproot.open('$RAW_FILE')['raw_waveform']
print(' '.join(str(e) for e in sorted(set(t['event'].array(library='np')))))
")
NEV=$(echo "$EVENTS" | wc -w)
echo "run $RUN: $NEV events in $(basename "$RAW_FILE")"

batch_init
for EV in $EVENTS; do
    batch_wait_slot
    (
        if "$PDVD_DIR/run_light_evt.sh" "${FARGS[@]}" "$RUN" "$EV" \
            > "$PDVD_DIR/work/.light_${RUN_PADDED}_${EV}${SUFFIX}.log" 2>&1; then
            echo "  evt $EV OK"
        else
            echo "  evt $EV FAILED (see work/.light_${RUN_PADDED}_${EV}${SUFFIX}.log)"
        fi
    ) &
done
batch_drain
echo "all done"
