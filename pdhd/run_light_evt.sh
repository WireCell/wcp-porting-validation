#!/bin/bash
# Convert and/or reconstruct one event of PDHD light data (temporary ROOT format).
# Usage: ./run_light_evt.sh [-f FILE] [-m convert|reco|both] <run> <evt>
#        ./run_light_evt.sh            # list available runs/files
#
#   Input:  example_light_data/np04hd_raw_run<RUN0>*.root (or -f FILE)
#   Output (convert): work/<RUN_PADDED>_<EVT>/opflash_pdhd.tar.gz
#                     work/<RUN_PADDED>_<EVT>/light-frames.tar.bz2
#   Output (reco):    work/<RUN_PADDED>_<EVT>/opflash_pdhd-wct.tar.gz
#                     work/<RUN_PADDED>_<EVT>/light-frames-wct.tar.bz2
#
# See toolkit flash/docs/design.md for product schemas and conventions.

set -e

PDHD_DIR=$(cd "$(dirname "$0")" && pwd)

WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}

LIGHT_DIR="$PDHD_DIR/input_data_7p8_new_coh_grouping"

INPUT_FILE=""
MODE="both"
DAQ_EVT=""   # DAQ event number in the ROOT file; defaults to the positional <evt>
while getopts "f:m:e:" opt; do
    case $opt in
        f) INPUT_FILE="$OPTARG" ;;
        m) MODE="$OPTARG" ;;
        e) DAQ_EVT="$OPTARG" ;;
        *) echo "usage: $0 [-f FILE] [-m convert|reco|both] [-e DAQ_EVT] <run> <evt>" >&2; exit 1 ;;
    esac
done
shift $((OPTIND-1))

if [ $# -lt 2 ]; then
    echo "Available light-data runs:"
    ls -d "$LIGHT_DIR"/run*/ 2>/dev/null | sed 's|.*/run|  run|;s|/$||'
    echo "usage: $0 [-f FILE] [-e DAQ_EVT] <run> <evt>"
    exit 0
fi

RUN=$1
EVT=$2                       # label for the work dir (0-based index in this dataset)
EVENT_NUM=${DAQ_EVT:-$EVT}   # event number selected from the ROOT file
RUN_PADDED=$(printf "%06d" "$RUN")

if [ -z "$INPUT_FILE" ]; then
    # Prefer the multi-event np04hd_raw file for the run (per-run subdir).
    INPUT_FILE=$(ls "$LIGHT_DIR"/run${RUN_PADDED}/np04hd_raw_run${RUN_PADDED}_*.root 2>/dev/null | head -1)
    if [ -z "$INPUT_FILE" ]; then
        INPUT_FILE=$(ls "$LIGHT_DIR"/run${RUN_PADDED}/*run${RUN}*.root 2>/dev/null | head -1)
    fi
fi
if [ -z "$INPUT_FILE" ] || [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: no light-data file for run $RUN in $LIGHT_DIR (use -f FILE)" >&2
    exit 1
fi

WORKDIR="$PDHD_DIR/work/${RUN_PADDED}_${EVT}"
mkdir -p "$WORKDIR"

echo "input:  $INPUT_FILE"
echo "output: $WORKDIR  (mode: $MODE)"

# Per-event readout-vs-trigger offset (us) = (tc - rd_timestamp)*16ns from the ROOT
# trigoff tree, selecting the candidate nearest the nominal 250us (same rule as the
# C++ PDHDLight::read_trigger).  The WCT-native reco can't recover this from the
# self-triggered snippets, so we read it here and pass it to OpFlashFinder, which
# stamps it into the opflash metadata for downstream Q/L matching.
OFFSET_US=$(python3 - "$INPUT_FILE" "$RUN" "$EVENT_NUM" <<'PY' || echo 0)
import sys, uproot, numpy as np
fn, run, event = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
off = 0.0
try:
    to = uproot.open(fn)["trigoff/trigger_offset"].arrays(library="np")
    sel = (to["run"] == run) & (to["event"] == event)
    if sel.any():
        i = np.argmin(np.abs(to["offset_us"][sel] - 250.0))
        off = float(to["offset_us"][sel][i])
except Exception:
    off = 0.0
print(off)
PY
echo "trigger offset_us: $OFFSET_US"

run_one() {
    local job=$1
    local cfg="$WORKDIR/.wct-light-${job}.json"
    # Only the WCT-native reco stamps the offset; convert (PDHDOpFlashSource) reads
    # the trigger itself, so its jsonnet has no offset_us parameter.
    local extra=()
    [ "$job" = reco ] && extra=(-S "offset_us=${OFFSET_US}")
    wcsonnet \
        -A "input_file=${INPUT_FILE}" \
        -A "output_dir=${WORKDIR}" \
        -S "run=${RUN}" \
        -S "event=${EVENT_NUM}" \
        "${extra[@]}" \
        -o "$cfg" \
        "$PDHD_DIR/wct-light-${job}.jsonnet"
    wire-cell -l stderr -l "${WORKDIR}/light-${job}.log:debug" -L debug -c "$cfg"
}

case $MODE in
    convert) run_one convert ;;
    reco)    run_one reco ;;
    both)    run_one convert; run_one reco ;;
    *) echo "ERROR: unknown mode $MODE" >&2; exit 1 ;;
esac

echo "done:"
ls -la "$WORKDIR"/opflash_pdhd*.tar.gz "$WORKDIR"/light-frames*.tar.bz2 2>/dev/null
