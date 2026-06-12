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

LIGHT_DIR="$PDHD_DIR/example_light_data"

INPUT_FILE=""
MODE="both"
while getopts "f:m:" opt; do
    case $opt in
        f) INPUT_FILE="$OPTARG" ;;
        m) MODE="$OPTARG" ;;
        *) echo "usage: $0 [-f FILE] [-m convert|reco|both] <run> <evt>" >&2; exit 1 ;;
    esac
done
shift $((OPTIND-1))

if [ $# -lt 2 ]; then
    echo "Available light-data files:"
    ls "$LIGHT_DIR"/*.root 2>/dev/null | sed 's|.*/|  |'
    echo "usage: $0 [-f FILE] <run> <evt>"
    exit 0
fi

RUN=$1
EVT=$2
RUN_PADDED=$(printf "%06d" "$RUN")

if [ -z "$INPUT_FILE" ]; then
    # Prefer the multi-event np04hd_raw file for the run.
    INPUT_FILE=$(ls "$LIGHT_DIR"/np04hd_raw_run${RUN_PADDED}_*.root 2>/dev/null | head -1)
    if [ -z "$INPUT_FILE" ]; then
        INPUT_FILE=$(ls "$LIGHT_DIR"/*run${RUN}*.root 2>/dev/null | head -1)
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

run_one() {
    local job=$1
    local cfg="$WORKDIR/.wct-light-${job}.json"
    wcsonnet \
        -A "input_file=${INPUT_FILE}" \
        -A "output_dir=${WORKDIR}" \
        -S "run=${RUN}" \
        -S "event=${EVT}" \
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
