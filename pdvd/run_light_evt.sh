#!/bin/bash
# WCT-native PDVD light reconstruction for one event: all 40 OpDets
# (51 DAPHNE channels) in one processing -- three population branches
# (cathode full-stream / membrane XA / PMT) merged at the OpHit level
# into a single all-PD OpFlashFinder.  See wct-light-reco.jsonnet and
# pdvd/docs/pdvd-light-chain.md.
#
# Usage: ./run_light_evt.sh [-f RAW_FILE] [-s SUFFIX] <run> <event>
#   default RAW_FILE: first match of
#     input_data_light/np02vd_raw_run<PAD>_*_rawwf.root
#     (WARNING printed when several stream files match -- run 039349 has
#      three; pass -f to pick one)
#   -s SUFFIX appends to the work-dir name so alternate configs do not
#     clobber the production archives.
#   Output: work/<RUN_PADDED>_light<EVENT><SUFFIX>/opflash_pdvd-wct.tar.gz
#
# The light<->charge offset is not yet calibrated: offset_us=0 is
# stamped (provisional; flash times are relative to the event's earliest
# light record).

set -e
PDVD_DIR=$(cd "$(dirname "$0")" && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}

RAW_FILE=""
SUFFIX=""
while getopts "f:s:" opt; do
    case $opt in
        f) RAW_FILE="$OPTARG" ;;
        s) SUFFIX="$OPTARG" ;;
        *) echo "usage: $0 [-f RAW_FILE] [-s SUFFIX] <run> <event>" >&2; exit 1 ;;
    esac
done
shift $((OPTIND-1))
if [ $# -lt 2 ]; then echo "usage: $0 [-f RAW_FILE] [-s SUFFIX] <run> <event>" >&2; exit 1; fi

RUN=$1
EVENT=$2
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
fi

WORKDIR=$PDVD_DIR/work/${RUN_PADDED}_light${EVENT}${SUFFIX}
mkdir -p "$WORKDIR"

echo "== run $RUN event $EVENT"
echo "   input:  $RAW_FILE"
echo "   output: $WORKDIR/opflash_pdvd-wct.tar.gz"

wcsonnet \
    -A input_file="$RAW_FILE" \
    -A output_dir="$WORKDIR" \
    -S run="$RUN" \
    -S event="$EVENT" \
    -o "$WORKDIR/.wct-light.json" \
    "$PDVD_DIR/wct-light-reco.jsonnet"

wire-cell -l stderr -l "$WORKDIR/light-reco.log:debug" -L debug -c "$WORKDIR/.wct-light.json"

echo "done:"
ls -l "$WORKDIR/opflash_pdvd-wct.tar.gz"
