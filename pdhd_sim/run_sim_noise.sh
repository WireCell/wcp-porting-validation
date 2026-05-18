#!/bin/bash
# Run electronics-noise-only wire-cell simulation for ProtoDUNE-HD.
# Wraps wct-sim-noise-only.jsonnet.
#
# Usage:
#   ./run_sim_noise.sh            # all 4 APAs in one wire-cell run
#   ./run_sim_noise.sh -a 2       # only APA 2
#   ./run_sim_noise.sh -g 7.8     # low-gain noise spectra (default 14 mV/fC)
#
# Output: work/noise/<all|anode<N>>/pdhd-noise-sim-anode<N>.tar.bz2

set -e

PDHD_SIM_DIR=$(cd "$(dirname "$0")" && pwd)

WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${PDHD_SIM_DIR}:${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}

ANODE=""
ELEC_GAIN="14"
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) echo "Usage: $0 [-a anode] [-g elecGain]"; exit 0 ;;
        -a) ANODE="$2"; shift 2 ;;
        -g) ELEC_GAIN="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -n "$ANODE" ]; then
    INDICES="[${ANODE}]"
    TAG="anode${ANODE}"
else
    INDICES="[0,1,2,3]"
    TAG="all"
fi

OUTDIR="$PDHD_SIM_DIR/work/noise/${TAG}"
mkdir -p "$OUTDIR"
PREFIX="$OUTDIR/pdhd-noise-sim"
LOG="$OUTDIR/wct.log"

echo "=== PDHD noise-only sim  APAs=${INDICES}  elecGain=${ELEC_GAIN} mV/fC ==="
echo "  output : ${PREFIX}-anode<N>.tar.bz2"
echo "  log    : $LOG"

cd "$PDHD_SIM_DIR"
wire-cell \
    -l stderr \
    -l "${LOG}:debug" \
    -L debug \
    -V "elecGain=${ELEC_GAIN}" \
    --tla-str  "output_prefix=${PREFIX}" \
    --tla-code "anode_indices=${INDICES}" \
    -c wct-sim-noise-only.jsonnet

echo "Done. Outputs in ${OUTDIR}"
