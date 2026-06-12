#!/bin/bash
# Run electronics-noise-only wire-cell simulation for ProtoDUNE-VD.
# Wraps wct-sim-noise-only.jsonnet.
#
# Usage:
#   ./run_sim_noise.sh            # all anodes 0..7 in one wire-cell run
#   ./run_sim_noise.sh -a 2       # only anode 2
#
# Output: work/noise/<all|anode<N>>/pdvd-noise-sim-anode<N>.tar.bz2

set -e

PDVD_SIM_DIR=$(cd "$(dirname "$0")" && pwd)

WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${PDVD_SIM_DIR}:${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}

ANODE=""
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) echo "Usage: $0 [-a anode]"; exit 0 ;;
        -a) ANODE="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -n "$ANODE" ]; then
    INDICES="[${ANODE}]"
    TAG="anode${ANODE}"
else
    INDICES="[0,1,2,3,4,5,6,7]"
    TAG="all"
fi

OUTDIR="$PDVD_SIM_DIR/work/noise/${TAG}"
mkdir -p "$OUTDIR"
PREFIX="$OUTDIR/pdvd-noise-sim"
LOG="$OUTDIR/wct.log"

echo "=== PDVD noise-only sim  anodes=${INDICES} ==="
echo "  output : ${PREFIX}-anode<N>.tar.bz2"
echo "  log    : $LOG"

cd "$PDVD_SIM_DIR"
wire-cell \
    -l stderr \
    -l "${LOG}:debug" \
    -L debug \
    --tla-str  "output_prefix=${PREFIX}" \
    --tla-code "anode_indices=${INDICES}" \
    -c wct-sim-noise-only.jsonnet

echo "Done. Outputs in ${OUTDIR}"
