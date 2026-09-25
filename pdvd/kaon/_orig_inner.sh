#!/bin/bash
# Runs INSIDE the SL7 apptainer.  Decode-only orig-frame dump (doc pdvd/118):
# RootInput on one single-event kaon art file (it already carries
# tpcrawdecoder:daq), release WCT, cfgmin/wcls-orig-frames-pdvd.jsonnet.
# Usage: _orig_inner.sh <kaon.root> <outdir>
# no set -e: the ups setup scripts return non-zero harmlessly
IN=$1; OUT=$2
KDIR=$(cd "$(dirname "$0")" && pwd)
source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup dunesw ${KAON_DUNESW:-v10_20_08d00} -q e26:prof
export WIRECELL_PATH=$KDIR/cfgmin:$WIRECELL_PATH   # release WCT + release cfg
# (no local WCT libs: release WCT only)
export FHICL_FILE_PATH=$KDIR:/nfs/data/1/xning/container/dune_data:$FHICL_FILE_PATH
mkdir -p "$OUT"; cd "$OUT"
set +e; lar -n 1 -c wcls_origframes_pdvd.fcl "$IN" > lar_tpc.log 2>&1; rc=$?
echo "lar rc=$rc"; exit $rc
