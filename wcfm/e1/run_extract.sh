#!/bin/bash
# wcfm E1 (doc 07): run the depo-extraction `lar` job on one WC_FM_Sim pilot G4 file.
# Usage: ./run_extract.sh <flavor: numu|nue> [nevents (default 20)]
# Input:  /home/xqian/work/WC_FM_Sim/runs/<flavor>/g4_<flavor>.root
# Output: wcfm/work/e1/depos-<flavor>.tar.bz2 (depo_data_N/depo_info_N per art event, N 0-based),
#         wcfm/work/e1/extract_<flavor>.log, extract_<flavor>.rc, extract_<flavor>-provenance.txt
# Environment: the SL7 apptainer image + cvmfs stock dunesw v10_20_08d00 (e26:prof); the
# custom larwirecell build of WC_FM_Sim is NOT needed (two-node graph, stock components).
set -u
E1_DIR=$(cd "$(dirname "$0")" && pwd)
WCFM_DIR=$(dirname "$E1_DIR")
FLAVOR=${1:?flavor numu|nue}; NEV=${2:-20}
G4=/home/xqian/work/WC_FM_Sim/runs/${FLAVOR}/g4_${FLAVOR}.root
[ -f "$G4" ] || { echo "no $G4" >&2; exit 1; }
OUTDIR=$WCFM_DIR/work/e1; mkdir -p "$OUTDIR"
RUNDIR=$OUTDIR/run_${FLAVOR}; mkdir -p "$RUNDIR"
OUT=$OUTDIR/depos-${FLAVOR}.tar.bz2
LOG=$OUTDIR/extract_${FLAVOR}.log
IMG=/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-dev-sl7:latest
DUNESW=v10_20_08d00; QUAL=e26:prof

# per-flavour wrapper fcl: only the output name differs
cat > "$RUNDIR/depo_extract_${FLAVOR}.fcl" <<EOT
#include "depo_extract_fdhd.fcl"
physics.producers.wirecell.wcls_main.params.outname: "${OUT}"
services.TFileService.fileName: "${RUNDIR}/depo_extract_${FLAVOR}_hist.root"
EOT

{
  echo "flavor=$FLAVOR nevents=$NEV"
  echo "input=$G4 sha256=$(sha256sum "$G4" | cut -c1-16)"
  echo "image=$IMG -> $(readlink -f "$IMG")"
  echo "dunesw=$DUNESW $QUAL"
  echo "wcp=$(git -C "$WCFM_DIR/.." rev-parse --short HEAD)"
  echo "started=$(date -Is)"
} > "$OUTDIR/extract_${FLAVOR}-provenance.txt"

rm -f "$OUT" "$LOG"
apptainer exec -B /cvmfs,/home/xqian,/nfs --ipc --pid "$IMG" bash -lc "
  source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh >/dev/null 2>&1
  setup dunesw $DUNESW -q $QUAL || exit 90
  export FHICL_FILE_PATH=$E1_DIR:$RUNDIR:\$FHICL_FILE_PATH
  export WIRECELL_PATH=$E1_DIR:\$WIRECELL_PATH
  cd $RUNDIR && lar -n $NEV -c depo_extract_${FLAVOR}.fcl -s $G4
" > "$LOG" 2>&1
rc=$?
echo "rc=$rc" > "$OUTDIR/extract_${FLAVOR}.rc"
echo "finished=$(date -Is) rc=$rc" >> "$OUTDIR/extract_${FLAVOR}-provenance.txt"
echo "extract $FLAVOR rc=$rc -> $OUT"
exit $rc
