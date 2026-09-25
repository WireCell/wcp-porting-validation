#!/bin/bash
# Runs INSIDE the SL7 apptainer.  Light input for one kaon art file (doc pdvd/118):
#  1. PDVDTriggerLightAna (jjo's pdsoffset mrb area) over the art file -> *_triglight.root
#  2. jjo's pdvd_dump_rawwf.py: art OpDetWaveforms + triglight -> *_rawwf.root
#     (rawdump/raw_waveform + trigoff/trigger_offset, the run_light_evt.sh input)
# Usage: _light_inner.sh <kaon.root> <outdir> <tag>
# no set -e: the ups setup scripts return non-zero harmlessly
IN=$1; OUT=$2; TAG=$3
KDIR=$(cd "$(dirname "$0")" && pwd)
source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh > /dev/null 2>&1
setup dunesw v10_21_00d00 -q e26:prof
source /nfs/data/1/jjo/tmp/pdvd_flash_validation/vddev/localProducts_dune_v10_21_00d00_e26_prof/setup
mrbslp
export FHICL_FILE_PATH=$KDIR:$FHICL_FILE_PATH
mkdir -p "$OUT"; cd "$OUT"
set +e
lar -c pdvd_triglight_rootinput.fcl -s "$IN" -T ${TAG}_triglight.root > ${TAG}_triglight.log 2>&1; rc=$?
echo "lar rc=$rc"; [ $rc -eq 0 ] || exit $rc
python /home/jjo/Work/protodune-pds-light/pdvd/scripts/pdvd_dump_rawwf.py "$IN" ${TAG}_rawwf.root \
    --max-events -1 --triglight ${TAG}_triglight.root > ${TAG}_dump.log 2>&1; rc=$?
echo "dump rc=$rc"; exit $rc
