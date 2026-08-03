#!/bin/bash
# Imaging + clustering + QLMatching over the input-10files-mc baseline.
# Prereq: ./scripts/analysis/misc/build_mcbase_stage.py  (remaps the 10 files to unique ids under
# $MCBASE_STAGE).  Each file's events run in parallel internally (run_*_evt.sh
# 'all' caps at nproc); files run in sequence.  Calib + cathode-diag dumps feed
# cathode_distortion.py.  Uses the SBND_INPUT_DIR override in _runlib.sh.
set -u
SBND_DIR=$(cd "$(dirname "$0")"/../.. && pwd)
cd "$SBND_DIR"
STAGE=${MCBASE_STAGE:-/home/xqian/tmp/mcbase_in}

for n in $(seq 1 10); do
    export SBND_INPUT_DIR="$STAGE/$n"
    echo "############### FILE $n : imaging ###############"
    ./run_img_evt.sh mc all 2>&1 | tail -4
    echo "############### FILE $n : Q/L matching (+calib +cathode-diag) ###############"
    ./run_ql_evt.sh mc all -calib -cathode-diag 2>&1 | tail -4
done

echo "############### DONE ###############"
echo "MC baseline calib JSONs (id>=700000): $(ls work/ql_evt7*/calib-evt7*.json 2>/dev/null | wc -l)"
echo "MC baseline QLCATHODE logs:           $(grep -l QLCATHODE work/ql_evt7*/*.log 2>/dev/null | wc -l)"
