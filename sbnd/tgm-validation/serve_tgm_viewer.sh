#!/bin/bash
# Serve the TGM tagged-track 3-panel viewer over HTTP (Bokeh server).
#
# Usage: ./serve_tgm_viewer.sh [port] [tgm_points.npz]
#   port  default 5011
#   npz   default tgm-validation/tgm_points.npz (relative to repo root)
#
# Runs inside the SL7 container (bokeh from the wire-cell-python venv via
# setup-local-opt.sh).  To view from a laptop, tunnel first:
#   ssh -L 5011:localhost:5011 <user>@sbndbuild03.fnal.gov
# then open http://localhost:5011/tgm_viewer
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${1:-5011}
NPZ=${2:-$HERE/tgm_points.npz}

exec /cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer exec \
    -B /cvmfs,/exp,/nashome,/pnfs \
    /cvmfs/singularity.opensciencegrid.org/fermilab/fnal-dev-sl7:latest \
    bash -c "
source /nashome/y/yuhw/.bashrc
source /exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd/setup-local-opt.sh
cd $HERE
bokeh serve --port $PORT \
    --allow-websocket-origin=localhost:$PORT \
    --allow-websocket-origin=127.0.0.1:$PORT \
    tgm_viewer.py --args $NPZ
"
