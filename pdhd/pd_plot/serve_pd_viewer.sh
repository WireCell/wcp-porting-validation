#!/bin/bash
# Serve the PDHD light (photon-detector) event display over HTTP for remote
# browser viewing.
#
# Usage: ./serve_pd_viewer.sh [port] [workdir] [opdet-geom.json]
#   port             (optional, default 5014; img_plot owns 5013)
#   workdir          (optional) directory scanned for <RUN6>_<EVT> events,
#                    default /nfs/data/1/xqian/toolkit-dev/toolkit/pdhd/work
#   opdet-geom.json  (optional) OpDet geometry, default the cfg/pdhd file
#
# To view from a remote laptop, set up SSH port forwarding first:
#   ssh -L 5014:localhost:5014 user@wcgpu1
# then open http://localhost:5014/pd_viewer in the laptop browser.

set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${1:-5014}
WORKDIR=${2:-/nfs/data/1/xqian/toolkit-dev/toolkit/pdhd/work}
GEOM=${3:-/nfs/data/1/xqian/toolkit-dev/toolkit/cfg/pgrapher/experiment/pdhd/pdhd-opdet-geom.json}

BOKEH=/nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/bin/bokeh

# The bokeh env has numpy+bokeh; the wirecell python package lives in a sibling
# tree.  Prepend both so the bokeh-env Python can find them.
export PYTHONPATH=/nfs/data/1/xqian/toolkit-dev/local/lib/python3.11/site-packages:/nfs/data/1/xqian/toolkit-dev/wire-cell-python${PYTHONPATH:+:$PYTHONPATH}

exec "$BOKEH" serve --port "$PORT" \
    --allow-websocket-origin="localhost:${PORT}" \
    --allow-websocket-origin="127.0.0.1:${PORT}" \
    --allow-websocket-origin="wcgpu1.phy.bnl.gov:${PORT}" \
    --allow-websocket-origin="wcgpu1:${PORT}" \
    "$HERE/pd_viewer.py" --args "$WORKDIR" "$GEOM"
