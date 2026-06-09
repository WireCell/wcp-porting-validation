#!/bin/bash
# Serve the PDVD imaging event display over HTTP for remote browser viewing.
#
# Usage: ./serve_img_viewer.sh [port] [evt.npz] [magnify-template]
#   port              (optional, default 5012)
#   evt.npz           (optional, default cache/evt0.npz)
#   magnify-template  (optional) path with '{anode}', for the waveform views, e.g.
#                     /.../039324_0/magnify-run039324-evt0-anode{anode}.root
#
# Build the artifact first:
#   ./preprocess_event.py            # writes cache/evt0.npz (+ .json sidecar)
#
# To view from a remote laptop, set up SSH port forwarding first:
#   ssh -L 5012:localhost:5012 user@wcgpu1
# then open http://localhost:5012/img_viewer in the laptop browser.

set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${1:-5012}
NPZ=${2:-$HERE/cache/evt0.npz}
# NOTE: keep the '{anode}' template out of any ${...:-default} expansion -- the '}'
# in '{anode}' prematurely closes the parameter expansion and corrupts the path.
if [ -n "${3:-}" ]; then
    MAGNIFY="$3"
else
    MAGNIFY='/nfs/data/1/xqian/toolkit-dev/toolkit/pdvd/work/039324_0/magnify-run039324-evt0-anode{anode}-dnnroi.root'
fi

BOKEH=/nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/bin/bokeh

# The bokeh env has numpy+bokeh; uproot/awkward and the wirecell python package
# live in sibling trees.  Prepend both so the bokeh-env Python can find them.
export PYTHONPATH=/nfs/data/1/xqian/toolkit-dev/local/lib/python3.11/site-packages:/nfs/data/1/xqian/toolkit-dev/wire-cell-python${PYTHONPATH:+:$PYTHONPATH}

exec "$BOKEH" serve --port "$PORT" \
    --allow-websocket-origin="localhost:${PORT}" \
    --allow-websocket-origin="127.0.0.1:${PORT}" \
    --allow-websocket-origin="wcgpu1.phy.bnl.gov:${PORT}" \
    --allow-websocket-origin="wcgpu1:${PORT}" \
    "$HERE/img_viewer.py" --args "$NPZ" "$MAGNIFY"
