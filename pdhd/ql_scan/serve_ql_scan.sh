#!/bin/bash
# Serve the PDHD Q/L matching hand-scan event display over HTTP.
#
# Usage: ./serve_ql_scan.sh [port] [--tag NAME] [calib-glob ...]
#   port        (optional, default 5015; img_plot owns 5013, pd_plot 5014)
#   --tag NAME  (optional) namespace saved scan results into work/ql_labels/NAME/
#               so separate displays keep their labels apart, e.g.
#                 ./serve_ql_scan.sh 5015 --tag data work/*/calib-evt*.json
#   calib-glob  (optional) one or more globs/paths of per-event calib JSONs
#               (default: ../work/*/calib-evt*.json)
#
# Produce the calib JSONs first with the clustering chain + -calib, e.g.
#   ./run_clus_evt.sh -calib <run> all
# which writes work/<run6>_<evt>/calib-evt<ID>.json (one combined file per event, both
# drift sides; the joint QLMatching node tags each entry with its drift side).
#
# To view from a remote laptop, set up SSH port forwarding first:
#   ssh -L 5015:localhost:5015 user@wcgpu1
# then open http://localhost:5015/ql_scan_viewer in the laptop browser.

set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${1:-5015}
shift || true

# Optional --tag NAME (forwarded to the viewer; subdirs the saved labels).
TAG_ARGS=()
if [ "$1" = "--tag" ] || [ "$1" = "-t" ]; then
    TAG_ARGS=(--tag "$2"); shift 2 || true
fi

# Calib JSONs to scan (default: every per-event dump under pdhd/work).
if [ "$#" -gt 0 ]; then
    SPECS=("$@")
else
    SPECS=("$HERE/../work/"*"/calib-evt"*".json")
fi

BOKEH=/nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/bin/bokeh

# The bokeh env carries numpy+bokeh; the viewer needs nothing else (stdlib json).
exec "$BOKEH" serve --port "$PORT" \
    --allow-websocket-origin="localhost:${PORT}" \
    --allow-websocket-origin="127.0.0.1:${PORT}" \
    --allow-websocket-origin="wcgpu1.phy.bnl.gov:${PORT}" \
    --allow-websocket-origin="wcgpu1:${PORT}" \
    "$HERE/ql_scan_viewer.py" --args "${TAG_ARGS[@]}" "${SPECS[@]}"
