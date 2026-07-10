#!/bin/bash
# Serve the PDVD Q/L matching hand-scan event display over HTTP.
#
# Usage: ./serve_ql_scan.sh [port] [--tag NAME] [calib-glob ...]
#   port        (optional, default 5016; img_plot owns 5013, pd_plot 5014,
#               pdhd ql_scan 5015)
#   --tag NAME  (optional) namespace saved scan results into work/ql_labels/NAME/
#               so separate displays keep their labels apart, e.g.
#                 ./serve_ql_scan.sh 5016 --tag data work/*/calib-evt*.json
#   calib-glob  (optional) one or more globs/paths of per-event calib JSONs
#               (default: ../work/*/calib-evt*.json)
#
# Produce the calib JSONs first with the clustering chain + -calib, e.g.
#   ./run_clus_evt.sh -calib <run> all
# which writes work/<run6>_<idx>/calib-evt<ID>.json (one file per event; PDVD
# has a single all-PD flash list shared by both drift volumes, the joint
# QLMatching node tags each bundle with its volume, apa 0=bottom / 4=top).
#
# To view from a remote laptop, set up SSH port forwarding first:
#   ssh -L 5016:localhost:5016 user@wcgpu1
# then open http://localhost:5016/ql_scan_viewer in the laptop browser.

set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${1:-5016}
shift || true

# Optional --tag NAME (forwarded to the viewer; subdirs the saved labels).
TAG_ARGS=()
if [ "$1" = "--tag" ] || [ "$1" = "-t" ]; then
    TAG_ARGS=(--tag "$2"); shift 2 || true
fi

# Calib JSONs to scan (default: every per-event dump under pdvd/work).
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
