#!/bin/bash
# Serve the SBND Q/L matching hand-scan event display over HTTP.
#
# Usage: ./serve_ql_scan.sh [port] [--tag NAME] [calib-glob ...]
#   port        (optional, default 5008)
#   --tag NAME  (optional) namespace saved scan results into work/ql_labels/NAME/
#               so the MC and data displays keep their labels apart, e.g.
#                 ./serve_ql_scan.sh 5008 --tag mc   work/ql_evt{2,9,...}/calib-*.json
#                 ./serve_ql_scan.sh 5009 --tag data work/ql_evt{686,...}/calib-*.json
#   calib-glob  (optional) one or more globs/paths of per-event calib JSONs
#               (default: ../work/ql_evt*/calib-evt*.json)
#
# Produce the calib JSONs first with the per-event chain + -calib, e.g.
#   ./run_img_evt.sh mc all && ./run_ql_evt.sh mc all -calib
# which writes work/ql_evt<ID>/calib-evt<ID>.json (both TPCs per file).
#
# To view from a remote laptop, set up SSH port forwarding first:
#   ssh -L 5008:localhost:5008 user@workstation
# then open http://localhost:5008/ql_scan_viewer in the laptop browser.

set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${1:-5008}
shift || true

# Optional --tag NAME (forwarded to the viewer; subdirs the saved labels).
TAG_ARGS=()
if [ "$1" = "--tag" ] || [ "$1" = "-t" ]; then
    TAG_ARGS=(--tag "$2"); shift 2 || true
fi

# Calib JSONs to scan (default: every per-event dump under sbnd_xin/work).
if [ "$#" -gt 0 ]; then
    SPECS=("$@")
else
    # .json.gz too: the 2026-07-24 consolidation gzipped archived dumps.
    SPECS=("$HERE/../work/ql_evt"*"/calib-evt"*".json" \
           "$HERE/../work/ql_evt"*"/calib-evt"*".json.gz")
fi

BOKEH=/nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/bin/bokeh

# The bokeh env carries numpy+bokeh; the viewer needs nothing else (stdlib json).
exec "$BOKEH" serve --port "$PORT" \
    --allow-websocket-origin="localhost:${PORT}" \
    --allow-websocket-origin="127.0.0.1:${PORT}" \
    --allow-websocket-origin="wcgpu1.phy.bnl.gov:${PORT}" \
    --allow-websocket-origin="wcgpu1:${PORT}" \
    "$HERE/ql_scan_viewer.py" --args "${TAG_ARGS[@]}" "${SPECS[@]}"
