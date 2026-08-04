#!/bin/bash
# Serve the SBND pattern-recognition (PR) event display over HTTP.
#
# Usage: ./serve_pr_display.sh [port] [calib-glob ...]
#   port        (optional, default 5017 -- img 5013 / pd 5014 / ql_scan 5015 /
#               wf_scan 5016 are already spoken for)
#   calib-glob  (optional) one or more globs/paths of per-event PR calib JSONs
#               (default: ../work-prdisp-*/pr_evt*/calib-pr-evt*.json)
#
# Produce the JSONs first with the PR chain's pr_display stage:
#   PR_EXTRA_STAGES=pr_display ./run_pr_chain_batch.sh <ql_root> <out> data 388
# which writes <out>/pr_evt<ID>/calib-pr-evt<ID>.json.
#
# To view from a remote laptop, set up SSH port forwarding first:
#   ssh -L 5017:localhost:5017 user@wcgpu1.phy.bnl.gov
# then open http://localhost:5017/pr_display_viewer in the laptop browser.
#
# See ../docs/pr/26_pr-event-display.md.

set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${1:-5017}
shift || true

if [ "$#" -gt 0 ]; then
    SPECS=("$@")
else
    SPECS=("$HERE/../work-prdisp-"*"/pr_evt"*"/calib-pr-evt"*".json")
fi

BOKEH=/nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/bin/bokeh

# The bokeh env carries numpy+bokeh; the viewer needs nothing else (stdlib
# json).  In particular it does NOT need uproot or ROOT: everything it draws
# comes from the one calib JSON.
exec "$BOKEH" serve --port "$PORT" \
    --allow-websocket-origin="localhost:${PORT}" \
    --allow-websocket-origin="127.0.0.1:${PORT}" \
    --allow-websocket-origin="wcgpu1.phy.bnl.gov:${PORT}" \
    --allow-websocket-origin="wcgpu1:${PORT}" \
    "$HERE/pr_display_viewer.py" --args "${SPECS[@]}"
