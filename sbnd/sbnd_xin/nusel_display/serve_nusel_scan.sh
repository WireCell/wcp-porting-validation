#!/bin/bash
# Serve the SBND nusel (TGM/STM/FC) hand-scan event display over HTTP.
#
# Usage: ./serve_nusel_scan.sh [port] [--tag NAME] [work_root ...]
#   port        (optional, default 5010)
#   --tag NAME  (optional) namespace saved scan results into
#               <work_root>/nusel_labels/NAME/ so different scan campaigns
#               keep their labels apart, e.g.
#                 ./serve_nusel_scan.sh 5010 --tag mcp10 ../work-mcp10
#   work_root   (optional) one or more work roots holding
#               nusel_evt<ID>/nusel-evt<ID>.tsv + ql_evt<ID>/mabc-all-apa.zip
#               + ql_evt<ID>/pctree-evt<ID>.tar.gz
#               (default: ../work-mcp10)
#
# Produce the inputs first with the per-event chain, e.g.
#   SBND_INPUT_DIR=... SBND_WORK_ROOT=$PWD/work-mcp10 ./run_nusel_evt.sh data all
# which leaves nusel_evt<ID>/ (table + PR outputs) and ql_evt<ID>/ (Bee zip,
# pctree) behind.
#
# To view from a remote laptop, set up SSH port forwarding first:
#   ssh -L 5010:localhost:5010 user@workstation
# then open http://localhost:5010/nusel_scan_viewer in the laptop browser.

set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${1:-5010}
shift || true

# Optional --tag NAME (forwarded to the viewer; subdirs the saved labels).
TAG_ARGS=()
if [ "$1" = "--tag" ] || [ "$1" = "-t" ]; then
    TAG_ARGS=(--tag "$2"); shift 2 || true
fi

# Work roots to scan (default: the MCP2025C 10-event sample).
if [ "$#" -gt 0 ]; then
    SPECS=("$@")
else
    SPECS=("$HERE/../work-mcp10")
fi

BOKEH=/nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/bin/bokeh

exec "$BOKEH" serve --port "$PORT" \
    --allow-websocket-origin="localhost:${PORT}" \
    --allow-websocket-origin="127.0.0.1:${PORT}" \
    --allow-websocket-origin="wcgpu1.phy.bnl.gov:${PORT}" \
    --allow-websocket-origin="wcgpu1:${PORT}" \
    "$HERE/nusel_scan_viewer.py" --args "${TAG_ARGS[@]}" "${SPECS[@]}"
