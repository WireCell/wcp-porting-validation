#!/bin/bash
# Serve the SBND overclustering-separation hand-scan display over HTTP.
#
# Usage: ./serve_overclustering_scan.sh [port] [--tag NAME] [dump-glob ...]
#   port        (optional, default 5018 -- img 5013 / pd 5014 / ql_scan 5015 /
#               wf_scan 5016 / pr_display 5017 are already spoken for)
#   --tag NAME  (optional) namespace saved labels into
#               overclustering_labels/NAME/ instead of overclustering_labels/
#   dump-glob   (optional) one or more globs/paths of per-event
#               oc56scan-evt*.jsonl dumps (default: ../work-pr56r4*-scan*/pr_evt*/oc56scan-evt*.jsonl)
#
# Produce the dumps first (doc pr/57):
#   PR_JOBS=6 SBND_PROTECT_GRAPH=relaxed_strict_img_2d \
#   WCT_RELAXED_EDGE_CENSUS=1 PR_OC56_SCAN_DUMP=1 \
#     ./run_pr_chain_batch.sh <ql_root> <out> data <evt ...>
# which writes <out>/pr_evt<ID>/oc56scan-evt<ID>.jsonl.
#
# To view from a remote laptop, set up SSH port forwarding first:
#   ssh -L 5018:localhost:5018 user@wcgpu1.phy.bnl.gov
# then open http://localhost:5018/overclustering_scan_viewer in the laptop browser.
#
# See ../docs/pr/57_overclustering-separation-scan.md.

set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${1:-5018}
shift || true

TAG_ARGS=()
if [ "$1" = "--tag" ]; then
    TAG_ARGS=(--tag "$2")
    shift 2
fi

if [ "$#" -gt 0 ]; then
    SPECS=("$@")
else
    SPECS=("$HERE/../work-pr56r4"*"-scan"*"/pr_evt"*"/oc56scan-evt"*".jsonl")
fi

BOKEH=/nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/bin/bokeh

exec "$BOKEH" serve --port "$PORT" \
    --allow-websocket-origin="localhost:${PORT}" \
    --allow-websocket-origin="127.0.0.1:${PORT}" \
    --allow-websocket-origin="wcgpu1.phy.bnl.gov:${PORT}" \
    --allow-websocket-origin="wcgpu1:${PORT}" \
    "$HERE/overclustering_scan_viewer.py" --args "${TAG_ARGS[@]}" "${SPECS[@]}"
