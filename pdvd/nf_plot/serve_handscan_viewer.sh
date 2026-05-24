#!/bin/bash
# Serve the L1SP hand-scan viewer over HTTP for remote browser review.
#
# Usage: ./serve_handscan_viewer.sh <handscan_dir> [port]
#   handscan_dir  Directory written by code/inference/score_full_corpus.py.
#                 Must contain handscan_config.json, scores_full.csv,
#                 and scan_pool_round1.csv.
#   port          (optional, default 5008)
#
# To view from a remote laptop, set up SSH port forwarding first:
#   ssh -L 5008:localhost:5008 user@workstation
# then open http://localhost:5008/handscan_viewer in the laptop's browser.

set -e
HERE=$(cd "$(dirname "$0")" && pwd)

if [ $# -lt 1 ]; then
    echo "Usage: $0 <handscan_dir> [port]" >&2
    exit 1
fi

DIR=$1
PORT=${2:-5008}

case "$DIR" in
    /*) ABS="$DIR" ;;
    *)  ABS="$(cd "$DIR" && pwd)" ;;
esac

exec bokeh serve --port "$PORT" \
    --allow-websocket-origin="localhost:${PORT}" \
    --allow-websocket-origin="127.0.0.1:${PORT}" \
    "$HERE/handscan_viewer.py" --args "$ABS"
