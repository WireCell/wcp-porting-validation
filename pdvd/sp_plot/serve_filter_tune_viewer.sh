#!/bin/bash
# Serve the SP filter-tuning Bokeh viewer over HTTP for remote browser viewing.
#
# Usage: ./serve_filter_tune_viewer.sh [port]
#   port  (optional, default 5007)
#
# Bundled defaults cover all PDHD APAs (0-3, evt0 of run 27409) and all
# PDVD anodes (0-7, evt0 of run 39324: 0-3 = bottom CRP, 4-7 = top CRP).
# Edit the SPECS array below to point at different magnify ROOT files.
#
# To view from a remote laptop, set up SSH port forwarding first:
#   ssh -L 5007:localhost:5007 user@workstation
# then open http://localhost:5007/filter_tune_viewer in the laptop browser.

set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${1:-5007}

PDHD_DIR=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/work/027409_0
PDVD_DIR=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work/039324_0

# Each spec: <label>|<path>|<ident>|<detector>
SPECS=(
    "PDHD APA0|${PDHD_DIR}/magnify-run027409-evt0-apa0.root|0|pdhd"
    "PDHD APA1|${PDHD_DIR}/magnify-run027409-evt0-apa1.root|1|pdhd"
    "PDHD APA2|${PDHD_DIR}/magnify-run027409-evt0-apa2.root|2|pdhd"
    "PDHD APA3|${PDHD_DIR}/magnify-run027409-evt0-apa3.root|3|pdhd"
    "PDVD bot anode0|${PDVD_DIR}/magnify-run039324-evt0-anode0.root|0|pdvd"
    "PDVD bot anode1|${PDVD_DIR}/magnify-run039324-evt0-anode1.root|1|pdvd"
    "PDVD bot anode2|${PDVD_DIR}/magnify-run039324-evt0-anode2.root|2|pdvd"
    "PDVD bot anode3|${PDVD_DIR}/magnify-run039324-evt0-anode3.root|3|pdvd"
    "PDVD top anode4|${PDVD_DIR}/magnify-run039324-evt0-anode4.root|4|pdvd"
    "PDVD top anode5|${PDVD_DIR}/magnify-run039324-evt0-anode5.root|5|pdvd"
    "PDVD top anode6|${PDVD_DIR}/magnify-run039324-evt0-anode6.root|6|pdvd"
    "PDVD top anode7|${PDVD_DIR}/magnify-run039324-evt0-anode7.root|7|pdvd"
)

BOKEH=/nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/bin/bokeh

# The bokeh env has numpy+bokeh; uproot+awkward live in the sibling 'local' env.
# Prepend the local env's site-packages so the bokeh-env Python can find them.
export PYTHONPATH=/nfs/data/1/xqian/toolkit-dev/local/lib/python3.11/site-packages${PYTHONPATH:+:$PYTHONPATH}

exec "$BOKEH" serve --port "$PORT" \
    --allow-websocket-origin="localhost:${PORT}" \
    --allow-websocket-origin="127.0.0.1:${PORT}" \
    "$HERE/filter_tune_viewer.py" --args "${SPECS[@]}"
