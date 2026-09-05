#!/bin/bash
# Serve the PDHD stopping-muon hand-scan display (doc stm-tagger-chain sec 13).
#
# Forked BY DUPLICATION from pdhd/ql_scan/serve_ql_scan.sh; that script is untouched.
#
# Usage: ./serve_stm_scan.sh [port] [--tag NAME]
#   port        (optional, default 5017; img_plot owns 5013, pd_plot 5014,
#               ql_scan 5015, wf_scan / pdvd ql_scan 5016)
#   --tag NAME  (optional, default 'retile0') namespaces saved labels into
#               work/stm_scan_labels/NAME/labels.json so separate passes
#               (e.g. a second scanner, or a re-scan) keep their labels apart.
#
# Labels are written on EVERY click -- a reload or a restart loses nothing --
# to work/stm_scan_labels/<tag>/labels.json, a sibling of the per-event
# work/<run6>_<evt>/ dirs so re-running an arm cannot delete them.
#
# To view from a laptop, forward the port first:
#   ssh -L 5017:localhost:5017 user@wcgpu1
# then open http://localhost:5017/stm_scan_viewer
#
# Score the labels afterwards (this reads the answer key; the viewer never does):
#   python3 score_stm_scan.py

set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${1:-5017}
shift || true

TAG_ARGS=()
if [ "$1" = "--tag" ] || [ "$1" = "-t" ]; then
    TAG_ARGS=(--tag "$2"); shift 2 || true
fi

BOKEH=/nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/bin/bokeh

exec "$BOKEH" serve --port "$PORT" \
    --allow-websocket-origin="localhost:${PORT}" \
    --allow-websocket-origin="127.0.0.1:${PORT}" \
    --allow-websocket-origin="wcgpu1.phy.bnl.gov:${PORT}" \
    --allow-websocket-origin="wcgpu1:${PORT}" \
    "$HERE/stm_scan_viewer.py" --args "${TAG_ARGS[@]}"
