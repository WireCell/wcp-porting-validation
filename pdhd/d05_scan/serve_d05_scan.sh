#!/bin/bash
# Serve the PDHD wrapped-channel MOVER hand-scan display (doc pdhd/05).
#
# Forked BY DUPLICATION from pdhd/stm_scan/serve_stm_scan.sh; that script is untouched.
#
# Usage: ./serve_d05_scan.sh [port] [--tag NAME]
#   port        (optional, default 5017; img_plot owns 5013, pd_plot 5014,
#               ql_scan 5015, wf_scan / pdvd ql_scan 5016).
#               5017 is SHARED with pdhd/stm_scan -- run one or the other.
#   --tag NAME  (optional, default 'movers0') namespaces saved labels into
#               work/d05_scan_labels/NAME/ so separate passes (a second scanner,
#               or a re-scan) keep their labels apart.
#
# Labels are written on EVERY click -- a reload or a restart loses nothing --
# to work/d05_scan_labels/<tag>/labels.json, a sibling of the per-event
# work/<run6>_<evt>_<arm>/ dirs so re-running an arm cannot delete them.  The
# same click refreshes work/d05_scan_labels/<tag>/filled_sheet.tsv; the
# committed blind sheet is never written to.
#
# To view from a laptop, forward the port first:
#   ssh -L 5017:localhost:5017 user@wcgpu1
# then open http://localhost:5017/d05_scan_viewer
#
# Score afterwards with the PRE-REGISTERED bar (doc pdhd/04 sec 11.5).  This
# directory does not re-implement it:
#   python3 docs/scripts/d04_movers_score.py \
#       --sheet work/d05_scan_labels/<tag>/filled_sheet.tsv \
#       --key   bee-pr-run029107-d04movers.KEY.tsv

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
    "$HERE/d05_scan_viewer.py" --args "${TAG_ARGS[@]}"
