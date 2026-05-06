#!/bin/bash
# Serve the PDVD shield coupling NF diagnostic Bokeh viewer.
#
# Usage: ./serve_shield_tune_viewer.sh <dump_dir> [<magnify_dir>] [port]
#
#   dump_dir     Directory containing shield_dump_ch*.npz files written by
#                PDVDShieldCouplingSub when dump_path is set in nf.jsonnet.
#   magnify_dir  Directory containing magnify-*anode*.root files (optional).
#                Supplies the hu_raw reference and hu_orig pre-NF view.
#   port         Bokeh server port (default: 5006).
#
# Quick-start for run 039324 event 0 (after rerunning NF with dump enabled):
#
#   DUMP=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work/039324_0/shield_dumps
#   MAG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work/039324_0
#   ./serve_shield_tune_viewer.sh "$DUMP" "$MAG"
#
# To enable the dump: pass shield_dump_path to run_nf_sp_evt.sh, e.g.:
#
#   DUMP=.../pdvd/work/039324_0/shield_dumps
#   run_nf_sp_evt.sh -R ... --tla shield_dump_path="$DUMP"
#
# Or temporarily set it in nf.jsonnet by changing:
#   dump_path: shield_dump_path,
# to:
#   dump_path: "/your/absolute/path/shield_dumps",
#
# Remote access:
#   ssh -L 5006:localhost:5006 user@wcgpu1.phy.bnl.gov
#   open http://localhost:5006/shield_tune_viewer in your laptop browser.

set -e
HERE=$(cd "$(dirname "$0")" && pwd)

if [ $# -lt 1 ]; then
    echo "Usage: $0 <dump_dir> [<magnify_dir>] [port]" >&2
    exit 1
fi

DUMP_DIR=$1
MAGNIFY_DIR=${2:-""}
PORT=${3:-5006}

BOKEH=/nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/bin/bokeh

export PYTHONPATH=/nfs/data/1/xqian/toolkit-dev/local/lib/python3.11/site-packages${PYTHONPATH:+:$PYTHONPATH}

exec "$BOKEH" serve --port "$PORT" \
    --allow-websocket-origin="localhost:${PORT}" \
    --allow-websocket-origin="127.0.0.1:${PORT}" \
    --allow-websocket-origin="wcgpu1.phy.bnl.gov:${PORT}" \
    --allow-websocket-origin="wcgpu1:${PORT}" \
    "$HERE/shield_tune_viewer.py" --args "$DUMP_DIR" "$MAGNIFY_DIR"
