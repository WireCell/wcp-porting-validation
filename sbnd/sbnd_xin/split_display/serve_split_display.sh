#!/bin/bash
# doc pr/138 Phase A -- serve the shower SPLIT scan tool.
#
#   ./split_display/serve_split_display.sh [PORT] [--scan-tag NAME] [--owner-only]
#
# PORT defaults to 5022.  Taken already: 5013 img, 5014 pd, 5008-5009 ql_scan,
# 5016 wf_scan, 5017 pr_display, 5018-5020 overclustering, 5021 em_display.
#
# From a laptop:
#   ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
#       -L 5022:localhost:5022 <user>@wcgpu1.phy.bnl.gov
#   then open http://localhost:5022/split_viewer
#
# The keepalive options and --session-token-expiration are NOT decoration; they
# are the same two traps em_display's serve script documents.  A bare `ssh -L`
# is reaped by an idle timeout during exactly the long pauses a hand scan is made
# of, and bokeh's JS does not auto-reconnect.  Bokeh's default 300 s session
# token expires mid-scan and THE SYMPTOM IS A HANG with no error, which reads as
# "the tool is broken" rather than "reload me".
#
# --scan-tag names a FRESH label dir (CLAUDE.md M13).  The viewer refuses to
# write into a directory holding labels it did not create.
#
#   owner's 50:  ./split_display/serve_split_display.sh 5022 \
#                    --scan-tag splitscan-0901-owner --owner-only
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${1:-5022}
shift || true

BOKEH=/nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/bin/bokeh

exec "$BOKEH" serve --port "$PORT" \
    --session-token-expiration "${SESSION_TOKEN_EXPIRATION:-86400}" \
    --allow-websocket-origin="localhost:${PORT}" \
    --allow-websocket-origin="127.0.0.1:${PORT}" \
    --allow-websocket-origin="wcgpu1.phy.bnl.gov:${PORT}" \
    --allow-websocket-origin="wcgpu1:${PORT}" \
    "$HERE/split_viewer.py" --args "$@"
