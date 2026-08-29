#!/bin/bash
# doc pr/116 -- serve the REVIEW variant (readable note box) of the hand-scan display.
#
#   ./em_display/serve_em_display.sh [PORT] [--scan-tag NAME] [calib json globs]
#
# PORT defaults to 5021.  5013 img / 5014 pd / 5008-5009 ql_scan / 5016 wf_scan /
# 5017 pr_display / 5018-5020 overclustering are taken, so 5021 is the next free
# one.  (The port table in pr_display/README.md is stale for ql_scan -- the serve
# scripts are authoritative, not that comment.)
#
# With no globs the viewer takes its event list from em114-manifest.tsv, which is
# what keeps a scan reproducible: the sample is a committed file, not whatever
# the shell happened to expand.  Regenerate it with prep_em_scan.py.
#
# From a laptop:
#   ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
#       -L 5021:localhost:5021 <user>@wcgpu1.phy.bnl.gov
#   then open http://localhost:5021/em_display_viewer
#
# The keepalive options are not decoration (doc pr/88): a bare `ssh -L` gets
# reaped by an idle timeout during exactly the long pauses a hand scan is made
# of, and Bokeh's JS does NOT auto-reconnect -- the tab shows "Client connection
# was lost" and keeps showing it after the tunnel is back.
#
# --session-token-expiration is 86400 for the same class of reason: bokeh's
# default is 300 s, a scan session routinely outlives it, and THE SYMPTOM IS A
# HANG -- the page never finishes loading and shows no error, so it reads as
# "the viewer is broken" rather than "reload me".  Safe here because the server
# is bound behind an ssh tunnel, so token lifetime is not a security boundary.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${1:-5021}
shift || true

VIEWER_OPTS=()
REST=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --scan-tag)   VIEWER_OPTS+=(--scan-tag "$2"); shift 2 ;;
        --scan-tag=*) VIEWER_OPTS+=(--scan-tag "${1#*=}"); shift ;;
        --manifest)   VIEWER_OPTS+=(--manifest "$2"); shift 2 ;;
        --prepdir)    VIEWER_OPTS+=(--prepdir "$2"); shift 2 ;;
        *)            REST+=("$1"); shift ;;
    esac
done
set -- "${REST[@]+"${REST[@]}"}"

BOKEH=/nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/bin/bokeh

exec "$BOKEH" serve --port "$PORT" \
    --session-token-expiration "${SESSION_TOKEN_EXPIRATION:-86400}" \
    --allow-websocket-origin="localhost:${PORT}" \
    --allow-websocket-origin="127.0.0.1:${PORT}" \
    --allow-websocket-origin="wcgpu1.phy.bnl.gov:${PORT}" \
    --allow-websocket-origin="wcgpu1:${PORT}" \
    "$HERE/em116_confirm_viewer.py" --args "${VIEWER_OPTS[@]+"${VIEWER_OPTS[@]}"}" "$@"
