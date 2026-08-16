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
# To view from a remote laptop, set up SSH port forwarding first.  Use the
# keepalive options -- doc pr/88: a bare `ssh -L` gets reaped by a NAT or
# firewall idle timeout during exactly the pauses a hand scan is made of (one
# event studied for a few minutes sends no bytes), and when the ssh session
# dies the tunnel dies with it.  Bokeh's JS does NOT auto-reconnect: the tab
# shows "Client connection was lost" and keeps showing it after the tunnel is
# back, so it reads as a server fault when the server never noticed.  This cost
# real confusion once during the pr/88 scan; `last` on the server (ssh session
# end time) against the bokeh log (WebSocket closed) identifies it in seconds.
#   ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
#       -L 5017:localhost:5017 user@wcgpu1.phy.bnl.gov
# then open http://localhost:5017/pr_display_viewer in the laptop browser.
# After any tunnel drop, RELOAD the tab -- restarting the server does not help
# and costs you every other open session's token (see below).
#
# See ../docs/pr/26_pr-event-display.md.

set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${1:-5017}
shift || true

# doc sbnd_xin/docs/pr/75: --scan-tag <name> selects the neutrino-vertex
# hand-scan label set (../vertex_labels/<name>/).  Consumed here and passed
# through; everything else stays a calib-JSON glob.
VIEWER_OPTS=()
REST=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --scan-tag) VIEWER_OPTS+=(--scan-tag "$2"); shift 2 ;;
        --scan-tag=*) VIEWER_OPTS+=(--scan-tag "${1#*=}"); shift ;;
        --wire-planes) VIEWER_OPTS+=(--wire-planes); shift ;;
        *) REST+=("$1"); shift ;;
    esac
done
set -- "${REST[@]+"${REST[@]}"}"

if [ "$#" -gt 0 ]; then
    SPECS=("$@")
else
    SPECS=("$HERE/../work-prdisp-"*"/pr_evt"*"/calib-pr-evt"*".json")
fi

BOKEH=/nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/bin/bokeh

# The bokeh env carries numpy+bokeh; the viewer needs nothing else (stdlib
# json).  In particular it does NOT need uproot or ROOT: everything it draws
# comes from the one calib JSON.
# --session-token-expiration: doc pr/88.  Bokeh's default is 300 s, and a
# hand-scan session routinely outlives that -- the owner leaves the tab open
# between instalments, or the server is restarted with a new event list under
# a tab that still holds the old token.  Either way the reconnect fails with
#
#     bokeh.protocol.exceptions.ProtocolError: Token is expired.
#
# and the SYMPTOM IS A HANG: the page never finishes loading and shows no
# error, so it reads as "the viewer is broken" rather than "reload me".  This
# cost real confusion twice during the pr/88 scan.  86400 s = one day, which
# is longer than any scanning session and costs nothing here: the server is
# bound to localhost behind an ssh tunnel, so token lifetime is not a security
# boundary.  Override with SESSION_TOKEN_EXPIRATION if you ever need to.
exec "$BOKEH" serve --port "$PORT" \
    --session-token-expiration "${SESSION_TOKEN_EXPIRATION:-86400}" \
    --allow-websocket-origin="localhost:${PORT}" \
    --allow-websocket-origin="127.0.0.1:${PORT}" \
    --allow-websocket-origin="wcgpu1.phy.bnl.gov:${PORT}" \
    --allow-websocket-origin="wcgpu1:${PORT}" \
    "$HERE/pr_display_viewer.py" --args "${VIEWER_OPTS[@]+"${VIEWER_OPTS[@]}"}" "${SPECS[@]}"
