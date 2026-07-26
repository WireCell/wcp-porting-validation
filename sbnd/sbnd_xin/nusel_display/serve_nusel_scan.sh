#!/bin/bash
# Serve the SBND nusel (TGM/STM/FC) hand-scan event display over HTTP.
#
# Usage: ./serve_nusel_scan.sh [port] [--tag NAME] [--prev ROOT[:TAG] ...] [work_root ...]
#   port        (optional, default 5010; must come first if given)
#   --tag NAME  (optional) namespace saved scan results into
#               <work_root>/nusel_labels/NAME/ so different scan campaigns
#               keep their labels apart, e.g.
#                 ./serve_nusel_scan.sh 5010 --tag mcp10 ../work-mcp10
#   --charge-src ql|pr  (optional, default ql = historical) where the drawn
#               charge comes from.  "ql" = the Q/L job's mabc-all-apa.zip, which
#               is PRE-un-merge, so a merged bundle is drawn under a trajectory
#               fitted on the POST-un-merge main (doc 50 Q1 -- this is what made
#               fits look like they crossed empty space).  "pr" = the nusel job's
#               own nusel_evt<ID>/mabc-pr.zip, i.e. exactly the cluster the
#               taggers saw.  Light is read from the Q/L zip either way.
#   --ai-scan FILE.tsv  (optional, repeatable) overlay somebody else's already-
#               made per-bundle verdicts as an extra READ-ONLY "AI scan" column
#               + reason line (doc 61).  Nothing is written to any
#               nusel_labels/ tree from it, so the owner's own scan record stays
#               separate.  TSV columns: event main_id flash_gid verdict
#               [conf] [reason]; '#' lines skipped, e.g.
#                 ./serve_nusel_scan.sh 5011 --tag s59k \
#                   --ai-scan ../scan-d59k/handscan-first20.tsv ...
#   --prev ROOT[:TAG]  (optional, repeatable, priority = given order) an
#               OLDER work root and the label tag scanned there, used as a
#               read-only baseline: its scan labels/comments are carried
#               into the current tag, and bundles whose tgm/stm/fc verdicts
#               changed are tinted amber until re-scanned, e.g.
#                 ./serve_nusel_scan.sh 5010 --tag mcp10-merge \
#                   --prev ../work-mcp10:mcp10 ../work-mcp10-merge2
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

# Optional leading port, then any mix of --tag NAME / --prev SPEC / work roots
# (--tag and --prev are forwarded to the viewer).
PORT=5010
case "$1" in ''|*[!0-9]*) : ;; *) PORT=$1; shift ;; esac
FWD_ARGS=()
SPECS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --tag|-t) FWD_ARGS+=(--tag "$2"); shift 2 ;;
        --prev)   FWD_ARGS+=(--prev "$2"); shift 2 ;;
        --charge-src) FWD_ARGS+=(--charge-src "$2"); shift 2 ;;
        --ai-scan) FWD_ARGS+=(--ai-scan "$2"); shift 2 ;;
        *)        SPECS+=("$1"); shift ;;
    esac
done

# Work roots to scan (default: the MCP2025C 10-event sample).
if [ "${#SPECS[@]}" -eq 0 ]; then
    SPECS=("$HERE/../work-mcp10")
fi

BOKEH=/nfs/data/1/xqian/toolkit-dev/.direnv/python-3.11.9/bin/bokeh

exec "$BOKEH" serve --port "$PORT" \
    --allow-websocket-origin="localhost:${PORT}" \
    --allow-websocket-origin="127.0.0.1:${PORT}" \
    --allow-websocket-origin="wcgpu1.phy.bnl.gov:${PORT}" \
    --allow-websocket-origin="wcgpu1:${PORT}" \
    "$HERE/nusel_scan_viewer.py" --args "${FWD_ARGS[@]}" "${SPECS[@]}"
