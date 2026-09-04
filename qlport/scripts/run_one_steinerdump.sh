#!/bin/bash
# doc pdvd/37: run the uboone-mabc MEASUREMENT COPY for one filelist entry, with
# the PR calib dump attached, in an isolated scratch dir.
#
# Usage: ./run_one_steinerdump.sh <idx> <label>
#   idx    0-based line index into ../filelist (matches mabc_<idx>.zip naming)
#   label  -> sweep/<label>/<idx>_<EV>/{meta.txt,mabc_<idx>.zip,calib-pr-evt<EV>.json,
#             track_com_5384_<EV>.root,wct_5384_<EV>.log,stdout.log}
#
# Everything not named below is IDENTICAL to run_one.sh, deliberately: the
# comparison against the April production Bee zips is only meaningful if the
# command line is the production one.  What differs:
#   * the jsonnet is uboone-mabc-steinerdump.jsonnet, whose compiled config at
#     default TLAs is byte-identical to uboone-mabc.jsonnet (verified);
#   * -A prdump=<path> attaches PrDisplayDump (read-only, one JSON out);
#   * ST_WIRE_TOL / ST_ADJ_SLICE expose the doc pr/29 D1/D12 terminal-filter
#     knobs.  Defaults 0/false = uBooNE production.  doc pdvd/37 also runs
#     1/true, which is what PDVD and SBND both run.
#
# LD_LIBRARY_PATH is PINNED to a snapshot: this is a shared tree and a peer's
# `wcbuild` mid-arm silently swaps the binary under you.  Set LIB_PIN= to opt out.
set -eu
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
QLPORT=$(dirname "$SCRIPTS")
ABTEST=$(cd "$QLPORT/../abtest" && pwd)

IDX=${1:?usage: run_one_steinerdump.sh <idx> <label>}
LABEL=${2:?usage: run_one_steinerdump.sh <idx> <label>}

FILE=$(sed -n "$((IDX+1))p" "$QLPORT/filelist")
[ -n "$FILE" ] || { echo "no filelist line for idx=$IDX" >&2; exit 1; }
[[ "$FILE" =~ nuselEval_([0-9]+)_([0-9]+)_([0-9]+)\.root ]] \
    || { echo "unparseable filelist line: $FILE" >&2; exit 1; }
RUN=${BASH_REMATCH[1]} SR=${BASH_REMATCH[2]} EV=${BASH_REMATCH[3]}

DEST="$SCRIPTS/sweep/$LABEL/${IDX}_${EV}"
mkdir -p "$DEST"
ln -sfn "$QLPORT/rootfiles" "$DEST/rootfiles"
ln -sfn "$QLPORT/uboone_track_fitting.json" "$DEST/uboone_track_fitting.json"

NOASLR="setarch $(uname -m) -R"
[ "${ASLR:-0}" = "1" ] && NOASLR=""

LIB_PIN=${LIB_PIN-/home/xqian/tmp/doc37/lib_pin}
if [ -n "$LIB_PIN" ]; then export LD_LIBRARY_PATH="$LIB_PIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi

cd "$DEST"
rm -f "wct_${RUN}_${EV}.log" "mabc_${IDX}.zip" "track_com_${RUN}_${EV}.root" "calib-pr-evt${EV}.json"
python3 "$ABTEST/timecmd.py" meta.txt \
    $NOASLR wire-cell -l stderr -l "wct_${RUN}_${EV}.log:debug" -L clus:debug \
    -A kind=both -A "beezip=mabc_${IDX}.zip" -A "initial_index=$IDX" \
    -A "initial_runNo=$RUN" -A "initial_subRunNo=$SR" -A "initial_eventNo=$EV" \
    -A "dl_weights=${DL_WEIGHTS:-}" \
    -A "dir_weak_use_score=${DIR_WEAK:-true}" \
    -A "fit_exclusion=${QL_FIT_EXCLUSION:-false}" \
    -A "dqdx_fit_keep_all_points=${QL_DQDX_KEEP_ALL:-false}" \
    -A "steiner_terminal_wire_tol=${ST_WIRE_TOL:-0}" \
    -A "steiner_terminal_adjacent_slice=${ST_ADJ_SLICE:-false}" \
    -A "prdump=calib-pr-evt${EV}.json" \
    -A "infiles=$FILE" "$QLPORT/uboone-mabc-steinerdump.jsonnet" \
    > stdout.log 2>&1 || true

NODE_S=$(sed -nE 's/.*Total node execution : ([0-9.]+) sec.*/\1/p' "wct_${RUN}_${EV}.log" | tail -1)
echo "node_exec_s=${NODE_S:-nan}" >> meta.txt
echo "run=$RUN subrun=$SR ev=$EV" >> meta.txt
echo "st_wire_tol=${ST_WIRE_TOL:-0} st_adj_slice=${ST_ADJ_SLICE:-false}" >> meta.txt
NTERM=$(sed -nE 's/.*steiner: ([0-9]+) cluster\(s\), ([0-9]+) point\(s\), ([0-9]+) terminal\(s\).*/\1 \2 \3/p' "wct_${RUN}_${EV}.log" | tail -1)
echo "steiner_clusters_points_terminals=${NTERM:-nan}" >> meta.txt
awk -F= '/^(rc|wall_s|maxrss_kb|node_exec_s)=/{printf "%s ", $0}' meta.txt
echo "[idx=$IDX ev=$EV]"
