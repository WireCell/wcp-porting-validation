#!/bin/bash
# Run the uboone-mabc job for one filelist entry in an isolated scratch dir.
#
# Usage: ./run_one.sh <idx> <label>
#   idx    0-based line index into ../filelist (matches run_5384.pl's
#          initial_index / mabc_<idx>.zip naming)
#   label  -> sweep/<label>/<idx>_<EV>/{meta.txt,mabc_<idx>.zip,
#             track_com_5384_<EV>.root,wct_5384_<EV>.log,stdout.log}
#
# The job runs inside the per-event dir (track_com_*.root is cwd-relative in
# uboone-mabc.jsonnet and would otherwise clobber the qlport outputs) with
# rootfiles/ and uboone_track_fitting.json symlinked in.  Command line is the
# exact production one from run_5384.pl; timecmd.py records wall_s and peak
# RSS (getrusage CHILDREN == VmHWM).
#
# ASLR is disabled (setarch -R) by default: the pattern-recognition stage is
# pointer-order dependent (track_fit/vertices/shower_track outputs vary
# run-to-run with ASLR on), and the byte-identity A/B gate needs stable
# addresses.  Set ASLR=1 to run with normal randomization.
#
# The DL (SCN) vertex is disabled by default (-A dl_weights=): the gate3
# reference was produced while the python env was broken (SCN import failed,
# silent geometric fallback), so byte-identity gates are defined DL-off —
# and SCN/GPU inference is not guaranteed bit-stable anyway.  Set
# DL_WEIGHTS=uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth for a DL-on
# (functional, non-gate) run.
#
# Prototype is_dir_weak() semantics (dir_weak_use_score knob, doc
# sbnd_xin/docs/pr/6) are DEFAULT ON (owner 2026-07-30).  Set DIR_WEAK=false
# for legacy raw-flag reads — required for byte-identity checks against the
# knob-OFF baselines (gate3, dirweakoff_ub).  The knob-ON reference sweep is
# sweep/dirweakon_ub.
set -eu
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
QLPORT=$(dirname "$SCRIPTS")
ABTEST=$(cd "$QLPORT/../abtest" && pwd)

IDX=${1:?usage: run_one.sh <idx> <label>}
LABEL=${2:?usage: run_one.sh <idx> <label>}

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

cd "$DEST"
rm -f "wct_${RUN}_${EV}.log" "mabc_${IDX}.zip" "track_com_${RUN}_${EV}.root"
python3 "$ABTEST/timecmd.py" meta.txt \
    $NOASLR wire-cell -l stderr -l "wct_${RUN}_${EV}.log:debug" -L clus:debug \
    -A kind=both -A "beezip=mabc_${IDX}.zip" -A "initial_index=$IDX" \
    -A "initial_runNo=$RUN" -A "initial_subRunNo=$SR" -A "initial_eventNo=$EV" \
    -A "dl_weights=${DL_WEIGHTS:-}" \
    -A "dir_weak_use_score=${DIR_WEAK:-true}" \
    -A "fit_exclusion=${QL_FIT_EXCLUSION:-false}" \
    -A "dqdx_fit_keep_all_points=${QL_DQDX_KEEP_ALL:-false}" \
    -A "infiles=$FILE" "$QLPORT/uboone-mabc.jsonnet" \
    > stdout.log 2>&1 || true

NODE_S=$(sed -nE 's/.*Total node execution : ([0-9.]+) sec.*/\1/p' "wct_${RUN}_${EV}.log" | tail -1)
echo "node_exec_s=${NODE_S:-nan}" >> meta.txt
echo "run=$RUN subrun=$SR ev=$EV" >> meta.txt
awk -F= '/^(rc|wall_s|maxrss_kb|node_exec_s)=/{printf "%s ", $0}' meta.txt
echo "[idx=$IDX ev=$EV]"
