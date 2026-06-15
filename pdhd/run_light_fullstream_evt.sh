#!/bin/bash
# Reconstruct the PDHD -x FULL-STREAM PDs (opch 120-159) for one event:
#   1. convert the continuous rawdump/raw_waveform stream into the decoana
#      snippet layout (fullstream_to_decoana.py),
#   2. run the WCT-native light chain on it (wct-light-fullstream-reco.jsonnet),
#      OpDecon at samples=343808 with the fixed Wiener filter.
#
# Usage: ./run_light_fullstream_evt.sh [-f RAW_FILE] <run> <event>
#   default RAW_FILE: /nfs/data/1/xning/wirecell-working/data/hd/run<PAD>/np04hd_raw_run<PAD>_*.root
#   Output: work/<RUN_PADDED>_fs<EVENT>/opflash_pdhd-fullstream-wct.tar.gz
#           work/<RUN_PADDED>_fs<EVENT>/light-frames-fullstream-wct.tar.bz2

set -e
PDHD_DIR=$(cd "$(dirname "$0")" && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}

RAW_FILE=""
while getopts "f:" opt; do
    case $opt in
        f) RAW_FILE="$OPTARG" ;;
        *) echo "usage: $0 [-f RAW_FILE] <run> <event>" >&2; exit 1 ;;
    esac
done
shift $((OPTIND-1))
if [ $# -lt 2 ]; then echo "usage: $0 [-f RAW_FILE] <run> <event>" >&2; exit 1; fi

RUN=$1
EVENT=$2
RUN_PADDED=$(printf "%06d" "$RUN")
if [ -z "$RAW_FILE" ]; then
    RAW_FILE=$(ls /nfs/data/1/xning/wirecell-working/data/hd/run${RUN_PADDED}/np04hd_raw_run${RUN_PADDED}_*.root 2>/dev/null | head -1)
fi
if [ -z "$RAW_FILE" ] || [ ! -f "$RAW_FILE" ]; then
    echo "ERROR: no raw file for run $RUN (use -f RAW_FILE)" >&2; exit 1
fi

WORKDIR="$PDHD_DIR/work/${RUN_PADDED}_fs${EVENT}"
mkdir -p "$WORKDIR"
CONV="$WORKDIR/fullstream_decoana.root"

echo "raw:    $RAW_FILE"
echo "output: $WORKDIR"

# 1. convert the full-stream waveforms into the decoana layout.
python3 "$PDHD_DIR/pd_plot/fullstream_to_decoana.py" "$RAW_FILE" "$RUN" "$EVENT" "$CONV"

# 2. per-event readout-vs-trigger offset (same rule as run_light_evt.sh).
OFFSET_US=$(python3 - "$RAW_FILE" "$RUN" "$EVENT" <<'PY' || echo 0)
import sys, uproot, numpy as np
fn, run, event = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
off = 0.0
try:
    to = uproot.open(fn)["trigoff/trigger_offset"].arrays(library="np")
    sel = (to["run"] == run) & (to["event"] == event)
    if sel.any():
        i = np.argmin(np.abs(to["offset_us"][sel] - 250.0))
        off = float(to["offset_us"][sel][i])
except Exception:
    off = 0.0
print(off)
PY
echo "trigger offset_us: $OFFSET_US"

# 3. run the WCT-native full-stream light chain.
cfg="$WORKDIR/.wct-light-fullstream.json"
wcsonnet \
    -A "input_file=${CONV}" \
    -A "output_dir=${WORKDIR}" \
    -S "run=${RUN}" \
    -S "event=${EVENT}" \
    -S "offset_us=${OFFSET_US}" \
    -o "$cfg" \
    "$PDHD_DIR/wct-light-fullstream-reco.jsonnet"
wire-cell -l stderr -l "${WORKDIR}/light-fullstream.log:debug" -L debug -c "$cfg"

# 4. self-trigger baseline for the comparison: reconstruct ALL 0-119 self-trigger
# snippets directly from the raw stream (the LArSoft decoana container is missing
# the -x snippets for most events, so we reconstruct them from rawdump uniformly,
# same chain + same fixed filter).  -> work/<RUN>_snip<EVENT>/opflash_pdhd-wct.tar.gz
SNIPDIR="$PDHD_DIR/work/${RUN_PADDED}_snip${EVENT}"
mkdir -p "$SNIPDIR"
SNIPCONV="$SNIPDIR/snippet_decoana.root"
python3 "$PDHD_DIR/pd_plot/fullstream_to_decoana.py" "$RAW_FILE" "$RUN" "$EVENT" "$SNIPCONV" 0 120
scfg="$SNIPDIR/.wct-light-reco.json"
wcsonnet \
    -A "input_file=${SNIPCONV}" \
    -A "output_dir=${SNIPDIR}" \
    -S "run=${RUN}" \
    -S "event=${EVENT}" \
    -S "offset_us=${OFFSET_US}" \
    -o "$scfg" \
    "$PDHD_DIR/wct-light-reco.jsonnet"
wire-cell -l stderr -l "${SNIPDIR}/light-reco.log:debug" -L debug -c "$scfg"

echo "done:"
ls -la "$WORKDIR"/opflash_pdhd-fullstream-wct.tar.gz "$SNIPDIR"/opflash_pdhd-wct.tar.gz 2>/dev/null
