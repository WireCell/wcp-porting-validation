#!/bin/bash
# Reconstruct ALL 160 PDHD photon detectors for one event in a SINGLE wire-cell
# processing: the self-trigger snippets (opch 0-119) and the -x full-stream PDs
# (opch 120-159) are reconstructed in two branches that merge at the OpHit level
# (OpHitMerge), so one OpFlashFinder builds flashes spanning all 160 PDs.  The
# -x flashes then cover the whole 80-159 wall (both readout halves) -- the
# product Q/L matching consumes.
#
# Usage: ./run_light_allpd_evt.sh [-f RAW_FILE] [-s SUFFIX] <run> <event>
#   default RAW_FILE: /nfs/data/1/xning/wirecell-working/data/hd/run<PAD>/np04hd_raw_run<PAD>_*.root
#   -s SUFFIX appends to the work-dir name (default empty); e.g. -s _nocut writes to
#             work/<RUN_PADDED>_allpd<EVENT>_nocut so an alternate config (e.g. with the
#             flash quality cut disabled) does not clobber the production archives.
#   Output: work/<RUN_PADDED>_allpd<EVENT><SUFFIX>/opflash_pdhd-allpd-wct.tar.gz
#
# The two inputs are the decoana files produced by fullstream_to_decoana.py
# (ch 0-120 for the snippet branch, ch 120-160 for the full stream).  If the
# per-stream work dirs (work/<run>_snip<evt>, work/<run>_fs<evt>) already carry
# them, they are reused; otherwise they are regenerated here.

set -e
PDHD_DIR=$(cd "$(dirname "$0")" && pwd)
WCT_BASE=/nfs/data/1/xqian/toolkit-dev
export WIRECELL_PATH=${WCT_BASE}/toolkit/cfg:${WCT_BASE}/wire-cell-data:${WIRECELL_PATH}

RAW_FILE=""
SUFFIX=""
while getopts "f:s:" opt; do
    case $opt in
        f) RAW_FILE="$OPTARG" ;;
        s) SUFFIX="$OPTARG" ;;
        *) echo "usage: $0 [-f RAW_FILE] [-s SUFFIX] <run> <event>" >&2; exit 1 ;;
    esac
done
shift $((OPTIND-1))
if [ $# -lt 2 ]; then echo "usage: $0 [-f RAW_FILE] [-s SUFFIX] <run> <event>" >&2; exit 1; fi

RUN=$1
EVENT=$2
RUN_PADDED=$(printf "%06d" "$RUN")
if [ -z "$RAW_FILE" ]; then
    RAW_FILE=$(ls /nfs/data/1/xning/wirecell-working/data/hd/run${RUN_PADDED}/np04hd_raw_run${RUN_PADDED}_*.root 2>/dev/null | head -1)
fi
if [ -z "$RAW_FILE" ] || [ ! -f "$RAW_FILE" ]; then
    echo "ERROR: no raw file for run $RUN (use -f RAW_FILE)" >&2; exit 1
fi

WORKDIR="$PDHD_DIR/work/${RUN_PADDED}_allpd${EVENT}${SUFFIX}"
mkdir -p "$WORKDIR"
echo "raw:    $RAW_FILE"
echo "output: $WORKDIR"

# 1. decoana inputs (reuse the per-stream ones if present, else regenerate).
SNIPCONV="$PDHD_DIR/work/${RUN_PADDED}_snip${EVENT}/snippet_decoana.root"
FSCONV="$PDHD_DIR/work/${RUN_PADDED}_fs${EVENT}/fullstream_decoana.root"
if [ ! -s "$SNIPCONV" ]; then
    SNIPCONV="$WORKDIR/snippet_decoana.root"
    python3 "$PDHD_DIR/pd_plot/fullstream_to_decoana.py" "$RAW_FILE" "$RUN" "$EVENT" "$SNIPCONV" 0 120
fi
if [ ! -s "$FSCONV" ]; then
    FSCONV="$WORKDIR/fullstream_decoana.root"
    python3 "$PDHD_DIR/pd_plot/fullstream_to_decoana.py" "$RAW_FILE" "$RUN" "$EVENT" "$FSCONV"
fi
echo "snip:   $SNIPCONV"
echo "fs:     $FSCONV"

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

# 3. single all-PD light chain (two branches -> OpHitMerge -> OpFlashFinder).
cfg="$WORKDIR/.wct-light-allpd.json"
wcsonnet \
    -A "snip_file=${SNIPCONV}" \
    -A "fs_file=${FSCONV}" \
    -A "output_dir=${WORKDIR}" \
    -S "run=${RUN}" \
    -S "event=${EVENT}" \
    -S "offset_us=${OFFSET_US}" \
    -o "$cfg" \
    "$PDHD_DIR/wct-light-allpd-reco.jsonnet"
# Resource recording (additive; does not change reco output, disable with
# PDHD_RESMON=off): run wire-cell in the background and sample its
# /proc/<pid>/status every 1 s.  Writes a per-event summary
# (light_resource_<run>_<evt>.txt: wall_s + peak_rss_gb) and the full
# RSS-vs-wallclock trace (light_rss_<run>_<evt>.csv).
RES_TXT="$WORKDIR/light_resource_${RUN_PADDED}_${EVENT}${SUFFIX}.txt"
RES_CSV="$WORKDIR/light_rss_${RUN_PADDED}_${EVENT}${SUFFIX}.csv"
_t0=$(date +%s.%N) _smpid=""
wire-cell -l stderr -l "${WORKDIR}/light-allpd.log:debug" -L debug -c "$cfg" &
_wcpid=$!
if [ "${PDHD_RESMON:-on}" != "off" ]; then
    echo "epoch_s,clock,vmrss_kb,vmhwm_kb" > "$RES_CSV"
    ( while kill -0 "$_wcpid" 2>/dev/null; do
        _line=$(awk '/^VmRSS:/{r=$2}/^VmHWM:/{h=$2}END{print r","h}' \
                "/proc/$_wcpid/status" 2>/dev/null)
        [ -n "$_line" ] && echo "$(date +%s),$(date +%H:%M:%S.%2N),$_line" >> "$RES_CSV"
        sleep 1
      done ) &
    _smpid=$!
fi
_rc=0
wait "$_wcpid" || _rc=$?
if [ -n "$_smpid" ]; then
    kill "$_smpid" 2>/dev/null || true
    wait "$_smpid" 2>/dev/null || true
fi
_wall=$(awk -v a="$_t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.2f", b-a}')
_peak_kb=0
[ -f "$RES_CSV" ] && _peak_kb=$(awk -F, 'NR>1 && $4>m{m=$4}END{print m+0}' "$RES_CSV")
awk -v r="$RUN_PADDED" -v e="$EVENT" -v w="$_wall" -v p="$_peak_kb" 'BEGIN{
    printf "run=%s evt=%s wall_s=%s peak_rss_gb=%.2f\n", r,e,w,p/1048576}' | tee "$RES_TXT"
if [ "$_rc" -ne 0 ]; then
    echo "ERROR: wire-cell exited with status $_rc" >&2
    exit "$_rc"
fi

echo "done:"
ls -la "$WORKDIR"/opflash_pdhd-allpd-wct.tar.gz 2>/dev/null
