#!/bin/bash
# WCT-native PDVD light reconstruction for one event: all 40 OpDets
# (51 DAPHNE channels) in one processing -- three population branches
# (cathode full-stream / membrane XA / PMT) merged at the OpHit level
# into a single all-PD OpFlashFinder.  See wct-light-reco.jsonnet and
# pdvd/docs/pdvd-light-chain.md.
#
# Usage: ./run_light_evt.sh [-f RAW_FILE] [-s SUFFIX] <run> <event>
#   default RAW_FILE: first match of
#     input_data_light/np02vd_raw_run<PAD>_*_rawwf.root
#     (WARNING printed when several stream files match -- run 039349 has
#      three; pass -f to pick one)
#   -s SUFFIX appends to the work-dir name so alternate configs do not
#     clobber the production archives.
#   Output: work/<RUN_PADDED>_light<EVENT><SUFFIX>/opflash_pdvd-wct.tar.gz
#
# The light<->charge offset is not yet calibrated: offset_us=0 is
# stamped (provisional; flash times are relative to the event's earliest
# light record).

set -e
PDVD_DIR=$(cd "$(dirname "$0")" && pwd)
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
    mapfile -t CANDS < <(ls "$PDVD_DIR"/input_data_light/np02vd_raw_run${RUN_PADDED}_*_rawwf.root 2>/dev/null)
    if [ ${#CANDS[@]} -eq 0 ]; then
        echo "no input_data_light/np02vd_raw_run${RUN_PADDED}_*_rawwf.root; use -f" >&2
        exit 1
    fi
    if [ ${#CANDS[@]} -gt 1 ]; then
        echo "WARNING: ${#CANDS[@]} stream files match run $RUN, using ${CANDS[0]}" >&2
    fi
    RAW_FILE=${CANDS[0]}
fi

WORKDIR=$PDVD_DIR/work/${RUN_PADDED}_light${EVENT}${SUFFIX}
mkdir -p "$WORKDIR"

echo "== run $RUN event $EVENT"
echo "   input:  $RAW_FILE"
echo "   output: $WORKDIR/opflash_pdvd-wct.tar.gz"

wcsonnet \
    -A input_file="$RAW_FILE" \
    -A output_dir="$WORKDIR" \
    -S run="$RUN" \
    -S event="$EVENT" \
    -o "$WORKDIR/.wct-light.json" \
    "$PDVD_DIR/wct-light-reco.jsonnet"

# Resource recording (additive; does not change reco output, disable with
# PDVD_RESMON=off): run wire-cell in the background and sample its
# /proc/<pid>/status every 1 s.  Writes a per-event summary
# (light_resource_<run>_<evt>.txt: wall_s + peak_rss_gb) and the full
# RSS-vs-wallclock trace (light_rss_<run>_<evt>.csv).
RES_TXT="$WORKDIR/light_resource_${RUN_PADDED}_${EVENT}${SUFFIX}.txt"
RES_CSV="$WORKDIR/light_rss_${RUN_PADDED}_${EVENT}${SUFFIX}.csv"
_t0=$(date +%s.%N) _smpid=""
wire-cell -l stderr -l "$WORKDIR/light-reco.log:debug" -L debug -c "$WORKDIR/.wct-light.json" &
_wcpid=$!
if [ "${PDVD_RESMON:-on}" != "off" ]; then
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
ls -l "$WORKDIR/opflash_pdvd-wct.tar.gz"
