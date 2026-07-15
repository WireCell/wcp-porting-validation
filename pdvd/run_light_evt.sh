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
# Per-event light<->charge offsets are measured from the rawwf trigoff
# tree (per CRATE: the TDE/BDE charge windows open up to ~32 us apart,
# each jittering ~+-15 us vs the trigger while the light full-stream
# window is trigger-locked to +-0.3 us) and stamped into the archive
# metadata as offset_bot_us/offset_top_us:
#     offset = chain_light_t0 - charge_window_start   (us, negative)
# where chain_light_t0 replicates PDVDOpWaveformSource's tick origin
# (min over ALL records of ts - 64 samples for self-trigger snippets).
# ADD the offset to a flash time to land on that crate's charge axis.
# The legacy offset_us key stays 0 (inert).

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

# Per-crate light<->charge offsets from the trigoff tree in the SAME rawwf
# file (see header comment).  Hard error if the tree/event row is missing --
# archives without offsets are useless for Q/L matching.
OFFSETS=$(python3 - "$RAW_FILE" "$RUN" "$EVENT" <<'PY'
import sys, uproot, numpy as np
fn, run, event = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
PERIOD_US = (1 << 40) * 0.016          # 40-bit DTS rollover period in us (~17592.19 s)
def wrap(d):                           # difference -> [-P/2, P/2)
    return (d + PERIOD_US / 2) % PERIOD_US - PERIOD_US / 2
f = uproot.open(fn)
try:
    to = f["trigoff/trigger_offset"].arrays(library="np")
except Exception:
    sys.exit("ERROR: no trigoff/trigger_offset tree in %s" % fn)
sel = (to["run"] == run) & (to["event"] == event)
if not sel.any():
    sys.exit("ERROR: no trigoff row for run %d event %d in %s" % (run, event, fn))
i = int(np.flatnonzero(sel)[0])
# Chain tick origin: replicate PDVDOpWaveformSource exactly -- min over ALL
# records of round(ts*62.5) - 64 ticks for self-trigger snippets (nsamp<=1024).
try:
    rw = f["rawdump/raw_waveform"]
except Exception:
    rw = f["raw_waveform"]
a = rw.arrays(["run", "event", "nsamp", "timestamp"], library="np")
m = (a["run"] == run) & (a["event"] == event)
ticks = np.round(a["timestamp"][m] * 62.5) - np.where(a["nsamp"][m] <= 1024, 64, 0)
t0_us = ticks.min() * 0.016
d_tree = t0_us - float(to["light_t0_us"][i])   # 0 or -1.024 (snippet earliest)
if not -2.0 < d_tree <= 0.0:
    sys.exit("ERROR: chain t0 %.3f vs tree light_t0 %.3f (d=%.3f us)"
             % (t0_us, to["light_t0_us"][i], d_tree))
bot = wrap(t0_us - float(to["charge_bde_us"][i]))
top = wrap(t0_us - float(to["charge_tde_us"][i]))
print("%.6f %.6f %.3f" % (bot, top, d_tree))
PY
)
read -r OFFSET_BOT_US OFFSET_TOP_US T0_VS_TREE <<< "$OFFSETS"
echo "   offsets (us, add to flash time): bot=$OFFSET_BOT_US top=$OFFSET_TOP_US (chain t0 - tree light_t0 = $T0_VS_TREE us)"

# PDVD_VETO_SATURATION=0 disables the DAPHNE-rail OpHit veto (see
# docs/qlmatch/pdvd-pd-mapping-investigation.md sec.6-7: the veto zeroes
# railed cathode channels in 42% of bright flashes).  Default (unset/1) =
# legacy veto ON, compiled config byte-identical.
# PDVD_FLAG_SATURATION=1 keeps-and-marks railed hits (10th ophit column ->
# flash_sat tensor -> QLMatching per-flash mask) and PDVD_SAT_REPAIR=1
# additionally bridges railed runs before decon -- the keep-and-mark chain of
# docs/qlmatch/pdvd-saturation-recovery.md.  Operating point for the _satrep
# reprocess: PDVD_VETO_SATURATION=0 PDVD_FLAG_SATURATION=1 PDVD_SAT_REPAIR=1.
# All default off, compiled config byte-identical.
VETO_SAT_ARG=()
if [ "${PDVD_VETO_SATURATION:-1}" = 0 ]; then
    VETO_SAT_ARG=(-S "veto_saturation=false")
fi
if [ "${PDVD_FLAG_SATURATION:-0}" = 1 ]; then
    VETO_SAT_ARG+=(-S "flag_saturation=true")
fi
if [ "${PDVD_SAT_REPAIR:-0}" = 1 ]; then
    VETO_SAT_ARG+=(-S "saturation_repair=true")
fi

wcsonnet \
    -A input_file="$RAW_FILE" \
    -A output_dir="$WORKDIR" \
    -S run="$RUN" \
    -S event="$EVENT" \
    -S offset_bot_us="$OFFSET_BOT_US" \
    -S offset_top_us="$OFFSET_TOP_US" \
    "${VETO_SAT_ARG[@]}" \
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
